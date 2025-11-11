
from hydra.utils import instantiate
from typing import List, Optional, Dict
from pathlib import Path
import copy
import numpy as np
import torch
from torch import nn
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from navsim.evaluate.pdm_score import pdm_score
from navsim.agents.diffusiondrive.modules.blocks import (
    gen_sineembed_for_position)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.common.dataclasses import Trajectory
from navsim.agents.diffusion_trajectory_model import (
    norm_odo, denorm_odo, calculate_component_losses, calculate_statistics)
from navsim.agents.diffusion_trajectory_model_v2 import DiffusionTrajectoryHeadv2
from navsim.agents.fiery.fiery_config import FieryConfig
import random; random.seed(42)


def return_decay(step, decay_type):
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001
        uphold = 0.5
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.999
    elif decay_type == 3:
        step = 0.9
        flat = 0
        uprate = 1
        uphold = 0.9
    else:
        assert False

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)


class DiffusionNFTTrajectoryHead(DiffusionTrajectoryHeadv2):
    """Diffusion-based trajectory model for NFT agent."""

    def __init__(
        self,
        config: FieryConfig,
        num_samples: int = 64,
        mini_batch_size: int = 512,
        beta: float = 1.0,
        Zc: float = 2.0,
        use_adaptive_weighted_policy_losses: bool = False,
        use_kl_div_loss: bool = False,
        simulator: Optional[PDMSimulator] = None,
        scorer: Optional[PDMScorer] = None,
        scene_filter: Optional[SceneFilter] = None,
        target_type: str = "x0", # "x0" or "velocity"
    ):
        """
        Initializes Diffusion-based trajectory model for NFT agent.
        :param config: global config of Fiery agent
        """
        super().__init__(
            config=config,
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )
        assert self.scheduler_type == "flow_matching_euler", "DiffusionNFTTrajectoryModel only supports flow_matching_euler scheduler."

        self._Zc = Zc
        self._num_samples = num_samples
        self.mini_batch_size = mini_batch_size
        self.beta = beta
        self.use_adaptive_weighted_policy_losses = use_adaptive_weighted_policy_losses
        self.simulator = simulator
        self.scorer = scorer
        self.scene_filter = scene_filter
        self.use_kl_div_loss = use_kl_div_loss
        self.decay_type = config.decay_type
        self.target_type = target_type
    
    def set_simulator(self, simulator: PDMSimulator, scorer: PDMScorer, scene_filter: SceneFilter):
        self.simulator = simulator
        self.scorer = scorer
        self.scene_filter = scene_filter
        self._initialize_pdm_scorer()

    def create_reference_modules(self):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = next(self.parameters()).device
        # create a frozen reference copy of selected modules
        self.ref_modules = nn.ModuleDict({
            "plan_anchor_encoder": copy.deepcopy(self.plan_anchor_encoder).to(device).eval(),
            "time_mlp": copy.deepcopy(self.time_mlp).to(device).eval(),
            "diff_decoder": copy.deepcopy(self.diff_decoder).to(device).eval(),
        })
        # freeze reference parameters so they're not included in training
        for p in self.ref_modules.parameters():
            p.requires_grad = False
        self.ref_modules.eval()
        print("Created reference modules for NFT diffusion model.")

    def update_reference_ema(self, global_step: int, decay: float = None, device: torch.device = None):
        """
        Update self.ref_modules using exponential moving average from the live modules.

        Args:
            decay: EMA decay factor in [0,1]. If None, uses self._config.ema_decay if present else 0.999.
            device: optional device to move reference modules before update (keeps dtype/device consistent).
        """
        if not hasattr(self, "ref_modules"):
            raise AttributeError("ref_modules not found. Call the reference-creation routine first.")

        if decay is None:
            decay = return_decay(global_step, self.decay_type) # min(0.001 * global_step, 0.5)

        with torch.no_grad():
            for name, ref in self.ref_modules.items():
                src = getattr(self, name)

                # # ensure reference on the desired device (optional)
                # if device is not None:
                #     ref.to(device)

                # update parameters with EMA: ref = decay * ref + (1 - decay) * src
                src_params = dict(src.named_parameters())
                ref_params = dict(ref.named_parameters())
                for param_name, src_p in src_params.items():
                    if param_name in ref_params:
                        ref_p = ref_params[param_name]
                        ref_p.data.mul_(decay).add_(src_p.data * (1.0 - decay))

                # copy non-parameter buffers (e.g., running_mean/running_var) directly
                src_bufs = dict(src.named_buffers())
                ref_bufs = dict(ref.named_buffers())
                for buf_name, src_b in src_bufs.items():
                    if buf_name in ref_bufs:
                        ref_bufs[buf_name].data.copy_(src_b.data)

    def _initialize_pdm_scorer(self):
        cfg = self._config
        assert (
            self.simulator.proposal_sampling == self.scorer.proposal_sampling
        ), "Simulator and scorer proposal sampling has to be identical"

        self.metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
        self.metric_cache_dict = {}
        self.scene_loader = SceneLoader(
            sensor_blobs_path=Path(cfg.sensor_blobs_path),
            data_path=Path(cfg.navsim_log_path),
            scene_filter=self.scene_filter,
            sensor_config=SensorConfig.build_no_sensors(),
        )

    def collect_data(self, 
                      ego_query,
                      agents_query,
                      bev_feature,
                      bev_spatial_shape,
                      status_encoding,
                      global_img,
                      num_samples: int = 1,
                      ) -> torch.Tensor:
        """Generate trajectory predictions using the diffusion model.

        Args:
            ego_query: Tensor of shape (batch_size, 1, d_model)
            agents_query: Tensor of shape (batch_size, num_agents, d_model)
            bev_feature: Tensor of shape (batch_size, d_model, H, W)
            bev_spatial_shape: Tuple (H, W)
            status_encoding: Tensor of shape (batch_size, status_dim)
            global_img: Tensor of shape (batch_size, C, H_img, W_img)
            num_samples: Number of trajectory samples to generate per instance.
        Returns:
            Tensor containing predicted trajectories
        """
        batch_size = ego_query.shape[0]
        device = ego_query.device
        shape = (batch_size, num_samples, self._num_poses, self._action_dim)
        
        self.diffusion_scheduler.set_timesteps(self.diffusion_inference_steps, device)
        noisy_traj_points = torch.randn(shape, device=device)  # Initial noise, (b, 1, trajectory_steps, action_dim)
        for i, t in enumerate(self.diffusion_scheduler.timesteps):
            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64) # (b, 1, trajectory_steps, 64)
            traj_pos_embed = traj_pos_embed.flatten(-2) # (b, 1, trajectory_steps*64)
            traj_feature = self.ref_modules['plan_anchor_encoder'](traj_pos_embed) # (b, 1, trajectory_steps, d_model=512)
            traj_feature = traj_feature.view(batch_size, num_samples, -1) # (b, 1, trajectory_steps*d_model)

            timesteps = t
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(device)
            
            # 3. embed the timesteps
            timesteps = timesteps.expand(batch_size)
            time_embed = self.ref_modules['time_mlp'](timesteps)
            time_embed = time_embed.view(batch_size, 1, -1)

            # 4. begin the stacked decoder
            poses_reg_list = self.ref_modules['diff_decoder'](
                traj_feature,
                noisy_traj_points,
                bev_feature,
                bev_spatial_shape,
                agents_query, 
                ego_query, 
                time_embed, 
                status_encoding,
                global_img,
                need_denormed=True,)
            poses_reg = poses_reg_list[-1]

            noisy_traj_points = self.diffusion_scheduler.step(
                model_output=poses_reg,
                timestep=t,
                sample=noisy_traj_points
            ).prev_sample
        
        pred_trajs = denorm_odo(noisy_traj_points)
        return pred_trajs
    
    def ref_forward_train(self, 
                      ego_query,
                      agents_query,
                      bev_feature,
                      bev_spatial_shape,
                      status_encoding, 
                      timesteps,
                      noisy_traj_points,
                      targets: Dict=None,
                      global_img=None,
                      
                      ) -> Dict[str, torch.Tensor]:
        """

        ego_query: (bs, 1, d_model)
        agents_query: (bs, num_agents, d_model)
        bev_feature: (bs, d_model, H, W) (64, 128)
        bev_spatial_shape: (H, W)
        status_encoding: (bs, status_dim)
        targets['trajectory']: (bs, 8, 3)
        """
        # device = ego_query.device
        batch_size = ego_query.shape[0]

        # 2. proj noisy_traj_points to the query
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64) # (b, 1, trajectory_steps, 64)
        traj_pos_embed = traj_pos_embed.flatten(-2) # (b, 1, trajectory_steps*64)
        traj_feature = self.ref_modules['plan_anchor_encoder'](traj_pos_embed)  # (b, 1, d_model=256)
        traj_feature = traj_feature.view(batch_size, -1, self._d_model) # (b, 1, d_model)

        # 3. embed the timesteps
        time_embed = self.ref_modules['time_mlp'](timesteps)  # (b, d_model)
        time_embed = time_embed.view(batch_size, 1, self._d_model) # (b, 1, d_model)

        # 4. begin the stacked decoder
        poses_reg_list = self.ref_modules['diff_decoder'](traj_feature, 
                                           noisy_traj_points,
                                           bev_feature,
                                           bev_spatial_shape,
                                           agents_query,
                                           ego_query,
                                           time_embed,
                                           status_encoding,
                                           global_img,
                                           need_denormed=True,
                                           )
        trajectory = poses_reg_list[-1]
        outputs = {
            "trajectory": trajectory,
            }

        return outputs

    def forward_train(self, 
                      ego_query, 
                      agents_query, 
                      bev_feature, 
                      bev_spatial_shape, 
                      status_encoding, 
                      targets = None, 
                      global_img=None,
                      tokens: List[str] = None,):
        if tokens is None:
            raise ValueError("Tokens must be provided for NFT training")
        # 1. data collection phrase 
        if not hasattr(self, "ref_modules"):
            self.create_reference_modules()
        batched_x_target = targets['trajectory'].unsqueeze(1).repeat(1, self._num_samples, 1, 1) # (bs, num_samples, T, 3)
        with torch.no_grad():
            rollout_trajs = self.collect_data(
                ego_query,
                agents_query,
                bev_feature,
                bev_spatial_shape,
                status_encoding,
                global_img,
                num_samples=self._num_samples) # (bs, num_samples, trajectory_steps, action_dim)
            rollout_trajs_error_dict = calculate_component_losses(
                batched_x_target, rollout_trajs, "rollout_"
            )

        # 2. compute PDM scores and optimality probabilities
        batch_size = ego_query.shape[0]
        rewards = np.zeros((batch_size, self._num_samples), dtype=np.float32)
        for i in range(batch_size):
            token = tokens[i]
            for j in range(self._num_samples):
                if token not in self.metric_cache_dict:
                    metric_cache = self.metric_cache_loader.get_from_token(token)
                    self.metric_cache_dict[token] = metric_cache
                metric_cache = self.metric_cache_dict[token]

                trajectory = Trajectory(rollout_trajs[i][j].cpu().numpy())
                pdm_result = pdm_score(
                    metric_cache=metric_cache,
                    model_trajectory=trajectory,
                    future_sampling=self.simulator.proposal_sampling,
                    simulator=self.simulator,
                    scorer=self.scorer,
                )
                rewards[i][j] = pdm_result.score
        # Normalize rewards to [0.0, 1.0]
        norm_rewards = rewards - np.mean(rewards, axis=1, keepdims=True)
        r_optimal_prob = np.clip(
            norm_rewards / self._Zc, -1, 1) * 0.5 + 0.5  # shape (bs, num_samples)
        r_optimal_prob = r_optimal_prob.reshape(-1) # (bs * num_samples, )
        r_optimal_prob = torch.tensor(r_optimal_prob, device=ego_query.device)
        
        bached_ego_query = ego_query.unsqueeze(1).repeat(1, self._num_samples, 1, 1) # (bs, num_samples, 1, C)
        bached_agents_query = agents_query.unsqueeze(1).repeat(1, self._num_samples, 1, 1) # (bs, num_samples, num_agents, C)
        batched_bev_feature = bev_feature.unsqueeze(1).repeat(1, self._num_samples, 1, 1, 1) # (bs, num_samples, C, H, W)

        reshaped_ego_query = bached_ego_query.view(
            batch_size * self._num_samples,
            bached_ego_query.shape[2], 
            bached_ego_query.shape[3]) # (bs * num_samples, 1, C)
        reshaped_agents_query = bached_agents_query.view(
            batch_size * self._num_samples, 
            bached_agents_query.shape[2], 
            bached_agents_query.shape[3]) # (bs * num_samples, num_agents, C)
        reshaped_x_target = batched_x_target.view(
            batch_size * self._num_samples, 
            batched_x_target.shape[2], 
            batched_x_target.shape[3]) # (bs * num_samples, T, 3)
        reshaped_bev_feature = batched_bev_feature.view(
            batch_size * self._num_samples, 
            batched_bev_feature.shape[2], 
            batched_bev_feature.shape[3], 
            batched_bev_feature.shape[4]) # (bs * num_samples, C, H, W)

        # 3. shuffle replay buffer
        len_replay_buffer = reshaped_ego_query.shape[0]
        indices = list(range(len_replay_buffer))
        random.shuffle(indices)

        # 4. gradient accumulation over mini-batches
        trajectory_loss = 0.
        nft_loss = 0.
        positive_loss = 0.
        negative_loss = 0.
        kl_div_loss = 0.
        trajectory_loss_dict = {}
        for i in range(self.num_layers):
            trajectory_loss_dict[f"trajectory_loss_{i}"] = 0.
        num_mini_batches = 0
        component_loss_dict = {}
        traj_stat_dict = {}
        for k in range(0, len_replay_buffer, self.mini_batch_size):
            mb_indices = indices[k : k + self.mini_batch_size]
            mb_ego_query = reshaped_ego_query[mb_indices]
            mb_agents_query = reshaped_agents_query[mb_indices]
            mb_x_target = reshaped_x_target[mb_indices]
            mb_bev_feature = reshaped_bev_feature[mb_indices]
            mb_r_optimal_prob = r_optimal_prob[mb_indices] # (mb, )

            mb_output_dict = super().forward_train(
                mb_ego_query,
                mb_agents_query,
                mb_bev_feature,
                bev_spatial_shape,
                status_encoding=None,
                targets={"trajectory": mb_x_target},
                global_img=None,
            )
            v_pred = mb_output_dict['trajectory'][:, 0]  # (mb, trajectory_steps, action_dim)
            timesteps = mb_output_dict['timesteps']
            noisy_traj_points = mb_output_dict['noisy_traj_points']
            curr_sta_dict = mb_output_dict['sta_dict']
            for key, val in curr_sta_dict.items():
                if key not in component_loss_dict:
                    component_loss_dict[f"curr_{key}"] = 0
                component_loss_dict[f"curr_{key}"] += val
            curr_traj_stat_dict = mb_output_dict['traj_dict']
            for key, val in curr_traj_stat_dict.items():
                if key not in traj_stat_dict:
                    traj_stat_dict[f"curr_{key}"] = 0
                traj_stat_dict[f"curr_{key}"] += val 

            trajectory_loss += mb_output_dict['trajectory_loss']
            for i in range(self.num_layers):
                trajectory_loss_dict[f"trajectory_loss_{i}"] += mb_output_dict['trajectory_loss_dict'][f"trajectory_loss_{i}"]

            ref_output_dict = self.ref_forward_train(
                mb_ego_query,
                mb_agents_query,
                mb_bev_feature,
                bev_spatial_shape,
                status_encoding=None,
                timesteps=timesteps,
                noisy_traj_points=noisy_traj_points,
            )
            ref_v_pred = ref_output_dict['trajectory'][:, 0]  # (mb, trajectory_steps, action_dim)

            xt = noisy_traj_points[:, 0]  # (mb, trajectory_steps, action_dim)
            t_expanded = timesteps.view(-1, *([1] * (len(xt.shape) - 1)))  # (mb, 1, 1)

            positive_prediction = (1 - self.beta) * ref_v_pred.detach() + self.beta * v_pred  # (mb, trajectory_steps, action_dim)
            negative_prediction = (1 + self.beta) * ref_v_pred.detach() - self.beta * v_pred  # (mb, trajectory_steps, action_dim)
            
            if self.target_type == "x0":
                normed_x0 = norm_odo(mb_x_target)
                x0_prediction = xt - t_expanded * positive_prediction  # (mb, trajectory_steps, action_dim)
                negative_x0_prediction = xt - t_expanded * negative_prediction # (mb, trajectory_steps, action_dim)
            elif self.target_type == "velocity":
                normed_x0 = mb_output_dict['targets'][:, 0]
                x0_prediction = positive_prediction
                negative_x0_prediction = negative_prediction
            else:
                raise NotImplementedError
            ref_v_pred_stats = calculate_statistics(
                noisy_traj_points=noisy_traj_points,
                ref_v_pred=ref_v_pred,
                x0_prediction=x0_prediction,
                negative_x0_prediction=negative_x0_prediction,
                )
            for key, val in ref_v_pred_stats.items():
                if key not in traj_stat_dict:
                    traj_stat_dict[f"ref_{key}"] = 0
                traj_stat_dict[f"ref_{key}"] += val
            
            # adaptive weighting
            if self.use_adaptive_weighted_policy_losses:
                with torch.no_grad():
                    weight_factor = (
                        torch.abs(x0_prediction.double() - normed_x0.double())
                        .mean(dim=tuple(range(1, normed_x0.ndim)), keepdim=True)
                        .clip(min=0.00001)
                    )  # (mb, 1, 1)
                p_loss = ((x0_prediction - normed_x0) ** 2 / weight_factor).mean(
                    dim=tuple(range(1, normed_x0.ndim))) # (mb, )

                with torch.no_grad():
                    negative_weight_factor = (
                        torch.abs(negative_x0_prediction.double() - normed_x0.double())
                        .mean(dim=tuple(range(1, normed_x0.ndim)), keepdim=True)
                        .clip(min=0.00001)
                    )  # (mb, 1, 1)
                n_loss = ((negative_x0_prediction - normed_x0) ** 2 / negative_weight_factor).mean(
                    dim=tuple(range(1, mb_x_target.ndim))) # (mb, )
            else:
                p_loss = ((x0_prediction - normed_x0) ** 2).mean(
                    dim=tuple(range(1, normed_x0.ndim))) # (mb, )
                n_loss = ((negative_x0_prediction - normed_x0) ** 2).mean(
                    dim=tuple(range(1, normed_x0.ndim))) # (mb, )
            
            positive_loss += p_loss.mean().item()
            negative_loss += n_loss.mean().item()
            mb_nft_loss = mb_r_optimal_prob * p_loss / self.beta + (1 - mb_r_optimal_prob) * n_loss / self.beta
            nft_loss += mb_nft_loss.mean()

            if self.use_kl_div_loss:
                mb_kl_div_loss = ((v_pred - ref_v_pred) ** 2).mean(
                    dim=tuple(range(1, normed_x0.ndim)))
                kl_div_loss += mb_kl_div_loss.mean()
            num_mini_batches += 1

            component_losses = calculate_component_losses(normed_x0, x0_prediction, prefix="ref_")
            for key, val in component_losses.items():
                if key not in component_loss_dict:
                    component_loss_dict[key] = 0
                component_loss_dict[key] += val 
        for key, val in component_loss_dict.items():
            component_loss_dict[key] = val / num_mini_batches
        for key, val in traj_stat_dict.items():
            traj_stat_dict[key] = val / num_mini_batches
        outputs = {
            "trajectory_loss": trajectory_loss / num_mini_batches,
            "nft_loss": nft_loss / num_mini_batches,
            "kl_div_loss": kl_div_loss / num_mini_batches,
            "sta_dict":{
                "r_optimal_prob/mean": r_optimal_prob.mean(),
                "r_optimal_prob/std": r_optimal_prob.std(),
                "rewards/mean": rewards.mean(),
                "rewards/std": rewards.std(),
                "rewards/min": rewards.min(),
                "rewards/max": rewards.max(),
                "positive_loss": positive_loss / num_mini_batches,
                "negative_loss": negative_loss / num_mini_batches,
                **component_loss_dict,
                **rollout_trajs_error_dict,
                **traj_stat_dict,
            },
            "trajectory_loss_dict": {k: v / num_mini_batches for k, v in trajectory_loss_dict.items()},
        }

        return outputs
