
from hydra.utils import instantiate
from typing import List, Optional, Dict, Tuple, Any
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
    """Diffusion-based trajectory model for NFT (Neural Flow Trajectory) agent.
    
    Extends `DiffusionTrajectoryHeadv2` to support NFT-style policy learning
    with reference model anchoring, PDM scoring, and adaptive trajectory optimization.
    """

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
    ):
        """
        Args:
            config: Global Fiery configuration.
            num_samples: Number of diffusion samples per data point.
            mini_batch_size: Size for mini-batch gradient updates.
            beta: Interpolation scaling factor for positive/negative predictions.
            Zc: Reward normalization constant.
            use_adaptive_weighted_policy_losses: Enable adaptive weighting of policy loss.
            use_kl_div_loss: Enable KL divergence consistency regularization.
            simulator: PDM simulator instance.
            scorer: PDM scorer instance.
            scene_filter: Scene filtering rules.
        """
        super().__init__(
            config=config,
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )
        assert self.scheduler_type == "flow_matching_euler", \
            "DiffusionNFTTrajectoryModel only supports flow_matching_euler scheduler."

        # --- Core configuration ---
        self._Zc = Zc
        self._num_samples = num_samples
        self.mini_batch_size = mini_batch_size
        self.beta = beta
        self.use_adaptive_weighted_policy_losses = use_adaptive_weighted_policy_losses
        self.use_kl_div_loss = use_kl_div_loss
        self.decay_type = config.decay_type
        self.target_type = config.target_type

        # --- External modules ---
        self.simulator = simulator
        self.scorer = scorer
        self.scene_filter = scene_filter

    # -------------------------------------------------------------------------
    # Reference Module Management
    # -------------------------------------------------------------------------
    def set_simulator(self, simulator, scorer, scene_filter):
        """Bind simulator, scorer, and filter, and initialize PDM scorer."""
        self.simulator = simulator
        self.scorer = scorer
        self.scene_filter = scene_filter
        self._initialize_pdm_scorer()

    def create_reference_modules(self):
        """Create frozen reference copies of the main encoder/decoder for EMA updates."""
        device = next(self.parameters()).device
        self.ref_modules = nn.ModuleDict({
            "plan_anchor_encoder": copy.deepcopy(self.plan_anchor_encoder).to(device).eval(),
            "time_mlp": copy.deepcopy(self.time_mlp).to(device).eval(),
            "diff_decoder": copy.deepcopy(self.diff_decoder).to(device).eval(),
        })
        for p in self.ref_modules.parameters():
            p.requires_grad = False
        print("✅ Created reference modules for NFT diffusion model.")

    def update_reference_ema(self, global_step: int, decay: float = None):
        """Exponential Moving Average (EMA) update of reference modules.
        
        EMA ensures stable pseudo-targets for imitation and reward shaping.
        """
        if not hasattr(self, "ref_modules"):
            raise AttributeError("ref_modules not found. Call create_reference_modules() first.")
        if decay is None:
            decay = return_decay(global_step, self.decay_type)

        with torch.no_grad():
            for name, ref in self.ref_modules.items():
                src = getattr(self, name)
                for (k, src_p), ref_p in zip(src.named_parameters(), ref.parameters()):
                    ref_p.data.mul_(decay).add_(src_p.data * (1.0 - decay))
                # Sync running stats (buffers)
                for (bn, src_buf), ref_buf in zip(src.named_buffers(), ref.buffers()):
                    ref_buf.data.copy_(src_buf.data)

    # -------------------------------------------------------------------------
    # PDM Scoring and Scene Management
    # -------------------------------------------------------------------------
    def _initialize_pdm_scorer(self):
        """Initialize PDM scorer, simulator, and metric caches."""
        cfg = self._config
        assert self.simulator.proposal_sampling == self.scorer.proposal_sampling, \
            "Simulator and scorer proposal sampling must match."

        self.metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
        self.metric_cache_dict = {}
        self.scene_loader = SceneLoader(
            sensor_blobs_path=Path(cfg.sensor_blobs_path),
            data_path=Path(cfg.navsim_log_path),
            scene_filter=self.scene_filter,
            sensor_config=SensorConfig.build_no_sensors(),
        )

    # -------------------------------------------------------------------------
    # Reference Forward Pass (frozen)
    # -------------------------------------------------------------------------
    def ref_forward_train(self, ego_query, agents_query, bev_feature, bev_spatial_shape,
                          status_encoding, timesteps, noisy_traj_points, targets=None, global_img=None):
        """Forward pass using the frozen reference modules (no gradient)."""
        bs = ego_query.shape[0]
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64).flatten(-2)
        traj_feature = self.ref_modules['plan_anchor_encoder'](traj_pos_embed).view(bs, -1, self._d_model)
        time_embed = self.ref_modules['time_mlp'](timesteps).view(bs, 1, self._d_model)
        poses_reg_list = self.ref_modules['diff_decoder'](
            traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
            agents_query, ego_query, time_embed, status_encoding, global_img, need_denormed=True)
        return {"trajectory": poses_reg_list[-1]}

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

    # -------------------------------------------------------------------------
    # Main Training Routine
    # -------------------------------------------------------------------------
    def forward_train(
        self,
        ego_query: torch.Tensor,
        agents_query: torch.Tensor,
        bev_feature: torch.Tensor,
        bev_spatial_shape: Tuple[int, int],
        status_encoding: Optional[torch.Tensor],
        targets: Dict[str, torch.Tensor],
        global_img: Optional[torch.Tensor] = None,
        tokens: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Full forward training pass for NFT-style diffusion trajectory learning.

        Pipeline:
        1. Reference rollout (frozen modules)
        2. PDM reward scoring → r_optimal_prob
        3. Replay buffer construction
        4. Mini-batch policy updates (positive/negative interpolation)
        5. Metrics & loss aggregation

        Returns:
            Dictionary with averaged losses and diagnostic statistics.
        """
        assert tokens is not None, "Tokens must be provided for NFT training."
        device = ego_query.device

        # ------------------------------------------------------------
        # 1️⃣ Create frozen reference modules if not yet built
        # ------------------------------------------------------------
        if not hasattr(self, "ref_modules"):
            self.create_reference_modules()

        batch_size = ego_query.shape[0]
        traj_target = targets["trajectory"]
        batched_target = traj_target.unsqueeze(1).repeat(1, self._num_samples, 1, 1)

        with torch.no_grad():
            rollout_trajs = self.collect_data(
                ego_query, agents_query, bev_feature, bev_spatial_shape,
                status_encoding, global_img, num_samples=self._num_samples
            )
            rollout_trajs_error_dict = calculate_component_losses(
                batched_target, rollout_trajs, prefix="rollout_"
            )

        # ------------------------------------------------------------
        # 2️⃣ Compute normalized rewards and r_optimal_prob
        # ------------------------------------------------------------
        rewards = np.zeros((batch_size, self._num_samples), dtype=np.float32)
        for i, token in enumerate(tokens):
            if token not in self.metric_cache_dict:
                self.metric_cache_dict[token] = self.metric_cache_loader.get_from_token(token)
            metric_cache = self.metric_cache_dict[token]

            for j in range(self._num_samples):
                traj = Trajectory(rollout_trajs[i][j].cpu().numpy())
                pdm_result = pdm_score(
                    metric_cache, traj,
                    self.simulator.proposal_sampling, self.simulator, self.scorer
                )
                rewards[i, j] = pdm_result.score

        norm_rewards = rewards - rewards.mean(axis=1, keepdims=True)
        r_optimal_prob = np.clip(norm_rewards / self._Zc, -1, 1) * 0.5 + 0.5
        r_optimal_prob = torch.tensor(r_optimal_prob.reshape(-1), device=device)

        # ------------------------------------------------------------
        # 3️⃣ Construct replay buffer (flattened)
        # ------------------------------------------------------------
        def repeat_flatten(x):
            return x.unsqueeze(1).repeat(1, self._num_samples, *([1] * (x.ndim - 1))).view(-1, *x.shape[1:])

        ego_q = repeat_flatten(ego_query)
        agents_q = repeat_flatten(agents_query)
        bev_f = repeat_flatten(bev_feature)
        x_target = repeat_flatten(traj_target)

        indices = torch.randperm(ego_q.shape[0])
        num_batches = 0

        # Accumulators
        trajectory_loss = 0.0
        nft_loss = 0.0
        kl_div_loss = 0.0
        positive_loss = 0.0
        negative_loss = 0.0
        trajectory_loss_dict = {f"trajectory_loss_{i}": 0.0 for i in range(self.num_layers)}
        component_loss_dict, traj_stat_dict = {}, {}

        # ------------------------------------------------------------
        # 4️⃣ Mini-batch training over replay buffer
        # ------------------------------------------------------------
        for start in range(0, len(indices), self.mini_batch_size):
            end = start + self.mini_batch_size
            mb_idx = indices[start:end]
            mb_ego, mb_agents, mb_bev, mb_x, mb_prob = (
                ego_q[mb_idx], agents_q[mb_idx], bev_f[mb_idx],
                x_target[mb_idx], r_optimal_prob[mb_idx]
            )

            # Forward live model
            mb_out = super().forward_train(
                mb_ego, mb_agents, mb_bev, bev_spatial_shape,
                status_encoding=None, targets={"trajectory": mb_x}
            )
            v_pred = mb_out["trajectory"][:, 0]
            timesteps = mb_out["timesteps"]
            noisy_traj_points = mb_out["noisy_traj_points"]

            # Aggregate per-layer trajectory losses
            trajectory_loss += mb_out["trajectory_loss"]
            for k in trajectory_loss_dict:
                trajectory_loss_dict[k] += mb_out["trajectory_loss_dict"][k]

            # Collect stats from live forward
            if "sta_dict" in mb_out:
                for k, v in mb_out["sta_dict"].items():
                    component_loss_dict[f"curr_{k}"] = component_loss_dict.get(f"curr_{k}", 0) + v
            if "traj_dict" in mb_out:
                for k, v in mb_out["traj_dict"].items():
                    traj_stat_dict[f"curr_{k}"] = traj_stat_dict.get(f"curr_{k}", 0) + v

            # Reference forward (frozen)
            ref_out = self.ref_forward_train(
                mb_ego, mb_agents, mb_bev, bev_spatial_shape,
                status_encoding=None, timesteps=timesteps,
                noisy_traj_points=noisy_traj_points
            )
            ref_v_pred = ref_out["trajectory"][:, 0]

            # --------------------------------------------------------
            # Compute NFT losses (positive/negative interpolation)
            # --------------------------------------------------------
            xt = noisy_traj_points[:, 0]
            t = timesteps.view(-1, *([1] * (len(xt.shape) - 1)))

            pos_pred = (1 - self.beta) * ref_v_pred.detach() + self.beta * v_pred
            neg_pred = (1 + self.beta) * ref_v_pred.detach() - self.beta * v_pred

            if self.target_type == "x0":
                norm_x0 = norm_odo(mb_x)
                x0_pos, x0_neg = xt - t * pos_pred, xt - t * neg_pred
            elif self.target_type == "velocity":
                norm_x0 = mb_out["targets"][:, 0]
                x0_pos, x0_neg = pos_pred, neg_pred
            else:
                raise NotImplementedError

            # Diagnostic statistics from reference outputs
            ref_stats = calculate_statistics(
                noisy_traj_points=noisy_traj_points,
                ref_v_pred=ref_v_pred,
                x0_prediction=x0_pos,
                negative_x0_prediction=x0_neg
            )
            for k, v in ref_stats.items():
                traj_stat_dict[f"ref_{k}"] = traj_stat_dict.get(f"ref_{k}", 0) + v

            # Adaptive weighted loss computation
            def compute_weighted_loss(pred, target):
                if self.use_adaptive_weighted_policy_losses:
                    weight = (
                        torch.abs(pred.double() - target.double())
                        .mean(dim=(1, 2), keepdim=True)
                        .clamp(min=1e-5)
                    )
                    return ((pred - target) ** 2 / weight).mean(dim=(1, 2))
                else:
                    return ((pred - target) ** 2).mean(dim=(1, 2))

            p_loss = compute_weighted_loss(x0_pos, norm_x0)
            n_loss = compute_weighted_loss(x0_neg, norm_x0)

            positive_loss += p_loss.mean().item()
            negative_loss += n_loss.mean().item()
            nft_loss += (mb_prob * p_loss / self.beta + (1 - mb_prob) * n_loss / self.beta).mean()

            if self.use_kl_div_loss:
                kl_div_loss += ((v_pred - ref_v_pred) ** 2).mean()

            # Component-wise error metrics
            comp_losses = calculate_component_losses(norm_x0, x0_pos, prefix="ref_")
            for k, v in comp_losses.items():
                component_loss_dict[k] = component_loss_dict.get(k, 0) + v

            num_batches += 1

        # ------------------------------------------------------------
        # 5️⃣ Aggregate all losses & statistics
        # ------------------------------------------------------------
        def avg_dict(d):
            return {k: v / max(1, num_batches) for k, v in d.items()}

        trajectory_loss_dict = avg_dict(trajectory_loss_dict)
        component_loss_dict = avg_dict(component_loss_dict)
        traj_stat_dict = avg_dict(traj_stat_dict)

        sta_dict = {
            "r_optimal_prob/mean": r_optimal_prob.mean(),
            "r_optimal_prob/std": r_optimal_prob.std(),
            "rewards/mean": rewards.mean(),
            "rewards/std": rewards.std(),
            "rewards/min": rewards.min(),
            "rewards/max": rewards.max(),
            "positive_loss": positive_loss / num_batches,
            "negative_loss": negative_loss / num_batches,
            **component_loss_dict,
            **rollout_trajs_error_dict,
            **traj_stat_dict,
        }

        return {
            "trajectory_loss": trajectory_loss / num_batches,
            "nft_loss": nft_loss / num_batches,
            "kl_div_loss": kl_div_loss / num_batches,
            "sta_dict": sta_dict,
            "trajectory_loss_dict": trajectory_loss_dict,
        }
