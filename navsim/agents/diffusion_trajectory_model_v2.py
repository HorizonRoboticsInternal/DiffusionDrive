import logging
from typing import List, Optional, Dict, Tuple, Type, Any, Union
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import timm
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler, DDIMScheduler

from navsim.agents.fiery.fiery_model import (
    _get_clones, ModulationLayer, StateSE2Index
)
from navsim.agents.diffusiondrive.modules.blocks import linear_relu_ln,bias_init_with_prob, gen_sineembed_for_position, GridSampleCrossBEVAttention
from navsim.agents.diffusiondrive.modules.conditional_unet1d import SinusoidalPosEmb
from navsim.agents.fiery.fiery_config import FieryConfig
from navsim.agents.diffusion_trajectory_model import (
    CustomTransformerDecoder,
    DiffMotionPlanningRefinementModule, 
    norm_odo, denorm_odo,
    calculate_component_losses,
    calculate_statistics
)


class AdaLnBlock(nn.Module):
    """Residual block with adaptive layer normalization conditioning."""

    def __init__(self, in_dim, out_dim, hidden_dim, cond_dim):
        """Configure the adaptive layer normalization block.

        Args:
            in_dim: Size of the input feature dimension.
            out_dim: Size of the output feature dimension.
            hidden_dim: Hidden dimension for the internal MLP.
            cond_dim: Dimensionality of the conditioning vector applied to AdaLN.
        """
        super().__init__()
        self._norm = nn.LayerNorm(in_dim, elementwise_affine=False)
        # self._fc1 = alf.layers.FC(in_dim,
        #                           hidden_dim,
        #                           activation=torch.nn.functional.silu)
        # self._fc2 = alf.layers.FC(hidden_dim, out_dim)
        self._fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self._ada = torch.nn.Sequential(
            # torch.nn.SiLU(),
            # alf.layers.FC(cond_dim, 3 * in_dim, use_bias=True),
            nn.Linear(cond_dim, 3 * in_dim),
            )

    def forward(self, x, cond):
        h = self._norm(x)
        scale, shift, gate = self._ada(cond).chunk(3, dim=-1)
        h = h * (1 + scale) + shift
        h = self._fc(h)
        return x + h * gate


class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, 
                 num_poses,
                 d_model,
                 d_ffn,
                 config,
                 ego_fut_mode,
                 enable_pooling: bool = False,
                 ):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.cross_agent_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.cross_ego_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )
        self.norm1 = nn.LayerNorm(config.tf_d_model)
        self.norm2 = nn.LayerNorm(config.tf_d_model)
        self.norm3 = nn.LayerNorm(config.tf_d_model)
        self.time_modulation = ModulationLayer(config.tf_d_model, 256)
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=config.tf_d_model,
            ego_fut_ts=num_poses,
        )
        self.enable_pooling = enable_pooling
        if enable_pooling:
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.mlp_bev = AdaLnBlock(
                in_dim=config.tf_d_model,
                out_dim=config.tf_d_model,
                hidden_dim=config.tf_d_ffn,
                cond_dim=config.tf_d_model,
            )
        else:
            self.cross_bev_attention = nn.MultiheadAttention(
                config.tf_d_model,
                config.tf_num_head,
                dropout=config.tf_dropout,
                batch_first=True,
            )

    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                time_embed, 
                status_encoding,
                global_img=None,
                need_denormed=False,
                ):
        """
        traj_feature: (bs, 1, trajectory_steps*d_model)
        noisy_traj_points: (bs, 1, trajectory_steps, 3)
        bev_feature: (bs, d_model, H, W) (64, 128)
        bev_spatial_shape: (H, W)
        agents_query: (bs, num_agents, d_model)
        ego_query: (bs, 1, d_model)
        time_embed: (bs, 1, d_model)

        """
        if len(bev_feature.shape) == 4:
            if self.enable_pooling:
                bev_feature = self.global_pool(bev_feature).squeeze(-1).squeeze(-1) # (bs, 1, d_model)
                num_samples = traj_feature.shape[1]
                bev_feature = bev_feature.unsqueeze(1).expand(-1, num_samples, -1) # (bs, num_samples, d_model)
            else:
                bev_feature = bev_feature.flatten(-2, -1).permute(0, 2, 1) # (bs, H*W, d_model)
        # 4.4 cross attention with bev feature
        if self.enable_pooling:
            traj_feature = self.mlp_bev(traj_feature, bev_feature)
        else:
            traj_feature = self.cross_bev_attention(
                traj_feature,
                bev_feature,
                bev_feature)[0]  # (bs, 1, d_model)
        traj_feature = traj_feature + self.dropout(
            self.cross_agent_attention(
                traj_feature, 
                agents_query,
                agents_query)[0])  # (bs, 1, d_model)
        traj_feature = self.norm1(traj_feature)
        
        # traj_feature = traj_feature + self.dropout(self.self_attn(traj_feature, traj_feature, traj_feature)[0])

        # 4.5 cross attention with  ego query
        traj_feature = traj_feature + self.dropout1(
            self.cross_ego_attention(
                traj_feature, 
                ego_query,
                ego_query)[0])
        traj_feature = self.norm2(traj_feature)  # (bs, 1, d_model)
        
        # 4.6 feedforward network
        traj_feature = self.norm3(self.ffn(traj_feature))
        # 4.8 modulate with time steps
        traj_feature = self.time_modulation(traj_feature, time_embed,global_cond=None,global_img=global_img)
        
        # 4.9 predict the offset & heading
        poses_reg = self.task_decoder(traj_feature) # (bs, 1, 8, 3)
        poses_reg = poses_reg + noisy_traj_points
        poses_reg[..., StateSE2Index.HEADING] = poses_reg[..., StateSE2Index.HEADING].tanh() * np.pi

        return poses_reg


class DiffusionTrajectoryHeadv2(nn.Module):
    """FlowMatching model for trajectory prediction
    """

    def __init__(
            self,
            num_poses: int,
            d_ffn: int,
            d_model: int,
            config: FieryConfig = FieryConfig,
        ):
        """Initialize the DiffusionTrajectoryHead."""
        super().__init__()
        
        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]

        self._config = config
        self._num_poses = num_poses
        self._action_dim = 3  # x, y, heading
        self._d_model = d_model
        self._d_ffn = d_ffn
        self.diff_loss_weight = 2.0
        self.training_time_type = config.training_time_type
        self.scheduler_type = config.diffusion_scheduler_type
        self.diffusion_train_steps = config.diffusion_train_steps
        self.diffusion_inference_steps = config.diffusion_inference_steps
        self.ego_fut_mode = config.ego_fut_mode  # currently only support 1 mode

        # Initialize diffusion scheduler
        if self.scheduler_type == 'ddim':
            self.diffusion_scheduler = DDIMScheduler(
                num_train_timesteps=config.diffusion_train_steps,
                beta_schedule="scaled_linear",
                prediction_type="sample",
            )
        elif self.scheduler_type == 'flow_matching_euler':
            self.diffusion_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=config.diffusion_train_steps)
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1,512),
            nn.Linear(d_model, d_model),
        )
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )
        diff_decoder_layer = CustomTransformerDecoderLayer(
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            config=config,
            ego_fut_mode=config.ego_fut_mode,
            enable_pooling=config.enable_pooling,
        )
        self.num_layers = 2
        self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, self.num_layers)

    def get_sigmas(self,
                   timesteps: torch.Tensor,
                   device: torch.device,
                   n_dim: int = 4,
                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Get noise schedule sigmas for given timesteps.

        Args:
            timesteps: Tensor of diffusion timesteps
            device: Target device for computation
            n_dim: Number of dimensions to expand sigma to
            dtype: Data type for sigma values

        Returns:
            torch.Tensor: Noise schedule sigmas expanded to match input dimensions
        """
        # Move noise schedule to specified device and dtype
        sigmas = self.diffusion_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.diffusion_scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)

        # Find indices in schedule corresponding to requested timesteps
        step_indices = [
            (schedule_timesteps.round() == t.round()).nonzero().item()
            for t in timesteps
        ]

        # Get sigma values and flatten to 1D
        sigma = sigmas[step_indices].flatten()

        # Expand dimensions to match required shape
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)

        return sigma

    def _prepare_model_input(self, x_target: torch.Tensor, noise: torch.Tensor,
                             timesteps: torch.Tensor, pow=0.5):
        # Add noise to the model input according to the noise magnitude at each timestep
        if "flow_matching" in self.scheduler_type:
            if self.training_time_type == "discrete":
                sigma = self.get_sigmas(timesteps, x_target.device, n_dim=4)
            elif self.training_time_type == "continuous":
                n_dim = len(x_target.shape)
                sigma = torch.rand(x_target.shape[0], device=x_target.device)
                sigma = sigma ** pow
                sigma = torch.clamp_max(sigma, 1.0)
                while len(sigma.shape) < n_dim:
                    sigma = sigma.unsqueeze(-1)
            else:
                raise NotImplementedError
            noisy_model_input = sigma * noise + (1.0 - sigma) * x_target
        else:
            noisy_model_input = self.diffusion_scheduler.add_noise(
                x_target, noise, timesteps)
        
        return noisy_model_input

    def forward_train(self, 
                      ego_query,
                      agents_query,
                      bev_feature,
                      bev_spatial_shape,
                      status_encoding, 
                      targets: Dict=None,
                      global_img=None,
                      tokens=None,
                      ) -> Dict[str, torch.Tensor]:
        """
        
        targets['trajectory']: (bs, 8, 3)
        """
        device = ego_query.device

        x_target = targets["trajectory"]  # (b, trajectory_steps, action_dim)
        x_target = x_target.to(torch.float32)
        if len(x_target.shape) == 3:
            x_target = x_target.unsqueeze(1) # add extra dim for num_modes=1
        assert len(x_target.shape) == 4, f"Expected trajectory target shape to be (b, num_modes, trajectory_steps, action_dim), but got {x_target.shape}"
        normed_x_target = norm_odo(x_target)
        batch_size = x_target.shape[0]

        noise = torch.randn_like(normed_x_target)  # eps ~ N(0, 1)
        timesteps = torch.randint(1,
                                  self.diffusion_train_steps, (batch_size, ),
                                  device=device).long()  # (b, )
        noisy_traj_points = self._prepare_model_input(normed_x_target, noise, timesteps) # (b, 1, trajectory_steps, action_dim)

        # 2. proj noisy_traj_points to the query
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64) # (b, 1, trajectory_steps, 64)
        traj_pos_embed = traj_pos_embed.flatten(-2) # (b, 1, trajectory_steps*64)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)  # (b, 1, d_model=256)
        traj_feature = traj_feature.view(batch_size, -1, self._d_model) # (b, 1, d_model)

        # 3. embed the timesteps
        time_embed = self.time_mlp(timesteps)  # (b, d_model)
        time_embed = time_embed.view(batch_size, 1, -1) # (b, 1, d_model)

        # 4. begin the stacked decoder
        poses_reg_list = self.diff_decoder(traj_feature, 
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
        trajectory_loss_dict = {}
        ret_traj_loss = 0
        for idx, poses_reg in enumerate(poses_reg_list):
            if self.scheduler_type == 'ddim':
                trajectory_loss = F.l1_loss(poses_reg, normed_x_target)
            elif "flow_matching" in self.scheduler_type:
                v_target = noise - normed_x_target
                trajectory_loss = F.mse_loss(poses_reg, v_target)
            trajectory_loss_dict[f"trajectory_loss_{idx}"] = trajectory_loss
            ret_traj_loss += trajectory_loss

        v_prediction = poses_reg_list[-1]
        x0_prediction = noisy_traj_points - timesteps[:,None,None,None] * v_prediction 
        component_losses = calculate_component_losses(normed_x_target, x0_prediction)
        trajectory_stats = calculate_statistics(
            noisy_traj_points=noisy_traj_points,
            normed_x_target=normed_x_target,
            x_target=x_target,
            model_prediction=v_prediction,
            model_x0_prediction=x0_prediction
        )
        outputs = {
            "trajectory": v_prediction,
            "trajectory_loss": ret_traj_loss,
            "trajectory_loss_dict": trajectory_loss_dict,
            "timesteps": timesteps,
            "noisy_traj_points": noisy_traj_points,
            **component_losses,  # Unpack component losses
            "traj_dict": trajectory_stats,
        }

        return outputs

    def p_sample_loop(self, 
                      ego_query,
                      agents_query,
                      bev_feature,
                      bev_spatial_shape,
                      status_encoding,
                      global_img,
                      num_samples: int = 1,
                      ) -> torch.Tensor:
        """Generate trajectory predictions using the diffusion model.
        Returns:
            Tensor containing predicted trajectories
        """
        batch_size = ego_query.shape[0]
        device = ego_query.device
        shape = (batch_size, num_samples, self._num_poses, self._action_dim)
        
        self.diffusion_scheduler.set_timesteps(self.diffusion_inference_steps, device)
        noisy_traj_points = torch.randn(shape, device=device)  # Initial noise, (b, 1, trajectory_steps, action_dim)
        for i, t in enumerate(self.diffusion_scheduler.timesteps):
            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64) # (b, 1, trajectory_steps, action_dim, 64)
            traj_pos_embed = traj_pos_embed.flatten(-2) # (b, 1, trajectory_steps, action_dim*64)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed) # (b, 1, trajectory_steps, d_model=512)
            traj_feature = traj_feature.view(batch_size, num_samples, -1) # (b, 1, trajectory_steps*d_model)

            timesteps = t
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(device)
            
            # 3. embed the timesteps
            timesteps = timesteps.expand(batch_size)
            time_embed = self.time_mlp(timesteps)
            time_embed = time_embed.view(batch_size, 1, -1)

            # 4. begin the stacked decoder
            poses_reg_list = self.diff_decoder(
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
        
        return noisy_traj_points

    @torch.no_grad()
    def forward_test(self, 
                     ego_query,
                     agents_query,
                     bev_feature,
                     bev_spatial_shape,
                     status_encoding,
                     global_img,
                     tokens=None,
                     ) -> Dict[str, torch.Tensor]:
        """Generate trajectory predictions using the diffusion model.

        Returns:
            Dictionary containing predicted trajectories and model outputs
        """
        pred_trajs = self.p_sample_loop(
            ego_query, agents_query, bev_feature, bev_spatial_shape, 
            status_encoding, global_img, num_samples=self.ego_fut_mode)
        trajectory = pred_trajs[:, 0] # (b, trajectory_steps, action_dim)
        trajectory = denorm_odo(trajectory)

        return  {"trajectory": trajectory}

    def forward(self, ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding, targets=None,global_img=None, tokens=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        if self.training:
            return self.forward_train(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,targets,global_img, tokens=tokens)
        else:
            x_target = targets["trajectory"]
            out_dict = self.forward_test(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,global_img)
            pred_traj = out_dict['trajectory']
            component_losses = calculate_component_losses(x_target, pred_traj)
            trajectory_stats = calculate_statistics(
                x_target=x_target,
                pred_traj=pred_traj,
            )
            out_dict.update(**component_losses)
            out_dict['traj_dict'] = trajectory_stats
            return out_dict