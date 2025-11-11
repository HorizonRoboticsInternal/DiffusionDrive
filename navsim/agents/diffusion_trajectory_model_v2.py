"""
Diffusion-based Trajectory Prediction Module

This module implements a diffusion model for autonomous vehicle trajectory prediction.
It uses a transformer-based architecture with flow matching for generating smooth,
realistic trajectories conditioned on BEV features, agent context, and ego state.
"""

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
from navsim.agents.diffusiondrive.modules.blocks import (
    linear_relu_ln, bias_init_with_prob, gen_sineembed_for_position,
    GridSampleCrossBEVAttention
)
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
    """
    Adaptive Layer Normalization Block with residual connection.
    
    This block applies conditional normalization based on external conditioning signals,
    allowing the model to modulate features based on context (e.g., timestep information).
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, cond_dim: int):
        """
        Initialize the AdaLN block.

        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension
            hidden_dim: Hidden layer dimension for the MLP
            cond_dim: Conditioning vector dimension
        """
        super().__init__()
        
        # Layer normalization without learnable affine parameters
        # (scale and shift will come from conditioning)
        self._norm = nn.LayerNorm(in_dim, elementwise_affine=False)
        
        # Main feature transformation pathway
        self._fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),  # Smooth activation function
            nn.Linear(hidden_dim, out_dim),
        )
        
        # Adaptive conditioning pathway
        # Outputs scale, shift, and gate parameters for modulation
        self._ada = nn.Sequential(
            nn.Linear(cond_dim, 3 * in_dim),  # 3x for scale, shift, gate
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with adaptive normalization.
        
        Args:
            x: Input features [batch, seq_len, in_dim]
            cond: Conditioning vector [batch, seq_len, cond_dim]
            
        Returns:
            Modulated features with residual connection [batch, seq_len, out_dim]
        """
        # Normalize input
        h = self._norm(x)
        
        # Generate adaptive parameters from conditioning
        scale, shift, gate = self._ada(cond).chunk(3, dim=-1)
        
        # Apply adaptive normalization: h = h * (1 + scale) + shift
        h = h * (1 + scale) + shift
        
        # Transform features
        h = self._fc(h)
        
        # Apply gated residual connection
        return x + h * gate


class CustomTransformerDecoderLayer(nn.Module):
    """
    Custom transformer decoder layer for trajectory prediction.
    
    This layer processes trajectory features by attending to multiple context sources:
    - BEV (Bird's Eye View) features for spatial scene understanding
    - Agent queries for interaction modeling
    - Ego vehicle queries for self-state awareness
    
    Time modulation allows the model to adjust predictions based on diffusion timestep.
    """

    def __init__(
        self,
        num_poses: int,
        d_model: int,
        d_ffn: int,
        config: FieryConfig,
        ego_fut_mode: int,
        enable_pooling: bool = False,
    ):
        """
        Initialize the decoder layer.
        
        Args:
            num_poses: Number of future trajectory poses to predict
            d_model: Model dimension for features
            d_ffn: Feed-forward network hidden dimension
            config: Configuration object with transformer parameters
            ego_fut_mode: Number of trajectory modes for ego vehicle
            enable_pooling: If True, use global pooling for BEV features instead of attention
        """
        super().__init__()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        
        # Cross-attention to other agents for interaction modeling
        self.cross_agent_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        
        # Cross-attention to ego vehicle state
        self.cross_ego_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        
        # Feed-forward network for feature transformation
        self.ffn = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )
        
        # Layer normalization after each sub-layer
        self.norm1 = nn.LayerNorm(config.tf_d_model)
        self.norm2 = nn.LayerNorm(config.tf_d_model)
        self.norm3 = nn.LayerNorm(config.tf_d_model)
        
        # Time-based feature modulation for diffusion process
        self.time_modulation = ModulationLayer(config.tf_d_model, 256)
        
        # Final trajectory refinement decoder
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=config.tf_d_model,
            ego_fut_ts=num_poses,
        )
        
        # BEV feature processing: either pooling or attention
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

    def forward(
        self,
        traj_feature: torch.Tensor,
        noisy_traj_points: torch.Tensor,
        bev_feature: torch.Tensor,
        bev_spatial_shape: Tuple[int, int],
        agents_query: torch.Tensor,
        ego_query: torch.Tensor,
        time_embed: torch.Tensor,
        status_encoding: torch.Tensor,
        global_img: Optional[torch.Tensor] = None,
        need_denormed: bool = False,
    ) -> torch.Tensor:
        """
        Process trajectory features through multiple attention mechanisms.
        
        Args:
            traj_feature: Trajectory embeddings [bs, num_samples, d_model]
            noisy_traj_points: Noisy trajectory coordinates [bs, num_samples, num_poses, 3]
            bev_feature: BEV features [bs, d_model, H, W] or [bs, H*W, d_model]
            bev_spatial_shape: Spatial dimensions (H, W) of BEV features
            agents_query: Agent context queries [bs, num_agents, d_model]
            ego_query: Ego vehicle query [bs, 1, d_model]
            time_embed: Timestep embedding [bs, 1, d_model]
            status_encoding: Vehicle status encoding
            global_img: Optional global image features
            need_denormed: Whether predictions need denormalization
            
        Returns:
            Refined trajectory predictions [bs, num_samples, num_poses, 3]
        """
        # Process BEV features based on chosen strategy
        if len(bev_feature.shape) == 4:
            if self.enable_pooling:
                # Global average pooling over spatial dimensions
                bev_feature = self.global_pool(bev_feature).squeeze(-1).squeeze(-1)
                num_samples = traj_feature.shape[1]
                # Expand to match number of trajectory samples
                bev_feature = bev_feature.unsqueeze(1).expand(-1, num_samples, -1)
            else:
                # Flatten spatial dimensions for attention
                bev_feature = bev_feature.flatten(-2, -1).permute(0, 2, 1)
        
        # Step 1: Incorporate BEV spatial context
        if self.enable_pooling:
            traj_feature = self.mlp_bev(traj_feature, bev_feature)
        else:
            traj_feature = self.cross_bev_attention(
                traj_feature, bev_feature, bev_feature
            )[0]
        
        # Step 2: Cross-attention with other agents for interaction modeling
        traj_feature = traj_feature + self.dropout(
            self.cross_agent_attention(
                traj_feature, agents_query, agents_query
            )[0]
        )
        traj_feature = self.norm1(traj_feature)
        
        # Step 3: Cross-attention with ego vehicle state
        traj_feature = traj_feature + self.dropout1(
            self.cross_ego_attention(
                traj_feature, ego_query, ego_query
            )[0]
        )
        traj_feature = self.norm2(traj_feature)
        
        # Step 4: Feed-forward transformation
        traj_feature = self.norm3(self.ffn(traj_feature))
        
        # Step 5: Time-conditioned modulation
        traj_feature = self.time_modulation(
            traj_feature, time_embed, global_cond=None, global_img=global_img
        )
        
        # Step 6: Predict trajectory offsets and refine coordinates
        poses_reg = self.task_decoder(traj_feature)  # [bs, num_samples, num_poses, 3]
        poses_reg = poses_reg + noisy_traj_points
        
        # Constrain heading angles to [-π, π] using tanh
        poses_reg[..., StateSE2Index.HEADING] = (
            poses_reg[..., StateSE2Index.HEADING].tanh() * np.pi
        )

        return poses_reg


class DiffusionTrajectoryHeadv2(nn.Module):
    """
    Diffusion-based trajectory prediction head using flow matching.
    
    This module predicts vehicle trajectories by iteratively denoising randomly
    sampled trajectories, conditioned on scene context, agent interactions, and
    ego vehicle state. It uses a flow matching framework for stable training.
    """

    def __init__(
        self,
        num_poses: int,
        d_ffn: int,
        d_model: int,
        config: FieryConfig = FieryConfig,
    ):
        """
        Initialize the diffusion trajectory head.
        
        Args:
            num_poses: Number of future poses to predict
            d_ffn: Feed-forward network dimension
            d_model: Model feature dimension
            config: Configuration object with diffusion and model parameters
        """
        super().__init__()
        
        # Query splitting configuration (ego + bounding boxes)
        self._query_splits = [1, config.num_bounding_boxes]
        
        # Store configuration and dimensions
        self._config = config
        self._num_poses = num_poses
        self._action_dim = 3  # x, y, heading
        self._d_model = d_model
        self._d_ffn = d_ffn
        
        # Training parameters
        self.diff_loss_weight = 2.0
        self.training_time_type = config.training_time_type
        self.scheduler_type = config.diffusion_scheduler_type
        self.diffusion_train_steps = config.diffusion_train_steps
        self.diffusion_inference_steps = config.diffusion_inference_steps
        self.ego_fut_mode = config.ego_fut_mode  # Number of trajectory modes

        # Initialize diffusion scheduler based on type
        if self.scheduler_type == 'ddim':
            self.diffusion_scheduler = DDIMScheduler(
                num_train_timesteps=config.diffusion_train_steps,
                beta_schedule="scaled_linear",
                prediction_type="sample",
            )
        elif self.scheduler_type == 'flow_matching_euler':
            self.diffusion_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=config.diffusion_train_steps
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")
        
        # Trajectory encoding network
        # Converts noisy trajectory points to high-dimensional features
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1, 512),
            nn.Linear(d_model, d_model),
        )
        
        # Time embedding network
        # Encodes diffusion timestep information
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )
        
        # Stacked transformer decoder layers
        diff_decoder_layer = CustomTransformerDecoderLayer(
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            config=config,
            ego_fut_mode=config.ego_fut_mode,
            enable_pooling=config.enable_pooling,
        )
        self.num_layers = 2
        self.diff_decoder = CustomTransformerDecoder(
            diff_decoder_layer, self.num_layers
        )

    def get_sigmas(
        self,
        timesteps: torch.Tensor,
        device: torch.device,
        n_dim: int = 4,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Get noise schedule sigmas for given timesteps.
        
        Sigmas control the interpolation between noise and data in flow matching.
        This method retrieves the appropriate sigma values from the scheduler.
        
        Args:
            timesteps: Diffusion timesteps [batch_size]
            device: Target device for computation
            n_dim: Number of dimensions to expand sigma to
            dtype: Data type for sigma values
            
        Returns:
            Noise schedule sigmas [batch_size, 1, 1, 1] (expanded to n_dim)
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

        # Expand dimensions to match required shape (for broadcasting)
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)

        return sigma

    def _prepare_model_input(
        self,
        x_target: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        pow: float = 0.5,
    ) -> torch.Tensor:
        """
        Prepare noisy model input by interpolating between noise and target.
        
        For flow matching, this creates an interpolation:
            noisy_input = sigma * noise + (1 - sigma) * target
        where sigma varies based on the timestep.
        
        Args:
            x_target: Clean trajectory targets [batch, num_modes, num_poses, 3]
            noise: Gaussian noise [batch, num_modes, num_poses, 3]
            timesteps: Current diffusion timesteps [batch]
            pow: Power for continuous time sampling
            
        Returns:
            Noisy trajectory input [batch, num_modes, num_poses, 3]
        """
        if "flow_matching" in self.scheduler_type:
            if self.training_time_type == "discrete":
                # Use predefined discrete timesteps
                sigma = self.get_sigmas(timesteps, x_target.device, n_dim=4)
            elif self.training_time_type == "continuous":
                # Sample continuous timesteps uniformly (with optional power scaling)
                n_dim = len(x_target.shape)
                sigma = torch.rand(x_target.shape[0], device=x_target.device)
                sigma = sigma ** pow  # Bias towards smaller sigmas
                sigma = torch.clamp_max(sigma, 1.0)
                # Expand to match input dimensions
                while len(sigma.shape) < n_dim:
                    sigma = sigma.unsqueeze(-1)
            else:
                raise NotImplementedError(
                    f"Unknown training_time_type: {self.training_time_type}"
                )
            
            # Linear interpolation between noise and target
            noisy_model_input = sigma * noise + (1.0 - sigma) * x_target
        else:
            # DDIM-style noise addition
            noisy_model_input = self.diffusion_scheduler.add_noise(
                x_target, noise, timesteps
            )
        
        return noisy_model_input

    def forward_train(
        self,
        ego_query: torch.Tensor,
        agents_query: torch.Tensor,
        bev_feature: torch.Tensor,
        bev_spatial_shape: Tuple[int, int],
        status_encoding: torch.Tensor,
        targets: Dict[str, torch.Tensor] = None,
        global_img: Optional[torch.Tensor] = None,
        tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass with diffusion loss computation.
        
        The model learns to predict the velocity field (flow) that transforms
        noise into the target trajectory distribution.
        
        Args:
            ego_query: Ego vehicle feature query [bs, 1, d_model]
            agents_query: Other agents' feature queries [bs, num_agents, d_model]
            bev_feature: BEV spatial features [bs, d_model, H, W]
            bev_spatial_shape: Spatial dimensions (H, W)
            status_encoding: Vehicle status encoding
            targets: Dictionary containing 'trajectory' key [bs, num_poses, 3]
            global_img: Optional global image features
            tokens: Optional tokenized features
            
        Returns:
            Dictionary containing:
                - trajectory: Predicted velocity field
                - trajectory_loss: Total loss value
                - trajectory_loss_dict: Per-layer losses
                - timesteps: Sampled timesteps
                - noisy_traj_points: Noisy inputs
                - sta_dict: Component losses (x, y, heading)
                - traj_dict: Trajectory statistics
                - targets: Target velocity field
        """
        device = ego_query.device

        # Extract and prepare trajectory targets
        x_target = targets["trajectory"]  # [bs, num_poses, 3]
        x_target = x_target.to(torch.float32)
        
        # Add mode dimension if not present
        if len(x_target.shape) == 3:
            x_target = x_target.unsqueeze(1)  # [bs, 1, num_poses, 3]
        
        assert len(x_target.shape) == 4, (
            f"Expected trajectory target shape (batch, num_modes, num_poses, 3), "
            f"but got {x_target.shape}"
        )
        
        # Normalize trajectories to standard range
        normed_x_target = norm_odo(x_target)
        batch_size = x_target.shape[0]

        # Sample random Gaussian noise
        noise = torch.randn_like(normed_x_target)  # ε ~ N(0, 1)
        
        # Sample random diffusion timesteps
        timesteps = torch.randint(
            1, self.diffusion_train_steps, (batch_size,), device=device
        ).long()
        
        # Create noisy trajectory by interpolating between noise and target
        noisy_traj_points = self._prepare_model_input(
            normed_x_target, noise, timesteps
        )

        # Step 1: Encode noisy trajectories as position embeddings
        traj_pos_embed = gen_sineembed_for_position(
            noisy_traj_points, hidden_dim=64
        )  # [bs, 1, num_poses, 64]
        traj_pos_embed = traj_pos_embed.flatten(-2)  # [bs, 1, num_poses*64]
        
        # Project to model dimension
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)
        traj_feature = traj_feature.view(batch_size, -1, self._d_model)

        # Step 2: Embed the diffusion timesteps
        time_embed = self.time_mlp(timesteps)  # [bs, d_model]
        time_embed = time_embed.view(batch_size, 1, -1)  # [bs, 1, d_model]

        # Step 3: Process through stacked transformer decoder
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
            need_denormed=True,
        )
        
        # Compute losses for each decoder layer
        trajectory_loss_dict = {}
        ret_traj_loss = 0
        
        for idx, poses_reg in enumerate(poses_reg_list):
            if self.scheduler_type == 'ddim':
                # Direct prediction loss
                trajectory_loss = F.l1_loss(poses_reg, normed_x_target)
            elif "flow_matching" in self.scheduler_type:
                # Velocity field prediction loss
                # v = (noise - target) represents the flow from target to noise
                v_target = noise - normed_x_target
                trajectory_loss = F.mse_loss(poses_reg, v_target)
            
            trajectory_loss_dict[f"trajectory_loss_{idx}"] = trajectory_loss
            ret_traj_loss += trajectory_loss

        # Extract final layer predictions
        v_prediction = poses_reg_list[-1]
        
        # Compute x0 (clean trajectory) from velocity prediction
        # x0 = noisy_input - timestep * velocity
        x0_prediction = noisy_traj_points - timesteps[:, None, None, None] * v_prediction
        
        # Calculate component-wise losses (x, y, heading separately)
        component_losses = calculate_component_losses(normed_x_target, x0_prediction)
        
        # Calculate trajectory statistics for monitoring
        trajectory_stats = calculate_statistics(
            noisy_traj_points=noisy_traj_points,
            normed_x_target=normed_x_target,
            x_target=x_target,
            model_prediction=v_prediction,
            model_x0_prediction=x0_prediction,
        )
        
        # Package outputs
        outputs = {
            "trajectory": v_prediction,
            "trajectory_loss": ret_traj_loss,
            "trajectory_loss_dict": trajectory_loss_dict,
            "timesteps": timesteps,
            "noisy_traj_points": noisy_traj_points,
            "sta_dict": {**component_losses},
            "traj_dict": trajectory_stats,
            "targets": v_target,
        }

        return outputs

    def p_sample_loop(
        self,
        ego_query: torch.Tensor,
        agents_query: torch.Tensor,
        bev_feature: torch.Tensor,
        bev_spatial_shape: Tuple[int, int],
        status_encoding: torch.Tensor,
        global_img: Optional[torch.Tensor],
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Iterative denoising loop to generate trajectory predictions.
        
        Starts from random Gaussian noise and iteratively refines it into
        a plausible trajectory by following the learned velocity field.
        
        Args:
            ego_query: Ego vehicle feature query [bs, 1, d_model]
            agents_query: Other agents' queries [bs, num_agents, d_model]
            bev_feature: BEV spatial features [bs, d_model, H, W]
            bev_spatial_shape: Spatial dimensions (H, W)
            status_encoding: Vehicle status encoding
            global_img: Optional global image features
            num_samples: Number of trajectory samples to generate
            
        Returns:
            Generated trajectories [bs, num_samples, num_poses, 3]
        """
        batch_size = ego_query.shape[0]
        device = ego_query.device
        shape = (batch_size, num_samples, self._num_poses, self._action_dim)
        
        # Initialize diffusion scheduler for inference
        self.diffusion_scheduler.set_timesteps(
            self.diffusion_inference_steps, device
        )
        
        # Start from pure Gaussian noise
        noisy_traj_points = torch.randn(shape, device=device)
        
        # Iteratively denoise
        for i, t in enumerate(self.diffusion_scheduler.timesteps):
            # Step 1: Encode current noisy trajectory
            traj_pos_embed = gen_sineembed_for_position(
                noisy_traj_points, hidden_dim=64
            )
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(batch_size, num_samples, -1)

            # Step 2: Prepare timestep tensor
            timesteps = t
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor(
                    [timesteps], dtype=torch.long, device=device
                )
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(device)
            
            # Step 3: Embed timesteps
            timesteps = timesteps.expand(batch_size)
            time_embed = self.time_mlp(timesteps)
            time_embed = time_embed.view(batch_size, 1, -1)

            # Step 4: Predict velocity field via transformer decoder
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
                need_denormed=True,
            )
            poses_reg = poses_reg_list[-1]  # Use final layer output

            # Step 5: Update trajectory using scheduler
            noisy_traj_points = self.diffusion_scheduler.step(
                model_output=poses_reg,
                timestep=t,
                sample=noisy_traj_points,
            ).prev_sample
        
        return noisy_traj_points

    @torch.no_grad()
    def forward_test(
        self,
        ego_query: torch.Tensor,
        agents_query: torch.Tensor,
        bev_feature: torch.Tensor,
        bev_spatial_shape: Tuple[int, int],
        status_encoding: torch.Tensor,
        global_img: Optional[torch.Tensor],
        tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference forward pass for trajectory generation.
        
        Args:
            ego_query: Ego vehicle feature query
            agents_query: Other agents' queries
            bev_feature: BEV spatial features
            bev_spatial_shape: Spatial dimensions
            status_encoding: Vehicle status encoding
            global_img: Optional global image features
            tokens: Optional tokenized features
            
        Returns:
            Dictionary containing:
                - trajectory: Predicted trajectory [bs, num_poses, 3]
        """
        # Generate trajectory samples via denoising
        pred_trajs = self.p_sample_loop(
            ego_query,
            agents_query,
            bev_feature,
            bev_spatial_shape,
            status_encoding,
            global_img,
            num_samples=self.ego_fut_mode,
        )
        
        # Select first sample (or could select best via scoring)
        trajectory = pred_trajs[:, 0]  # [bs, num_poses, 3]
        
        # Denormalize to real-world coordinates
        trajectory = denorm_odo(trajectory)

        return {"trajectory": trajectory}

    def forward(
        self,
        ego_query: torch.Tensor,
        agents_query: torch.Tensor,
        bev_feature: torch.Tensor,
        bev_spatial_shape: Tuple[int, int],
        status_encoding: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        global_img: Optional[torch.Tensor] = None,
        tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Main forward pass that routes to training or inference mode.
        
        Args:
            ego_query: Ego vehicle feature query
            agents_query: Other agents' queries
            bev_feature: BEV spatial features
            bev_spatial_shape: Spatial dimensions
            status_encoding: Vehicle status encoding
            targets: Optional ground truth trajectories (for training/evaluation)
            global_img: Optional global image features
            tokens: Optional tokenized features
            
        Returns:
            Dictionary of outputs (format depends on training/inference mode)
        """
        if self.training:
            return self.forward_train(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,targets,global_img, tokens=tokens)
        else:
            out_dict = self.forward_test(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,global_img)
            if targets is not None and "trajectory" in targets:
                pred_traj = out_dict['trajectory']
                x_target = targets["trajectory"]
                component_losses = calculate_component_losses(x_target, pred_traj)
                trajectory_stats = calculate_statistics(
                    x_target=x_target,
                    pred_traj=pred_traj,
                )
                out_dict.update(**component_losses)
                out_dict['traj_dict'] = trajectory_stats
            return out_dict
