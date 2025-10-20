import logging
import os
from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from omegaconf import DictConfig, OmegaConf

from .convolutions import UpsamplingConcat
from .temporal_model import TemporalModel, TemporalModelIdentity, TemporalModelConcat
from ..utils import (
    pack_sequence_dim, set_bn_momentum, unpack_sequence_dim,
    VoxelsSumming, calculate_birds_eye_view_parameters,
    cumulative_warp_features)

logger = logging.getLogger(__name__)


class Fiery(nn.Module):
    """
    FIERY encoder for E2E model, including backbone, feature-lifting and temporal model.
    Paper reference:
    https://arxiv.org/pdf/2104.10490.pdf
    Github reference:
    https://github.com/wayveai/fiery
    """

    def __init__(self,
                 cfg,
                 pretrained: bool = False,
                 freeze_encoder: bool = False):
        super().__init__()

        if isinstance(cfg, str) and os.path.isfile(cfg):
            cfg = OmegaConf.load(cfg)
        elif isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        elif not isinstance(cfg, DictConfig):
            raise TypeError(f"Unsupported cfg type: {type(cfg)}")

        cfg.model.encoder.pretrained = pretrained
        self.cfg = cfg

        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            self.cfg.lift.x_bound, self.cfg.lift.y_bound,
            self.cfg.lift.z_bound)
        # self.bev_resolution = nn.Parameter(bev_resolution, requires_grad=False)
        # self.bev_start_position = nn.Parameter(
        #     bev_start_position, requires_grad=False)
        # self.bev_dimension = nn.Parameter(bev_dimension, requires_grad=False)
        # self.bev_resolution = bev_resolution.requires_grad_(False)
        # self.bev_start_position = bev_start_position.requires_grad_(False)
        # self.bev_dimension = bev_dimension.requires_grad_(False)
        self.register_buffer("bev_resolution", bev_resolution)
        self.register_buffer("bev_start_position", bev_start_position)
        self.register_buffer("bev_dimension", bev_dimension)

        self.encoder_downsample = self.cfg.model.encoder.downsample
        self.encoder_out_channels = self.cfg.model.encoder.out_channels

        self.frustum = self.create_frustum()
        self.depth_channels, _, _, _ = self.frustum.shape

        if self.cfg.time_receptive_field == 1:
            assert self.cfg.model.temporal_model.name == 'identity'

        # temporal block
        self.receptive_field = self.cfg.time_receptive_field
        self.n_future = self.cfg.n_future_frames
        # self.latent_dim = self.cfg.MODEL.DISTRIBUTION.LATENT_DIM

        # Spatial extent in bird's-eye view, in meters
        self.spatial_extent = (self.cfg.lift.x_bound[1],
                               self.cfg.lift.y_bound[1])
        self.bev_size = (self.bev_dimension[0].item(),
                         self.bev_dimension[1].item())

        # Encoder
        self.encoder = Encoder(
            cfg=self.cfg.model.encoder, D=self.depth_channels)

        # Temporal model
        temporal_in_channels = self.encoder_out_channels
        if self.cfg.model.temporal_model.input_egopose:
            temporal_in_channels += 6
        if self.cfg.model.temporal_model.name == 'identity':
            self.temporal_model = TemporalModelIdentity(
                temporal_in_channels, self.receptive_field)
        elif cfg.model.temporal_model.name == 'temporal_block':
            self.temporal_model = TemporalModel(
                temporal_in_channels,
                self.receptive_field,
                input_shape=self.bev_size,
                start_out_channels=self.cfg.model.temporal_model.
                start_out_channels,
                extra_in_channels=self.cfg.model.temporal_model.
                extra_in_channels,
                n_spatial_layers_between_temporal_layers=self.cfg.model.
                temporal_model.inbetween_layers,
                use_pyramid_pooling=self.cfg.model.temporal_model.
                pyramid_pooling,
            )
        elif self.cfg.model.temporal_model.name == 'concat':
            self.temporal_model = TemporalModelConcat(
                temporal_in_channels, self.receptive_field)
        else:
            raise NotImplementedError(
                f'Temporal module {self.cfg.model.temporal_model.name}.')

        set_bn_momentum(self, self.cfg.model.bn_momentum)

        normalise_image = cfg.model.encoder.normalise_image
        if normalise_image is None or not normalise_image['enable']:
            self.normalise_image = False
        else:
            self.normalise_image = True
            self.register_buffer('image_mean', torch.tensor(normalise_image['mean']).view(1, 1, 1, 3, 1, 1))
            self.register_buffer('image_std', torch.tensor(normalise_image['std']).view(1, 1, 1, 3, 1, 1))

        if pretrained:
            self.load_pretrained_encoder(self.cfg.pretrained_path)

        if freeze_encoder:
            for param in self.parameters():
                param.requires_grad = False
        
        self.view_cnt = 0

    @staticmethod
    def clean_state_dict(state_dict: Dict):
        """
        'Clean' checkpoint by removing 'model' prefix from state dict
        """
        cleaned_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[6:] if k.startswith('model.') else k
            cleaned_state_dict[name] = v
        return cleaned_state_dict

    def load_pretrained_encoder(self, ckpt_path: str):
        """
        Load model state_dict and remove 'state_dict.model' prefix
        """
        assert os.path.isfile(ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict_key = ''
        if 'state_dict' in checkpoint:
            state_dict_key = 'state_dict'
        elif 'model' in checkpoint:
            state_dict_key = 'model'
        state_dict = self.clean_state_dict(
            checkpoint[state_dict_key] if state_dict_key else checkpoint)
        logger.info("Loaded {} from checkpoint '{}'".format(
            state_dict_key, ckpt_path))

        self.encoder.load_state_dict(state_dict, strict=False)

    def create_frustum(self):
        # Create grid in image plane
        h, w = self.cfg.image.final_dim
        downsampled_h, downsampled_w = h // self.encoder_downsample, w // self.encoder_downsample

        # Depth grid
        depth_grid = torch.arange(*self.cfg.lift.d_bound, dtype=torch.float)
        depth_grid = depth_grid.view(-1, 1, 1).expand(-1, downsampled_h,
                                                      downsampled_w)
        n_depth_slices = depth_grid.shape[0]

        # x and y grids
        x_grid = torch.linspace(0, w - 1, downsampled_w, dtype=torch.float)
        x_grid = x_grid.view(1, 1, downsampled_w).expand(
            n_depth_slices, downsampled_h, downsampled_w)
        y_grid = torch.linspace(0, h - 1, downsampled_h, dtype=torch.float)
        y_grid = y_grid.view(1, downsampled_h, 1).expand(
            n_depth_slices, downsampled_h, downsampled_w)

        # Dimension (n_depth_slices, downsampled_h, downsampled_w, 3)
        # containing data points in the image: left-right, top-bottom, depth
        frustum = torch.stack((x_grid, y_grid, depth_grid), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def forward(self, image, intrinsics, extrinsics, future_egomotion):

        start_idx = image.shape[1] - self.receptive_field
        # Only process features from the past and present
        image = image[:, start_idx:].contiguous()
        intrinsics = intrinsics[:, start_idx:].contiguous()
        extrinsics = extrinsics[:, start_idx:].contiguous()
        future_egomotion = future_egomotion[:, start_idx:].contiguous()

        if self.normalise_image:
            B, T, N, H, W, C = image.shape  # normalize inside model
            image = image.permute(0, 1, 2, 5, 3, 4) / 255.0            
            image_mean = self.image_mean.to(image.device)
            image_std = self.image_std.to(image.device)
            image = (image - image_mean) / image_std

        # Lifting features and project to bird's-eye view
        x, camera_feature = self.calculate_birds_eye_view_features(
            image, intrinsics, extrinsics)

        # Warp past features to the present's reference frame
        x = cumulative_warp_features(
            x.clone(),
            future_egomotion,
            mode='bilinear',
            spatial_extent=self.spatial_extent,
        )

        if self.cfg.model.temporal_model.input_egopose:
            b, s, c = future_egomotion.shape
            h, w = x.shape[-2:]
            future_egomotions_spatial = future_egomotion.view(
                b, s, c, 1, 1).expand(b, s, c, h, w)
            # at time 0, no egomotion so feed zero vector
            future_egomotions_spatial = torch.cat([
                torch.zeros_like(future_egomotions_spatial[:, :1]),
                future_egomotions_spatial[:, :(self.receptive_field - 1)]
            ],
                                                  dim=1)
            x = torch.cat([x, future_egomotions_spatial], dim=-3)

        #  Temporal model
        states = self.temporal_model(x)  # [B, T, C, H, W]

        N, T, C = states.shape[:3]
        states = states.view(N, T * C, *states.shape[3:])

        return states, camera_feature

    def get_geometry(self, intrinsics, extrinsics):
        """Calculate the (x, y, z) 3D position of the features.
        """
        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]
        B, N, _ = translation.shape
        # Add batch, camera dimension, and a dummy dimension at the end
        points = self.frustum.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        # Camera to ego reference frame
        points = torch.cat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
             points[:, :, :, :, :, 2:3]), 5)
        # combined_transformation = rotation.matmul(torch.inverse(intrinsics))  # a bug with torch1.9.0+cu111 on 4090, we cannot call inverse here
        # assert torch.all(intrinsics[:, :, 0, 0] == intrinsics[0, 0, 0, 0])
        # assert torch.all(intrinsics[:, :, 0, 0] == intrinsics[:, :, 1, 1])
        
        try:
            focal_length = intrinsics[:,:,0,0]
            inv_intrinsics = intrinsics.clone()
            inv_intrinsics[:, :, 0, 0] = 1.0 / focal_length
            inv_intrinsics[:, :, 1, 1] = 1.0 / focal_length
            inv_intrinsics[:, :, 0, 2] /= -focal_length
            inv_intrinsics[:, :, 1, 2] /= -focal_length
            combined_transformation = rotation.matmul(inv_intrinsics)
        except:
            #FIXME, ALF forwards a zero tensor to calculate tensor spce, which 
            # causes torch.inverse to raise error
            intrinsics_copy = intrinsics.clone().detach()
            intrinsics_copy[..., :3, :3] = torch.eye(3)
            combined_transformation = rotation.matmul(intrinsics_copy)

        points = combined_transformation.view(B, N, 1, 1, 1, 3,
                                              3).matmul(points.float()).squeeze(-1).half()
        points += translation.view(B, N, 1, 1, 1, 3)

        # The 3 dimensions in the ego reference frame are: (forward, sides,
        # height)
        return points

    def encoder_forward(self, x):
        # batch, n_cameras, channels, height, width
        b, n, c, h, w = x.shape

        x = x.view(b * n, c, h, w)
        x = self.encoder(x)
        x = x.view(b, n, *x.shape[1:])
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x

    def projection_to_birds_eye_view(self, x, geometry):
        """
        Adapted from https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L200
        """
        # batch, n_cameras, depth, height, width, channels
        batch, n, d, h, w, c = x.shape
        # remove camera features
        # zero_feature = torch.zeros_like(x)
        # zero_feature[:, self.view_cnt] = x[:, self.view_cnt]
        # x = zero_feature
        # self.view_cnt += 1
        # x[:, 3:] = torch.zeros((3, d, h, w, c), dtype=x.dtype, device=x.device)
        output = torch.zeros(
            (batch, c, self.bev_dimension[0], self.bev_dimension[1]),
            dtype=torch.float,
            device=x.device)

        # Number of 3D points
        N = n * d * h * w
        for b in range(batch):
            # flatten x
            x_b = x[b].reshape(N, c)

            # Convert positions to integer indices
            geometry_b = (
                (geometry[b] -
                 (self.bev_start_position - self.bev_resolution / 2.0)) /
                self.bev_resolution)
            geometry_b = geometry_b.view(N, 3).long()

            # Mask out points that are outside the considered spatial extent.
            mask = ((geometry_b[:, 0] >= 0)
                    & (geometry_b[:, 0] < self.bev_dimension[0])
                    & (geometry_b[:, 1] >= 0)
                    & (geometry_b[:, 1] < self.bev_dimension[1])
                    & (geometry_b[:, 2] >= 0)
                    & (geometry_b[:, 2] < self.bev_dimension[2]))
            x_b = x_b[mask]
            geometry_b = geometry_b[mask]

            # Sort tensors so that those within the same voxel are
            # consecutives.
            ranks = (
                geometry_b[:, 0] *
                (self.bev_dimension[1] * self.bev_dimension[2]) +
                geometry_b[:, 1] * (self.bev_dimension[2]) + geometry_b[:, 2])
            ranks_indices = ranks.argsort()
            x_b, geometry_b, ranks = x_b[ranks_indices], geometry_b[
                ranks_indices], ranks[ranks_indices]

            # Project to bird's-eye view by summing voxels.
            x_b, geometry_b = VoxelsSumming.apply(x_b, geometry_b, ranks)

            bev_feature = torch.zeros(
                (self.bev_dimension[2], self.bev_dimension[0],
                 self.bev_dimension[1], c),
                device=x_b.device)
            bev_feature[geometry_b[:, 2], geometry_b[:, 0],
                        geometry_b[:, 1]] = x_b

            # Put channel in second position and remove z dimension
            bev_feature = bev_feature.permute((0, 3, 1, 2))
            bev_feature = bev_feature.squeeze(0)

            output[b] = bev_feature

        return output

    def calculate_birds_eye_view_features(self, x, intrinsics, extrinsics):
        b, s, n, c, h, w = x.shape
        # Reshape
        x = pack_sequence_dim(x)
        intrinsics = pack_sequence_dim(intrinsics)
        extrinsics = pack_sequence_dim(extrinsics)

        geometry = self.get_geometry(intrinsics, extrinsics)
        camera_feature = self.encoder_forward(x)
        bev_feature = self.projection_to_birds_eye_view(camera_feature, geometry)
        bev_feature = unpack_sequence_dim(bev_feature, b, s)
        # [b, n, d, h, w, c] -> [b, n, c, d, h, w]
        camera_feature = camera_feature.permute(0, 1, 5, 2, 3, 4)
        return bev_feature, camera_feature


class Encoder(nn.Module):
    def __init__(self, cfg: DictConfig, D: int):
        super().__init__()
        self.D = D
        self.C = cfg.out_channels
        self.use_depth_distribution = cfg.use_depth_distribution
        self.downsample = cfg.downsample
        self.version = cfg.name.split('-')[1]

        if cfg.pretrained:
            self.backbone = EfficientNet.from_pretrained(cfg.name)
        else:
            self.backbone = EfficientNet.from_name(cfg.name)
        self.delete_unused_layers()

        if self.downsample == 16:
            if self.version == 'b0':
                upsampling_in_channels = 320 + 112
            elif self.version == 'b4':
                upsampling_in_channels = 448 + 160
            upsampling_out_channels = 512
        elif self.downsample == 8:
            if self.version == 'b0':
                upsampling_in_channels = 112 + 40
            elif self.version == 'b4':
                upsampling_in_channels = 160 + 56
            upsampling_out_channels = 128
        else:
            raise ValueError(
                f'Downsample factor {self.downsample} not handled.')

        self.upsampling_layer = UpsamplingConcat(upsampling_in_channels,
                                                 upsampling_out_channels)
        if self.use_depth_distribution:
            self.depth_layer = nn.Conv2d(
                upsampling_out_channels,
                self.C + self.D,
                kernel_size=1,
                padding=0)
        else:
            self.depth_layer = nn.Conv2d(
                upsampling_out_channels, self.C, kernel_size=1, padding=0)

    def delete_unused_layers(self):
        indices_to_delete = []
        for idx in range(len(self.backbone._blocks)):
            if self.downsample == 8:
                if self.version == 'b0' and idx > 10:
                    indices_to_delete.append(idx)
                if self.version == 'b4' and idx > 21:
                    indices_to_delete.append(idx)

        for idx in reversed(indices_to_delete):
            del self.backbone._blocks[idx]

        del self.backbone._conv_head
        del self.backbone._bn1
        del self.backbone._avg_pooling
        del self.backbone._dropout
        del self.backbone._fc

    def get_features(self, x):
        # Adapted from
        # https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.backbone._swish(
            self.backbone._bn0(self.backbone._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

            if self.downsample == 8:
                if self.version == 'b0' and idx == 10:
                    break
                if self.version == 'b4' and idx == 21:
                    break

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        if self.downsample == 16:
            input_1, input_2 = endpoints['reduction_5'], endpoints[
                'reduction_4']
        elif self.downsample == 8:
            input_1, input_2 = endpoints['reduction_4'], endpoints[
                'reduction_3']

        x = self.upsampling_layer(input_1, input_2)
        return x

    def forward(self, x):
        x = self.get_features(x)  # get feature vector

        x = self.depth_layer(x)  # feature and depth head

        if self.use_depth_distribution:
            depth = x[:, :self.D].softmax(dim=1)
            # outer product depth and features
            x = depth.unsqueeze(1) * x[:, self.D:
                                       (self.D + self.C)].unsqueeze(2)
        else:
            x = x.unsqueeze(2).repeat(1, 1, self.D, 1, 1)

        return x
    