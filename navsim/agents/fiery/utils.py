
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn


def pack_sequence_dim(x):
    b, s = x.shape[:2]
    return x.view(b * s, *x.shape[2:])


def unpack_sequence_dim(x, b, s):
    return x.view(b, s, *x.shape[1:])


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = momentum


def mat2pose_vec(matrix: torch.Tensor):
    """
    Convert a 4x4 pose matrix into a 6-dof pose vector
    :param matrix (ndarray): 4x4 pose matrix in torch tensor format.
    :return vector (ndarray): 6-dof pose vector comprising translation components (tx, ty, tz) and
                              rotation components (rx, ry, rz).
    """

    # M[1, 2] = -sinx*cosy, M[2, 2] = +cosx*cosy
    rotx = torch.atan2(-matrix[..., 1, 2], matrix[..., 2, 2])

    # M[0, 2] = +siny, M[1, 2] = -sinx*cosy, M[2, 2] = +cosx*cosy
    cosy = torch.sqrt(matrix[..., 1, 2]**2 + matrix[..., 2, 2]**2)
    roty = torch.atan2(matrix[..., 0, 2], cosy)

    # M[0, 0] = +cosy*cosz, M[0, 1] = -cosy*sinz
    rotz = torch.atan2(-matrix[..., 0, 1], matrix[..., 0, 0])

    rotation = torch.stack((rotx, roty, rotz), dim=-1)

    # Extract translation params
    translation = matrix[..., :3, 3]
    return torch.cat((translation, rotation), dim=-1)


def mat2pose_vec_np(matrix: np.ndarray):
    """
    Convert a 4x4 pose matrix into a 6-dof pose vector (numpy version)
    :param matrix (ndarray): 4x4 pose matrix in numpy array format.
    :return vector (ndarray): 6-dof pose vector comprising translation components (tx, ty, tz) and
                              rotation components (rx, ry, rz).
    """

    # M[1, 2] = -sinx*cosy, M[2, 2] = +cosx*cosy
    rotx = np.arctan2(-matrix[..., 1, 2], matrix[..., 2, 2])

    # M[0, 2] = +siny, M[1, 2] = -sinx*cosy, M[2, 2] = +cosx*cosy
    cosy = np.sqrt(matrix[..., 1, 2]**2 + matrix[..., 2, 2]**2)
    roty = np.arctan2(matrix[..., 0, 2], cosy)

    # M[0, 0] = +cosy*cosz, M[0, 1] = -cosy*sinz
    rotz = np.arctan2(-matrix[..., 0, 1], matrix[..., 0, 0])

    rotation = np.stack((rotx, roty, rotz), axis=-1)

    # Extract translation params
    translation = matrix[..., :3, 3]
    return np.concatenate((translation, rotation), axis=-1)


def euler2mat(angle: torch.Tensor):
    """Convert euler angles to rotation matrix.
    Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    :param angle: rotation angle along 3 axis (in radians) [Bx3]
    :return: Rotation matrix corresponding to the euler angles [Bx3x3]
    """
    shape = angle.shape
    angle = angle.view(-1, 3)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = torch.zeros_like(z)
    ones = torch.ones_like(z)
    zmat = torch.stack(
        [cosz, -sinz, zeros, sinz, cosz, zeros, zeros, zeros, ones],
        dim=1).view(-1, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack(
        [cosy, zeros, siny, zeros, ones, zeros, -siny, zeros, cosy],
        dim=1).view(-1, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack(
        [ones, zeros, zeros, zeros, cosx, -sinx, zeros, sinx, cosx],
        dim=1).view(-1, 3, 3)

    rot_mat = xmat.bmm(ymat).bmm(zmat)
    rot_mat = rot_mat.view(*shape[:-1], 3, 3)
    return rot_mat


def pose_vec2mat(vec: torch.Tensor):
    """
    Convert 6DoF parameters to transformation matrix.
    :param vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz [B,6]
    :return: A transformation matrix [B,4,4]
    """
    translation = vec[..., :3].unsqueeze(-1)  # [...x3x1]
    rot = vec[..., 3:].contiguous()  # [...x3]
    rot_mat = euler2mat(rot)  # [...,3,3]
    transform_mat = torch.cat([rot_mat, translation], dim=-1)  # [...,3,4]
    transform_mat = torch.nn.functional.pad(
        transform_mat, [0, 0, 0, 1], value=0)  # [...,4,4]
    transform_mat[..., 3, 3] = 1.0
    return transform_mat


def warp_features(
        x: torch.Tensor,
        flow: Optional[torch.Tensor],
        mode='nearest',
        spatial_extent=None,
):
    """
    Applies a rotation and translation to a feature map. Used to warp BEV feature from
    previous frames to the current frame.
    :param x: (b, c, h, w) feature map
    :param flow: (b, 6) 6DoF vector (only uses the xy poriton)
    :param mode: use 'nearest' when dealing with categorical inputs
    :return: in plane transformed feature map
    """
    if flow is None:
        return x
    b, c, h, w = x.shape
    # z-rotation
    angle = flow[:, 5].clone()  # torch.atan2(flow[:, 1, 0], flow[:, 0, 0])
    # x-y translation
    translation = flow[:, :2].clone()  # flow[:, :2, 3]

    # Normalise translation. Need to divide by how many meters is half of the image.
    # because translation of 1.0 correspond to translation of half of the
    # image.
    translation[:, 0] /= spatial_extent[0]
    translation[:, 1] /= spatial_extent[1]
    # forward axis is inverted
    translation[:, 0] *= -1

    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)

    # output = Rot.input + translation
    # tx and ty are inverted as is the case when going from real coordinates to numpy coordinates
    # translation_pos_0 -> positive value makes the image move to the left
    # translation_pos_1 -> positive value makes the image move to the top
    # Angle -> positive value in rad makes the image move in the trigonometric
    # way
    transformation = torch.stack([
        cos_theta, -sin_theta, translation[:, 1], sin_theta, cos_theta,
        translation[:, 0]
    ],
                                 dim=-1).view(b, 2, 3)

    # Note that a rotation will preserve distances only if height = width. Otherwise there's
    # resizing going on. e.g. rotation of pi/2 of a 100x200 image will make what's in the center of the image
    # elongated.
    grid = torch.nn.functional.affine_grid(
        transformation, size=x.shape, align_corners=False)
    warped_x = torch.nn.functional.grid_sample(
        x, grid.float(), mode=mode, padding_mode='zeros', align_corners=False)

    return warped_x


def cumulative_warp_features(
        x: torch.Tensor,
        flow: Optional[torch.Tensor],
        mode='nearest',
        spatial_extent=None,
):
    """
    Warp a sequence of feature maps by accumulating incremental 2d flow.
    Used in FIERY model after backbone to merge BEV features from context frames.

    x[:, -1] remains unchanged
    x[:, -2] is warped using flow[:, -2]
    x[:, -3] is warped using flow[:, -3] @ flow[:, -2]
    ...
    x[:, 0] is warped using flow[:, 0] @ ... @ flow[:, -3] @ flow[:, -2]

    :param x: (b, t, c, h, w) sequence of feature maps
    :param flow: (b, t, 6) sequence of 6 DoF pose
                 from t to t+1 (only uses the xy poriton)
    """
    sequence_length = x.shape[1]
    if sequence_length == 1:
        return x

    flow = pose_vec2mat(flow)

    out = [x[:, -1]]
    cum_flow = flow[:, -2]
    for t in reversed(range(sequence_length - 1)):
        out.append(
            warp_features(
                x[:, t],
                mat2pose_vec(cum_flow),
                mode=mode,
                spatial_extent=spatial_extent))
        # @ is the equivalent of torch.bmm
        cum_flow = flow[:, t - 1] @ cum_flow

    return torch.stack(out[::-1], 1)


def calculate_birds_eye_view_parameters(
        x_bounds: List,
        y_bounds: List,
        z_bounds: List,
):
    """
    Calculate BEV related paramters for FIERY based on settings along 3 axes.

    :param x_bounds: Range and resolution along x-axis.
    :param y_bounds: Range and resolution along y-axis.
    :param z_bounds: Range and resolution along z-axis.

    :return bev_resolution: Bird's-eye view bev_resolution
    :return bev_start_position: Bird's-eye view first element
    :return bev_dimension: Bird's-eye view tensor spatial dimension
    """
    bev_resolution = torch.tensor(
        [row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor(
        [row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = torch.tensor(
        [(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]],
        dtype=torch.long)

    return bev_resolution, bev_start_position, bev_dimension


class VoxelsSumming(torch.autograd.Function):
    """
    Project camera features to BEV by summing voxels.
    Adapted from https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/tools.py#L193.
    """

    @staticmethod
    def forward(ctx, x, geometry, ranks):
        """The features `x` and `geometry` are ranked by voxel positions."""
        # Cumulative sum of all features.
        x = x.cumsum(0)

        # Indicates the change of voxel.
        mask = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        mask[:-1] = ranks[1:] != ranks[:-1]

        x, geometry = x[mask], geometry[mask]
        # Calculate sum of features within a voxel.
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        ctx.save_for_backward(mask)
        ctx.mark_non_differentiable(geometry)

        return x, geometry

    @staticmethod
    def backward(ctx, grad_x, grad_geometry):
        (mask, ) = ctx.saved_tensors
        # Since the operation is summing, we simply need to send gradient
        # to all elements that were part of the summation process.
        indices = torch.cumsum(mask, 0)
        indices[mask] -= 1

        output_grad = grad_x[indices]

        return output_grad, None, None
