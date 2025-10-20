from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


@dataclass
class FieryConfig:
    """Global TransFuser config."""

    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

    image_architecture: str = "resnet34"
    # bkb_path: str = "/mnt/nas25/wenxin.shao/workspace/DiffusionDrive/download/resnet34/pytorch_model.bin"
    plan_anchor_path: str = "/mnt/nas25/wenxin.shao/workspace/DiffusionDrive/download/kmeans_navsim_traj_20.npy"

    # encoder
    encoder_cfg: str = "navsim/agents/fiery/fiery_nuplan.yaml"
    encoder_pretrained: bool = True
    encoder_freeze: bool = False

    # bev encoder
    raster_num_input_channels: int = 64
    bev_embed_dims: int = 256
    bev_encoder_freeze: bool = False

    latent: bool = False
    latent_rad_thresh: float = 4 * np.pi / 9

    max_height_lidar: float = 100.0
    pixels_per_meter: float = 4.0
    hist_max_per_pixel: int = 5

    # image
    image_params = dict(
        original_height = 1080,
        original_width = 1920,
        resize_scale = 0.25,
        top_crop = 46,
        final_dim = [224, 480],
    )

    lidar_min_x: float = 0
    lidar_max_x: float = 32
    lidar_min_y: float = -32
    lidar_max_y: float = 32

    lidar_split_height: float = 0.2
    use_ground_plane: bool = False

    # new
    lidar_seq_len: int = 1

    lidar_resolution_width = 256
    lidar_resolution_height = 128

    perspective_downsample_factor = 1
    transformer_decoder_join = True
    detect_boxes = True
    use_bev_semantic = True
    use_semantic = False
    use_depth = False
    add_features = True

    # Transformer
    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0

    # detection
    num_bounding_boxes: int = 15

    # loss weights
    trajectory_weight: float = 12.0
    trajectory_cls_weight: float = 10.0
    trajectory_reg_weight: float = 8.0
    diff_loss_weight: float = 20.0
    agent_class_weight: float = 10.0
    agent_box_weight: float = 1.0
    bev_semantic_weight: float = 14.0
    use_ema: bool = False
    # BEV mapping
    bev_semantic_classes = {
        1: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),  # road
        2: ("polygon", [SemanticMapLayer.WALKWAYS]),  # walkways
        3: ("linestring", [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]),  # centerline
        4: (
            "box",
            [
                TrackedObjectType.CZONE_SIGN,
                TrackedObjectType.BARRIER,
                TrackedObjectType.TRAFFIC_CONE,
                TrackedObjectType.GENERIC_OBJECT,
            ],
        ),  # static_objects
        5: ("box", [TrackedObjectType.VEHICLE]),  # vehicles
        6: ("box", [TrackedObjectType.PEDESTRIAN]),  # pedestrians
    }

    bev_pixel_width: int = lidar_resolution_width
    bev_pixel_height: int = lidar_resolution_height
    bev_pixel_size: float = (lidar_max_x - lidar_min_x) / lidar_resolution_height

    num_bev_classes = 7
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2


    # optmizer
    weight_decay: float = 1e-4
    lr_steps = [70]
    optimizer_type = "AdamW"
    scheduler_type = "MultiStepLR"
    cfg_lr_mult = 0.5
    opt_paramwise_cfg = {
        "name":{
            "image_encoder":{
                "lr_mult": cfg_lr_mult
            }
        }
    }
    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        values = [self.lidar_min_x, self.lidar_max_x, self.lidar_min_y, self.lidar_max_y]
        return max([abs(value) for value in values])
