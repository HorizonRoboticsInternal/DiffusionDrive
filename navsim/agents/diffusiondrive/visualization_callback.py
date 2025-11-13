import copy
import math
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import torch
import torch.utils.data
import yaml
from matplotlib import cm
from torch.utils.tensorboard import SummaryWriter

from nuplan.planning.training.modeling.types import (
    FeaturesType,
    TargetsType,
)
from navsim.agents.fiery.fiery_config import FieryConfig
from navsim.planning.training.dataset import CacheOnlyDataset


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def read_yaml(yaml_path: str) -> Dict[str, Any]:
    """Load a YAML file into a dictionary.

    Args:
        yaml_path: Path to a YAML configuration file.

    Returns:
        Parsed YAML content as a Python dictionary.
    """
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def _batch_abstract_features(
    reference_features: FeaturesType,
    features_to_batch: List[FeaturesType],
) -> FeaturesType:
    """Batch feature dictionaries with a custom collate function.

    The function assumes that all feature dictionaries share the same keys and
    types. Tensors are stacked along a new batch dimension, while all other
    types are collected into Python lists.

    Args:
        reference_features: Feature dictionary used only for key/type reference.
        features_to_batch: List of feature dictionaries to batch.

    Returns:
        A new feature dictionary containing batched values.
    """
    batched: Dict[str, Any] = {}

    for key, ref_value in reference_features.items():
        values = [sample[key] for sample in features_to_batch]

        if isinstance(ref_value, torch.Tensor):
            batched[key] = torch.stack(values, dim=0)
        else:
            batched[key] = values

    return batched


class FeatureCollate:
    """Wrapper class that collates multiple samples into a batch.

    This collate function operates on a list of ``(features, targets)`` tuples
    where each element is a mapping of string keys to tensors or other
    serializable objects.
    """

    def __call__(
        self, batch: List[Tuple[FeaturesType, TargetsType]]
    ) -> Tuple[FeaturesType, TargetsType]:
        """Collate a list of ``(features, targets)`` into batched structures.

        Args:
            batch: List of ``(features, targets)`` tuples.

        Returns:
            A tuple ``(batched_features, batched_targets)``.
        """
        if not batch:
            raise ValueError("Batch size has to be greater than 0!")

        features_list = [item[0] for item in batch]
        targets_list = [item[1] for item in batch]

        reference_features, reference_targets = batch[0]

        batched_features = _batch_abstract_features(reference_features, features_list)
        batched_targets = _batch_abstract_features(reference_targets, targets_list)

        return batched_features, batched_targets


# ---------------------------------------------------------------------------
# Geometry / coordinate utilities
# ---------------------------------------------------------------------------


def get_global_to_local_transform(pose: Iterable[float]) -> npt.NDArray[np.float32]:
    """Compute the 2D homogeneous transform from global to local coordinates.

    The pose is assumed to be ``(x, y, heading)`` where the heading is in
    radians in the global frame.

    Args:
        pose: Pose of the local frame expressed in the global frame as
            ``(x, y, heading)``.

    Returns:
        A 3x3 homogeneous transformation matrix (global → local).
    """
    x, y, heading = pose
    c = math.cos(heading)
    s = math.sin(heading)

    transform = np.array(
        [
            [c, s, -c * x - s * y],
            [-s, c, s * x - c * y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return transform


def global_to_image_coord(
    coords: npt.NDArray[np.floating],
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    img: npt.NDArray[np.uint8],
) -> npt.NDArray[np.int32]:
    """Map world coordinates to pixel coordinates in an image.

    The mapping assumes a simple axis-aligned rectangular region in world
    coordinates, which is linearly mapped to the image extent.

    Args:
        coords: Array of shape ``(..., 2)`` with world coordinates ``(x, y)``.
        x0: Minimum x in world coordinates.
        y0: Minimum y in world coordinates.
        x1: Maximum x in world coordinates.
        y1: Maximum y in world coordinates.
        img: Target image whose height/width define the pixel grid.

    Returns:
        Integer pixel coordinates as an array with the same leading shape as
        ``coords`` and final dimension 2.
    """
    img_height, img_width = img.shape[:2]

    coords_img = coords.astype(np.float32).copy()
    coords_img[..., 0] = (coords_img[..., 0] - x0) / (x1 - x0) * img_width
    coords_img[..., 1] = (y1 - coords_img[..., 1]) / (y1 - y0) * img_height

    return coords_img.astype(np.int32)


def world_to_pixel(
    x: float,
    y: float,
    x_range: Tuple[float, float] = (0.0, 32.0),
    y_range: Tuple[float, float] = (-32.0, 32.0),
    resolution: float = 0.25,
) -> Tuple[int, int]:
    """Convert ego/world coordinates to pixel indices on a flipped BEV image.

    The BEV image is assumed to be 128x256 with a resolution of 0.25m per
    pixel, where the ego vehicle initially sits near the center of the image
    after flipping.

    Args:
        x: X position in meters (forward).
        y: Y position in meters (left).
        x_range: World x-span covered by the BEV image.
        y_range: World y-span covered by the BEV image.
        resolution: Spatial resolution in meters per pixel.

    Returns:
        Pixel coordinates ``(v, u)`` suitable for indexing into an image.
    """
    px = (x - x_range[0]) / resolution  # [0, 32]m → [0, 128)
    py = (y - y_range[0]) / resolution  # [-32, 32]m → [0, 256)

    u = 128 - int(np.floor(px)) - 1  # flip vertically (x)
    v = 256 - int(np.floor(py)) - 1
    return v, u


def draw_bev_box(
    img: npt.NDArray[np.uint8],
    x: float,
    y: float,
    heading: float,
    length: float,
    width: float,
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 1,
) -> None:
    """Draw a rotated bounding box in BEV space onto an image.

    The box is defined in the ego frame where ``x`` is forward and ``y`` is
    left.

    Args:
        img: BEV image to draw on (modified in-place).
        x: Center x position of the box in meters.
        y: Center y position of the box in meters.
        heading: Yaw angle in radians.
        length: Box length in meters.
        width: Box width in meters.
        color: BGR color for the box outline.
        thickness: Line thickness in pixels.
    """
    half_l, half_w = length / 2.0, width / 2.0

    # Corners in ego frame: [x_fwd, y_left]
    corners = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ],
        dtype=np.float32,
    )

    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    corners = corners @ rot.T + np.array([x, y], dtype=np.float32)

    pixel_corners = [world_to_pixel(cx, cy) for cx, cy in corners]
    cv2.polylines(
        img,
        [np.array(pixel_corners, dtype=np.int32)],
        isClosed=True,
        color=color,
        thickness=thickness,
    )


def ego_trajectory_to_box(
    ego_trajectory: torch.Tensor,
    ego_width: float,
    ego_front_length: float,
    ego_rear_length: float,
) -> torch.Tensor:
    """Convert an ego trajectory into oriented bounding boxes.

    Args:
        ego_trajectory: Tensor of shape ``(B, num_modes, num_frames, 3)`` with
            elements ``(x, y, heading)``.
        ego_width: Width of the ego vehicle.
        ego_front_length: Length from the vehicle center to the front bumper.
        ego_rear_length: Length from the vehicle center to the rear bumper.

    Returns:
        Tensor of shape ``(B, num_modes, num_frames, 5)`` where each element is
        ``(center_x, center_y, heading, half_length, half_width)``.
    """
    ego_x = ego_trajectory[..., 0]
    ego_y = ego_trajectory[..., 1]
    ego_heading = ego_trajectory[..., 2]

    c = torch.cos(ego_heading)
    s = torch.sin(ego_heading)

    half_length = (ego_rear_length + ego_front_length) / 2.0
    half_width = ego_width / 2.0
    d = half_length - ego_rear_length

    x = ego_x + d * c
    y = ego_y + d * s

    half_length_tensor = torch.full_like(ego_x, half_length)
    half_width_tensor = torch.full_like(ego_x, half_width)

    return torch.stack([x, y, ego_heading, half_length_tensor, half_width_tensor], dim=-1)


# ---------------------------------------------------------------------------
# Visualization callback
# ---------------------------------------------------------------------------


class VisualizationCallback(pl.Callback):
    """Callback that visualizes planner model inputs/outputs in TensorBoard.

    The callback samples a fixed set of training/validation examples once and
    reuses them across epochs, making it easy to visually compare model
    behavior over time.
    """

    def __init__(
        self,
        config: FieryConfig,
        target_width: int = 256,
        target_height: int = 512,
        images_per_tile: int = 4,
        num_train_tiles: int = 4,
        num_val_tiles: int = 4,
        frequency: int = 1,
        skip_train: bool = False,
        cache_path: str = "/mnt/cwai/hpfs0/navsim/fiery_training_cache_part_1_v2",
        train_val_test_log_split_yaml: str = "/mnt/cwai/hpfs0/kun.li/DiffusionDrive/navsim/planning/script/config/training/default_train_val_test_log_split.yaml",
    ) -> None:
        """Initialize the visualization callback.

        Args:
            config: Global Fiery configuration object.
            target_width: Target width of composed visualization images.
            target_height: Target height of composed visualization images.
            images_per_tile: Number of images per tile in a grid.
            num_train_tiles: Number of tiles from the training set.
            num_val_tiles: Number of tiles from the validation set.
            frequency: Visualization frequency in epochs.
            skip_train: If ``True``, skip visualization on the training split.
            cache_path: Path to the dataset cache.
            train_val_test_log_split_yaml: Path to the YAML file specifying
                the train/val/test log splits.
        """
        super().__init__()

        self._config = config
        self.target_width = target_width
        self.target_height = target_height

        self.custom_batch_size = images_per_tile
        self.num_train_images = num_train_tiles * images_per_tile
        self.num_val_images = num_val_tiles * images_per_tile

        self.frequency = frequency
        self.skip_train = skip_train

        self.cache_path = cache_path
        self.train_val_test_log_split_yaml = train_val_test_log_split_yaml

        self.train_dataloader: Optional[torch.utils.data.DataLoader] = None
        self.val_dataloader: Optional[torch.utils.data.DataLoader] = None

        self.ego_width = config.ego_width
        self.ego_length = config.ego_front_length + config.ego_rear_length

    # ------------------------------------------------------------------
    # Data loading utilities
    # ------------------------------------------------------------------

    def _initialize_dataloaders(self, pl_module: pl.LightningModule) -> None:
        """Initialize deterministic train/val dataloaders for visualization.

        The same subset of samples is used across epochs for easier visual
        comparison.
        """
        log_split = read_yaml(self.train_val_test_log_split_yaml)

        train_set = CacheOnlyDataset(
            cache_path=self.cache_path,
            feature_builders=pl_module.agent.get_feature_builders(),
            target_builders=pl_module.agent.get_target_builders(),
            log_names=log_split["train_logs"],
        )

        val_set = CacheOnlyDataset(
            cache_path=self.cache_path,
            feature_builders=pl_module.agent.get_feature_builders(),
            target_builders=pl_module.agent.get_target_builders(),
            log_names=log_split["val_logs"],
        )

        self.train_dataloader = self._create_dataloader(train_set, self.num_train_images)
        self.val_dataloader = self._create_dataloader(val_set, self.num_val_images)

    def _create_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        num_samples: int,
    ) -> torch.utils.data.DataLoader:
        """Create a small deterministic dataloader used only for visualization."""
        dataset_size = len(dataset)
        num_keep = min(dataset_size, num_samples)

        # Deterministic sampling for reproducible visualization
        indices = random.sample(range(dataset_size), num_keep)
        subset = torch.utils.data.Subset(dataset=dataset, indices=indices)

        return torch.utils.data.DataLoader(
            dataset=subset,
            batch_size=self.custom_batch_size,
            collate_fn=FeatureCollate(),
        )

    # ------------------------------------------------------------------
    # Core visualization logic
    # ------------------------------------------------------------------

    def _log_from_dataloader(
        self,
        pl_module: pl.LightningModule,
        dataloader: torch.utils.data.DataLoader,
        logger: Any,
        training_step: int,
        prefix: str,
    ) -> None:
        """Visualize and log all batches from a given dataloader.

        Args:
            pl_module: Lightning module used for inference.
            dataloader: DataLoader providing visualization samples.
            logger: Trainer logger or raw ``SummaryWriter``.
            training_step: Global training step.
            prefix: Prefix for TensorBoard tags (e.g., ``"train"`` or ``"val"``).
        """
        for batch_idx, batch in enumerate(dataloader):
            features, targets = batch

            features = self._move_features_type_to_device(features, pl_module.device)
            targets = self._move_features_type_to_device(targets, pl_module.device)

            predictions = self._infer_model(pl_module, features, targets)

            # Move back to CPU for NumPy-based visualization
            features_cpu = self._move_features_type_to_device(features, torch.device("cpu"))
            targets_cpu = self._move_features_type_to_device(targets, torch.device("cpu"))

            self._log_batch(
                logger=logger,
                features=features_cpu,
                targets=targets_cpu,
                predictions=predictions,
                batch_idx=batch_idx,
                training_step=training_step,
                prefix=prefix,
            )

    def _infer_model(
        self,
        pl_module: pl.LightningModule,
        features: FeaturesType,
        targets: TargetsType,
    ) -> TargetsType:
        """Run a forward pass using the underlying agent in evaluation mode."""
        pl_module.eval()
        with torch.no_grad():
            predictions = pl_module.agent.forward(features, targets)
            predictions = self._move_features_type_to_device(predictions, torch.device("cpu"))
        pl_module.train()
        return predictions

    def _log_batch(
        self,
        logger: Any,
        features: FeaturesType,
        targets: TargetsType,
        predictions: TargetsType,
        batch_idx: int,
        training_step: int,
        prefix: str,
    ) -> None:
        """Visualize and log a single batch of data.

        Args:
            logger: Trainer logger or TensorBoard ``SummaryWriter``.
            features: Input features on CPU.
            targets: Ground-truth targets on CPU.
            predictions: Model predictions on CPU.
            batch_idx: Batch index within the visualization dataloader.
            training_step: Global training step.
            prefix: Tag prefix (e.g., ``"train"`` or ``"val"``).
        """
        # ------------------------------------------------------------------
        # Extract and cache all CPU numpy arrays
        # ------------------------------------------------------------------
        cameras = features["cameras"][:, -1].numpy()            # [B, C, H, W, 3]
        target_traj = targets["trajectory"].numpy()             # [B, T, 3]
        pred_traj = predictions["trajectory"].numpy()

        target_bev = targets["bev_semantic_map"].numpy()        # [B, H, W]
        pred_bev = predictions["bev_semantic_map"].numpy()      # [B, C, H, W]

        gt_states = targets["agent_states"].numpy()              # [B, N, 5]
        gt_logits = targets["agent_labels"].numpy().astype(np.float32)

        pred_states = predictions["agent_states"].numpy()        # [B, N, 5]
        pred_logits = predictions["agent_labels"].numpy().astype(np.float32)

        tokens = features.get("token", [])

        # ------------------------------------------------------------------
        # Pre‑compute colormap once per batch
        # ------------------------------------------------------------------
        num_classes = pred_bev.shape[1]
        cmap = cm.get_cmap("tab20", num_classes)
        class_colors = (cmap(np.arange(num_classes))[:, :3] * 255).astype(np.uint8)

        # ------------------------------------------------------------------
        # Precompute sigmoid for agent probabilities (vectorized)
        # ------------------------------------------------------------------
        pred_probs = 1.0 / (1.0 + np.exp(-pred_logits))   # [B, N]
        gt_probs = 1.0 / (1.0 + np.exp(-gt_logits))       # [B, N]

        pred_masks = pred_probs > 0.5                     # boolean [B, N]
        gt_masks = gt_probs > 0.5

        # ------------------------------------------------------------------
        # Loop through samples
        # ------------------------------------------------------------------
        batch_images = []
        ego_state = (0.0, 0.0, 0.0)
        for i in range(len(pred_bev)):
            # -----------------------------
            # Camera image
            # -----------------------------
            front_view = cameras[i, 1]  # [H, W, 3]

            # -----------------------------
            # Semantic BEV
            # -----------------------------
            bev_logits = pred_bev[i]                   # [C, H, W]
            bev_class = np.argmax(bev_logits, axis=0)  # [H, W]
            bev_img = class_colors[bev_class]

            # flip to match BEV orientation
            bev_img = np.flip(bev_img, (0, 1)).copy()

            # -----------------------------
            # Draw predicted + GT agents
            # -----------------------------
            for agent in pred_states[i][pred_masks[i]]:
                x, y, heading, L, W = map(float, agent)
                draw_bev_box(bev_img, x, y, heading, L, W, color=(0, 255, 0))

            for agent in gt_states[i][gt_masks[i]]:
                x, y, heading, L, W = map(float, agent)
                draw_bev_box(bev_img, x, y, heading, L, W, color=(255, 0, 0))

            # ego
            draw_bev_box(
                bev_img,
                *ego_state,
                length=self.ego_length,
                width=self.ego_width,
                color=(0, 0, 0),
            )

            # -----------------------------
            # Draw trajectories
            # -----------------------------
            bev_img = self._render_trajectory_bev_space(
                bev_img, pred_traj[i], ego_state, color=(0, 0, 0), radius=1
            )
            bev_img = self._render_trajectory_bev_space(
                bev_img, target_traj[i], ego_state, color=(255, 0, 0), radius=1
            )

            # -----------------------------
            # Harmonize camera + BEV width
            # -----------------------------
            target_width = max(front_view.shape[1], bev_img.shape[1])

            if front_view.shape[1] != target_width:
                scale = target_width / front_view.shape[1]
                front_view = cv2.resize(
                    front_view,
                    (target_width, int(front_view.shape[0] * scale)),
                    interpolation=cv2.INTER_NEAREST,
                )

            if bev_img.shape[1] != target_width:
                scale = target_width / bev_img.shape[1]
                bev_img = cv2.resize(
                    bev_img,
                    (target_width, int(bev_img.shape[0] * scale)),
                    interpolation=cv2.INTER_NEAREST,
                )

            batch_images.append(np.vstack([front_view, bev_img]).astype(np.uint8))

        # ------------------------------------------------------------------
        # Write batch to TensorBoard
        # ------------------------------------------------------------------
        logger.experiment.add_images(
                    tag=f"{prefix}_visualization_{batch_idx}",
                    img_tensor=torch.from_numpy(np.stack(batch_images, axis=0)),
                    global_step=training_step,
                    dataformats='NHWC',
                )

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def _render_trajectory_bev_space(
        self,
        image: npt.NDArray[np.uint8],
        traj: npt.NDArray[np.floating],
        ego_state: Tuple[float, float, float],
        color: Tuple[int, int, int],
        thickness: int = -1,
        radius: int = 2,
    ) -> npt.NDArray[np.uint8]:
        """Render an ego trajectory onto a BEV image.

        Args:
            image: BEV image (modified in-place and also returned).
            traj: Trajectory points in global coordinates with shape ``(T, 3)``
                or ``(T, 2)`` where the first two dimensions are ``(x, y)``.
            ego_state: Ego pose ``(x, y, heading)`` in global coordinates.
            color: BGR color for the rendered trajectory.
            thickness: Line thickness (``cv2.circle`` argument).
            radius: Circle radius.

        Returns:
            The input image after rendering the trajectory.
        """
        if traj is None or not np.any(traj):
            return image

        x0, y0, x1, y1 = -100.0, 0.0, 100.0, 100.0  # render 200m x 200m around ego

        pts = np.vstack([traj[:, 0], traj[:, 1]]).T  # (N, 2) in global frame
        transform = get_global_to_local_transform(ego_state)  # (3, 3)

        pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
        pts_e = (transform @ pts_h.T).T[:, :2]  # (N, 2): [x_fwd, y_left]

        # ego -> plot (x_right, y_up)
        pts_plot = np.c_[-pts_e[:, 1], pts_e[:, 0]]  # [-y_left, x_fwd]

        # plot -> image pixels (origin top-left)
        coords = global_to_image_coord(pts_plot, x0, y0, x1, y1, image)

        for u, v in coords:
            cv2.circle(image, (int(u), int(v)), radius=radius, color=color, thickness=thickness)

        return image

    @staticmethod
    def _move_features_type_to_device(
        batch: Mapping[str, Any],
        device: torch.device,
    ) -> FeaturesType:
        """Move all tensor values in a ``FeaturesType``-like mapping to a device.

        Non-tensor values are passed through unchanged. This function is kept
        intentionally shallow (no recursion) to match the structure used in
        nuPlan/NavSim features and targets.
        """
        output: Dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                output[key] = value.to(device)
            else:
                output[key] = value
        return output

    # ------------------------------------------------------------------
    # PyTorch Lightning callback hooks
    # ------------------------------------------------------------------

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        unused: Optional[Any] = None,  # type: ignore[override]
    ) -> None:
        """Visualize and log training examples at the end of an epoch."""
        if self.skip_train:
            return

        if not hasattr(trainer, "global_step"):
            raise AttributeError("Trainer is missing 'global_step' attribute.")

        if self.train_dataloader is None:
            self._initialize_dataloaders(pl_module)

        if trainer.current_epoch % self.frequency == 0:
            assert self.train_dataloader is not None  # for type checkers
            self._log_from_dataloader(
                pl_module=pl_module,
                dataloader=self.train_dataloader,
                logger=trainer.logger,
                training_step=trainer.global_step,
                prefix="train",
            )

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        unused: Optional[Any] = None,  # type: ignore[override]
    ) -> None:
        """Visualize and log validation examples at the end of an epoch."""
        if not hasattr(trainer, "global_step"):
            raise AttributeError("Trainer is missing 'global_step' attribute.")

        if self.val_dataloader is None:
            self._initialize_dataloaders(pl_module)

        if trainer.current_epoch % self.frequency == 0:
            assert self.val_dataloader is not None  # for type checkers
            self._log_from_dataloader(
                pl_module=pl_module,
                dataloader=self.val_dataloader,
                logger=trainer.logger,
                training_step=trainer.global_step,
                prefix="val",
            )
