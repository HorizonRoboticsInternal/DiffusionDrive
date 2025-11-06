from pathlib import Path
from typing import Optional, Iterable, Union

import pytorch_lightning as pl
# from pytorch_lightning.utilities.distributed import rank_zero_info
from lightning_fabric.utilities.rank_zero import rank_zero_info
from navsim.agents.diffusiondrive.ema_model_callback import EMA


class ModelCheckpointAtEpochEnd(pl.callbacks.ModelCheckpoint):
    """Customized callback for saving Lightning checkpoint for every epoch."""

    def __init__(
        self,
        save_top_k: int = 1,
        save_last: bool = True,
        dirpath: Optional[str] = None,
        monitor: Optional[str] = "val/trajectory_loss_epoch",
        mode: Optional[str] = "min",
    ):
        """
        Initialize the callback
        :param save_top_k: Choose how many best checkpoints we want to save:
            save_top_k == 0 means no models are saved.
            save_top_k == -1 means all models are saved.
        :param save_last: Whether to save the last model as last.ckpt.
        :param dirpath: Directory where the checkpoints are saved.
        :param monitor: The metrics to monitor for saving best checkpoints.
        :param mode: How we want to choose the best model: min, max or auto for the metrics we choose.
        """
        if mode is None:
            if monitor.endswith("loss"):
                mode = "min"
            elif monitor.endswith("accuracy") or monitor.endswith("score"):
                mode = "max"
            else:
                raise ValueError("Could not infer mode from monitor name.")
        super().__init__(save_last=save_last, save_top_k=save_top_k, dirpath=dirpath, monitor=monitor, mode=mode)
        self.FILE_EXTENSION = '.ckpt'
        self.verbose = False

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Customized callback function to save checkpoint every epoch.
        :param trainer: Pytorch lightning trainer instance.
        :param pl_module: LightningModule.
        """
        checkpoint_dir = Path(trainer.checkpoint_callback.dirpath).parent / 'checkpoints'
        checkpoint_name = f'epoch={trainer.current_epoch}{self.FILE_EXTENSION}'
        checkpoint_path = checkpoint_dir / checkpoint_name

        ema_callback = self._ema_callback(trainer)
        if ema_callback is not None:
            # with ema_callback.save_original_optimizer_state(trainer):
            #     super()._save_checkpoint(trainer, filepath)

            # save EMA copy of the model as well.
            with ema_callback.save_ema_model(trainer):
                # checkpoint_path = self._ema_format_filepath(checkpoint_path)
                if self.verbose:
                    rank_zero_info(f"Saving EMA weights to separate checkpoint {checkpoint_path}")
                # super()._save_checkpoint(trainer, checkpoint_path)
                trainer.save_checkpoint(str(checkpoint_path))
        else:
            trainer.save_checkpoint(str(checkpoint_path))

    def _ema_format_filepath(self, filepath: str) -> str:
        return filepath.replace(self.FILE_EXTENSION, f'-EMA{self.FILE_EXTENSION}')

    def _has_ema_ckpts(self, checkpoints: Iterable[Path]) -> bool:
        return any(self._is_ema_filepath(checkpoint_path) for checkpoint_path in checkpoints)

    def _is_ema_filepath(self, filepath: Union[Path, str]) -> bool:
        return str(filepath).endswith(f'-EMA{self.FILE_EXTENSION}')

    def _ema_callback(self, trainer: 'pytorch_lightning.Trainer') -> Optional[EMA]:
        ema_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, EMA):
                ema_callback = callback
        return ema_callback


class EvaluationResumeCallback(pl.Callback):
    """Resumes evaluation at the specified epoch number."""

    def __init__(self, epoch_to_resume: int):
        """
        Initialize the callback.
        :param epoch_to_resume: The epoch count of previous evaluation.
        """
        self.epoch_to_resume = epoch_to_resume
        assert self.epoch_to_resume >= 0, f"Invalid epoch number to resume: {self.epoch_to_resume}"
        self._run_eval = True

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called when starting validation.
        :param trainer: The current pytorch_lightning.trainer.Trainer instance.
        :param pl_module: The current pytorch_lightning.core.lightning.LightningModule instance.
        """
        # Inject evaluation epoch to trainer and start evaluation logging
        if self._run_eval:
            if trainer.current_epoch == 0:
                # Restore training states from the checkpoint.
                # trainer.validate() doesn't load the checkpoint when a model is provided.
                trainer.checkpoint_connector.restore_weights()
            trainer.current_epoch = self.epoch_to_resume

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called when finishing validation.
        :param trainer: the current pytorch_lightning.trainer.Trainer instance.
        :param pl_module: the current pytorch_lightning.core.lightning.LightningModule instance.
        """
        # Turn off epoch resuming.
        if self._run_eval:
            self._run_eval = False

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called when starting testing.
        :param trainer: The current pytorch_lightning.trainer.Trainer instance.
        :param pl_module: The current pytorch_lightning.core.lightning.LightningModule instance.
        """
        self.on_validation_start(trainer, pl_module)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called when finishing testing.
        :param trainer: The current pytorch_lightning.trainer.Trainer instance.
        :param pl_module: The current pytorch_lightning.core.lightning.LightningModule instance.
        """
        self.on_validation_end(trainer, pl_module)
