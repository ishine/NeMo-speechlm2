# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SaveNemoCallback: Automatic .nemo format saving during training.

This callback enables automatic conversion of PyTorch Lightning checkpoints (.ckpt)
to NeMo format (.nemo) during training when the `always_save_nemo` config flag is enabled.

Usage:
    In your training config YAML:
    ```yaml
    trainer:
      callbacks:
        - _target_: nemo.collections.speechlm2.parts.save_nemo_callback.SaveNemoCallback
          always_save_nemo: true  # Enable automatic .nemo saving
          save_top_k: -1          # Optional: save all checkpoints as .nemo (default: -1)
    ```

    Or programmatically:
    ```python
    from nemo.collections.speechlm2.parts.save_nemo_callback import SaveNemoCallback

    callback = SaveNemoCallback(always_save_nemo=True)
    trainer = Trainer(callbacks=[callback])
    ```
"""

from pathlib import Path
from typing import Optional

from lightning import Callback, LightningModule, Trainer
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero


class SaveNemoCallback(Callback):
    """
    PyTorch Lightning callback for automatic .nemo format saving during training.

    This callback monitors checkpoint save events and automatically converts
    saved .ckpt files to .nemo format when `always_save_nemo` is enabled.

    Features:
    - Automatic .nemo conversion after each checkpoint save
    - Integration with Solution 1 (Enhanced Checkpoint Saving)
    - Support for save_top_k to control which checkpoints get .nemo format
    - Distributed training support (only rank 0 saves)
    - Preserves complete perception configuration from Solution 1

    Args:
        always_save_nemo: Whether to save .nemo format for all checkpoints (default: False)
        save_top_k: Number of best checkpoints to save as .nemo (-1 for all, default: -1)

    Example:
        >>> from nemo.collections.speechlm2.parts.save_nemo_callback import SaveNemoCallback
        >>> callback = SaveNemoCallback(always_save_nemo=True, save_top_k=3)
        >>> trainer = Trainer(callbacks=[callback])
        >>> trainer.fit(model)
        # Automatically saves .nemo files alongside .ckpt files
    """

    def __init__(
        self,
        always_save_nemo: bool = False,
        save_top_k: int = -1,
    ):
        super().__init__()
        self.always_save_nemo = always_save_nemo
        self.save_top_k = save_top_k
        self._saved_checkpoints = []

    def on_save_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: dict,
    ) -> None:
        """
        Called when the trainer saves a checkpoint.

        This hook is called after the checkpoint is saved but we need to track
        which checkpoint was saved to convert it in on_train_batch_end or on_epoch_end.
        """
        # Just mark that a checkpoint was saved
        # Actual conversion happens in on_train_batch_end
        self._checkpoint_saved = True

    def _convert_checkpoint_to_nemo(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """
        Convert the most recently saved .ckpt checkpoint to .nemo format.

        This method:
        1. Finds the most recent checkpoint saved by the trainer
        2. Loads the model from that checkpoint (to get complete config from Solution 1)
        3. Saves the model in .nemo format using save_to()
        4. Cleans up old .nemo files if save_top_k is set

        Args:
            trainer: PyTorch Lightning Trainer instance
            pl_module: The model being trained
        """
        if not self.always_save_nemo:
            return

        # Only save on rank 0
        if not is_global_rank_zero():
            return

        try:
            # Get the checkpoint callback
            checkpoint_callback = trainer.checkpoint_callback
            if checkpoint_callback is None:
                logging.warning("No checkpoint callback found, cannot convert to .nemo format")
                return

            # Get the most recent checkpoint path
            if hasattr(checkpoint_callback, 'last_model_path') and checkpoint_callback.last_model_path:
                ckpt_path = Path(checkpoint_callback.last_model_path)
            elif hasattr(checkpoint_callback, 'best_model_path') and checkpoint_callback.best_model_path:
                ckpt_path = Path(checkpoint_callback.best_model_path)
            else:
                logging.debug("No checkpoint path available for .nemo conversion")
                return

            # Skip if checkpoint doesn't exist
            if not ckpt_path.exists():
                logging.warning(f"Checkpoint file not found: {ckpt_path}")
                return

            # Create .nemo path by replacing .ckpt extension
            nemo_path = ckpt_path.with_suffix('.nemo')

            # Skip if .nemo file already exists (avoid re-saving)
            if nemo_path.exists():
                logging.debug(f".nemo file already exists, skipping: {nemo_path}")
                return

            logging.info(f"Converting checkpoint to .nemo format: {ckpt_path.name} → {nemo_path.name}")

            # The model in memory already has complete perception config from Solution 1
            # because on_save_checkpoint hook was called before this
            # So we can directly call save_to()
            pl_module.save_to(str(nemo_path))

            # Track saved .nemo files for cleanup
            self._saved_checkpoints.append(nemo_path)

            # Clean up old .nemo files if save_top_k is set
            if self.save_top_k > 0 and len(self._saved_checkpoints) > self.save_top_k:
                # Remove oldest .nemo files
                to_remove = self._saved_checkpoints[: -self.save_top_k]
                for old_nemo in to_remove:
                    if old_nemo.exists():
                        old_nemo.unlink()
                        logging.info(f"Removed old .nemo file: {old_nemo.name}")
                # Keep only recent checkpoints in tracking
                self._saved_checkpoints = self._saved_checkpoints[-self.save_top_k :]

            logging.info(f"✓ Successfully saved .nemo format: {nemo_path}")

        except Exception as e:
            logging.error(f"Failed to convert checkpoint to .nemo format: {e}", exc_info=True)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        """
        Called after training batch ends.

        This is a good place to convert checkpoints saved during the batch.
        """
        if hasattr(self, '_checkpoint_saved') and self._checkpoint_saved:
            self._convert_checkpoint_to_nemo(trainer, pl_module)
            self._checkpoint_saved = False

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """
        Called at the end of each training epoch.

        This ensures any checkpoints saved at epoch end are converted to .nemo format.
        """
        self._convert_checkpoint_to_nemo(trainer, pl_module)

    def on_train_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """
        Called at the end of training.

        This ensures the final checkpoint is converted to .nemo format.
        """
        self._convert_checkpoint_to_nemo(trainer, pl_module)

        if self.always_save_nemo and is_global_rank_zero():
            logging.info("✓ Completed automatic .nemo format saving")
            if self._saved_checkpoints:
                logging.info(f"  Total .nemo files saved: {len(self._saved_checkpoints)}")
