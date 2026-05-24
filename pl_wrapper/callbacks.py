import time
from pathlib import Path
from lightning.pytorch.callbacks import Callback


class TimeBasedCheckpoint(Callback):
    def __init__(self, save_interval_hours: float = 1.0, keep_last_n: int = 5, save_dir: str = None):
        super().__init__()
        self.save_interval_seconds = save_interval_hours * 3600
        self.save_dir = Path(save_dir) if save_dir else None
        self.keep_last_n = keep_last_n
        self.last_save_time = None
        self.start_time = None
        self.saved_checkpoints = []

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        self.last_save_time = self.start_time

        if hasattr(trainer, 'logger') and hasattr(trainer.logger, 'log_dir'):
            self.save_dir = Path(trainer.logger.log_dir) / 'checkpoints_hourly'

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _save_and_cleanup(self, trainer, filepath, epoch, global_step, hours, minutes, is_val=False):
        trainer.save_checkpoint(filepath)

        self.saved_checkpoints.append(filepath)

        if len(self.saved_checkpoints) > self.keep_last_n:
            old_checkpoint = self.saved_checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
                print(f"Deleted old checkpoint: {old_checkpoint.name}")

        val_tag = " (during validation)" if is_val else ""
        print(f"\n{'='*80}")
        print(f"Time-based checkpoint saved{val_tag}: {filepath}")
        print(f"Epoch: {epoch}, Global Step: {global_step}")
        print(f"Training time: {hours}h {minutes}m")
        print(f"Keeping last {self.keep_last_n} checkpoints ({len(self.saved_checkpoints)} currently saved)")
        print(f"{'='*80}\n")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        current_time = time.time()

        if self.last_save_time is None:
            self.last_save_time = current_time
            return

        elapsed = current_time - self.last_save_time

        if elapsed >= self.save_interval_seconds:
            total_elapsed = current_time - self.start_time
            hours = int(total_elapsed // 3600)
            minutes = int((total_elapsed % 3600) // 60)

            epoch = trainer.current_epoch
            global_step = trainer.global_step

            filename = f"time_checkpoint_epoch{epoch:03d}_step{global_step}_time{hours}h{minutes}m.ckpt"
            filepath = self.save_dir / filename

            self._save_and_cleanup(trainer, filepath, epoch, global_step, hours, minutes, is_val=False)

            self.last_save_time = current_time

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        current_time = time.time()

        if self.last_save_time is None:
            self.last_save_time = current_time
            return

        elapsed = current_time - self.last_save_time

        if elapsed >= self.save_interval_seconds:
            total_elapsed = current_time - self.start_time
            hours = int(total_elapsed // 3600)
            minutes = int((total_elapsed % 3600) // 60)

            epoch = trainer.current_epoch
            global_step = trainer.global_step

            filename = f"time_checkpoint_epoch{epoch:03d}_step{global_step}_time{hours}h{minutes}m_val.ckpt"
            filepath = self.save_dir / filename

            self._save_and_cleanup(trainer, filepath, epoch, global_step, hours, minutes, is_val=True)

            self.last_save_time = current_time
