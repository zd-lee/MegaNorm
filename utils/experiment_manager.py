import json
import shutil

import numpy as np

from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import datetime
import logging

import matplotlib.pyplot as plt
import torch

import subprocess
import os

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")


class ExperimentManager:
    def __init__(self, config: Dict[str, Any], experiment_name: Optional[str] = None):
        self.config = config
        self.start_time = datetime.datetime.now()
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            exp_name = f"{experiment_name}_{timestamp}"
        else:
            exp_name = (
                f"{config.get('experiment', {}).get('name', 'ptv3_seg')}_{timestamp}"
            )
        experiments_base_path = config.get("experiment", {}).get(
            "base_path", "experiments"
        )
        self.experiment_dir = Path(experiments_base_path) / exp_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir = self.experiment_dir / "config"
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        self.training_dir = self.experiment_dir / "training"
        self.validation_dir = self.experiment_dir / "validation"
        self.analysis_dir = self.experiment_dir / "analysis"
        self.logs_dir = self.experiment_dir / "logs"
        for dir_path in [
            self.config_dir,
            self.checkpoints_dir,
            self.training_dir,
            self.validation_dir,
            self.analysis_dir,
            self.logs_dir,
        ]:
            dir_path.mkdir(exist_ok=True)
        self._save_config()
        self._setup_logging()
        self.writer = None
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=str(self.logs_dir / "tensorboard"))
        self.train_history = {
            "epoch": [],
            "batch": [],
            "loss": [],
            "accuracy": [],
            "learning_rate": [],
            "time": [],
            "precision": [],
            "recall": [],
            "mean_gt": [],
        }
        self.train_epoch_history = {
            "epoch": [],
            "loss": [],
            "accuracy": [],
            "learning_rate": [],
            "time": [],
            "precision": [],
            "recall": [],
            "mean_gt": [],
        }
        self.val_history = {
            "epoch": [],
            "loss": [],
            "accuracy": [],
            "iou": [],
            "f1": [],
            "precision": [],
            "recall": [],
            "mean_gt": [],
            "time": [],
        }
        self.best_metrics = {
            "val_loss": float("inf"),
            "val_accuracy": 0.0,
            "val_iou": 0.0,
            "epoch": 0,
        }
        self._backup_git_code()

    def _save_config(self):
        config_path = self.config_dir / "config.yaml"
        config_with_meta = {
            **self.config,
            "experiment_metadata": {
                "start_time": self.start_time.isoformat(),
                "experiment_dir": str(self.experiment_dir),
                "git_commit": self._get_git_commit(),
            },
        }
        with open(config_path, "w") as f:
            try:
                import yaml

                yaml.dump(config_with_meta, f, default_flow_style=False, indent=2)
            except ImportError:
                json.dump(config_with_meta, f, indent=2)

    def _setup_logging(self):
        log_file = self.logs_dir / "training.log"
        self.logger = logging.getLogger(f"experiment_{self.experiment_dir.name}")
        self.logger.setLevel(logging.INFO)
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _get_git_commit(self) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def _backup_git_code(self):
        try:
            code_backup_dir = self.experiment_dir / "code_backup"
            code_backup_dir.mkdir(exist_ok=True)
            try:
                git_root = subprocess.check_output(
                    ["git", "rev-parse", "--show-toplevel"], cwd=os.getcwd(), text=True
                ).strip()
            except subprocess.CalledProcessError:
                print("Warning: Not in a git repository, skipping code backup")
                return
            try:
                tracked_files = (
                    subprocess.check_output(
                        ["git", "ls-files"], cwd=git_root, text=True
                    )
                    .strip()
                    .split("\n")
                )
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to get git files: {e}")
                return
            tracked_files = [f for f in tracked_files if f.strip()]
            exclude_patterns = [
                ".pyc",
                "__pycache__",
                ".git",
                ".pytest_cache",
                "node_modules",
                ".vscode",
                "experiments/",
                "*.log",
            ]
            copied_count = 0
            total_size = 0
            backup_info = []
            for file_path in tracked_files:
                source_path = os.path.join(git_root, file_path)
                should_exclude = any(
                    pattern in file_path for pattern in exclude_patterns
                )
                if should_exclude:
                    continue
                if not os.path.exists(source_path):
                    continue
                try:
                    file_size = os.path.getsize(source_path)
                    if file_size > 10 * 1024 * 1024:
                        continue
                    dest_path = code_backup_dir / file_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, dest_path)
                    copied_count += 1
                    total_size += file_size
                    backup_info.append(
                        {
                            "file": file_path,
                            "size": file_size,
                            "modified": datetime.datetime.fromtimestamp(
                                os.path.getmtime(source_path)
                            ).isoformat(),
                        }
                    )
                except Exception as e:
                    print(f"Warning: Failed to backup {file_path}: {e}")
            backup_meta = {
                "backup_time": datetime.datetime.now().isoformat(),
                "git_root": git_root,
                "git_commit": self._get_git_commit(),
                "copied_files": copied_count,
                "total_size_mb": total_size / (1024 * 1024),
                "files": backup_info,
            }
            with open(code_backup_dir / "backup_info.json", "w") as f:
                json.dump(backup_meta, f, indent=2)
            print(
                f"Code backup completed: {copied_count} files ({total_size / (1024 * 1024):.1f} MB)"
            )
        except Exception as e:
            print(f"Warning: Code backup failed: {e}")

    def log_training_step(
        self,
        epoch: int,
        batch_idx: int,
        loss: float,
        accuracy: float,
        learning_rate: float,
        step: int,
        precision: float = 0.0,
        recall: float = 0.0,
        mean_gt: float = 0.0,
    ):
        current_time = datetime.datetime.now()
        self.train_history["epoch"].append(epoch)
        self.train_history["batch"].append(batch_idx)
        self.train_history["loss"].append(loss)
        self.train_history["accuracy"].append(accuracy)
        self.train_history["learning_rate"].append(learning_rate)
        self.train_history["precision"].append(precision)
        self.train_history["recall"].append(recall)
        self.train_history["mean_gt"].append(mean_gt)
        self.train_history["time"].append(current_time.isoformat())
        if self.writer:
            self.writer.add_scalar("Train/Loss", loss, step)
            self.writer.add_scalar("Train/Accuracy", accuracy, step)
            self.writer.add_scalar("Train/LearningRate", learning_rate, step)
            self.writer.add_scalar("Train/Precision", precision, step)
            self.writer.add_scalar("Train/Recall", recall, step)
            self.writer.add_scalar("Train/MeanGT", mean_gt, step)

    def log_training_epoch(self, epoch: int):
        import numpy as np

        current_time = datetime.datetime.now()
        epoch_indices = [
            i for i, e in enumerate(self.train_history["epoch"]) if e == epoch
        ]
        if not epoch_indices:
            self.logger.warning(f"No training data found for epoch {epoch}")
            return
        avg_loss = np.mean([self.train_history["loss"][i] for i in epoch_indices])
        avg_accuracy = np.mean(
            [self.train_history["accuracy"][i] for i in epoch_indices]
        )
        avg_precision = np.mean(
            [self.train_history["precision"][i] for i in epoch_indices]
        )
        avg_recall = np.mean([self.train_history["recall"][i] for i in epoch_indices])
        avg_mean_gt = np.mean([self.train_history["mean_gt"][i] for i in epoch_indices])
        learning_rate = self.train_history["learning_rate"][epoch_indices[-1]]
        self.train_epoch_history["epoch"].append(epoch)
        self.train_epoch_history["loss"].append(avg_loss)
        self.train_epoch_history["accuracy"].append(avg_accuracy)
        self.train_epoch_history["precision"].append(avg_precision)
        self.train_epoch_history["recall"].append(avg_recall)
        self.train_epoch_history["mean_gt"].append(avg_mean_gt)
        self.train_epoch_history["learning_rate"].append(learning_rate)
        self.train_epoch_history["time"].append(current_time.isoformat())
        if self.writer:
            self.writer.add_scalar("Train_Epoch/Loss", avg_loss, epoch)
            self.writer.add_scalar("Train_Epoch/Accuracy", avg_accuracy, epoch)
            self.writer.add_scalar("Train_Epoch/Precision", avg_precision, epoch)
            self.writer.add_scalar("Train_Epoch/Recall", avg_recall, epoch)
            self.writer.add_scalar("Train_Epoch/MeanGT", avg_mean_gt, epoch)
            self.writer.add_scalar("Train_Epoch/LearningRate", learning_rate, epoch)

    def log_validation_epoch(self, epoch: int, metrics: Dict[str, float]):
        current_time = datetime.datetime.now()
        self.val_history["epoch"].append(epoch)
        self.val_history["loss"].append(metrics.get("loss", 0.0))
        self.val_history["accuracy"].append(metrics.get("accuracy", 0.0))
        self.val_history["iou"].append(metrics.get("iou", 0.0))
        self.val_history["f1"].append(metrics.get("f1", 0.0))
        self.val_history["precision"].append(metrics.get("precision", 0.0))
        self.val_history["recall"].append(metrics.get("recall", 0.0))
        self.val_history["mean_gt"].append(metrics.get("mean_gt", 0.0))
        self.val_history["time"].append(current_time.isoformat())
        if self.writer:
            for metric_name, value in metrics.items():
                self.writer.add_scalar(
                    f"Validation/{metric_name.title()}", value, epoch
                )
        is_best = False
        if metrics.get("iou", 0.0) > self.best_metrics["val_iou"]:
            self.best_metrics.update(
                {
                    "val_loss": metrics.get("loss", float("inf")),
                    "val_accuracy": metrics.get("accuracy", 0.0),
                    "val_iou": metrics.get("iou", 0.0),
                    "epoch": epoch,
                }
            )
            is_best = True
        return is_best

    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, float],
        is_best: bool = False,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "train_history": self.train_history,
            "train_epoch_history": self.train_epoch_history,
            "val_history": self.val_history,
            "best_metrics": self.best_metrics,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        if scheduler:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        latest_path = self.checkpoints_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)
        if epoch % self.config.get("training", {}).get("save_freq", 10) == 0:
            epoch_path = self.checkpoints_dir / f"checkpoint_epoch_{epoch:03d}.pth"
            torch.save(checkpoint, epoch_path)
        if is_best:
            best_path = self.checkpoints_dir / "best_checkpoint.pth"
            torch.save(checkpoint, best_path)

    def save_sample_predictions(
        self,
        epoch: int,
        batch_idx: int,
        point_clouds: torch.Tensor,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
        query_points: torch.Tensor,
        query_labels: torch.Tensor,
        phase: str = "train",
        max_samples: int = 2,
        filenames: list = None,
    ):
        from .pointcloud_io import save_segmentation_results

        try:
            if phase == "train":
                phase_dir = self.training_dir
            elif phase == "validation":
                phase_dir = self.validation_dir
            else:
                phase_dir = self.validation_dir
            if filenames and len(filenames) > 0:
                clean_filename = self._clean_filename(filenames[0])
                batch_dir = (
                    phase_dir
                    / f"epoch_{epoch:03d}"
                    / f"batch_{batch_idx:03d}_{clean_filename}"
                )
            else:
                batch_dir = phase_dir / f"epoch_{epoch:03d}" / f"batch_{batch_idx:03d}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            if hasattr(point_clouds, "shape") and len(point_clouds.shape) >= 2:
                num_points_per_sample = point_clouds.shape[0]
                if filenames and len(filenames) > 0:
                    sample_name = self._clean_filename(filenames[0])
                    sample_dir = batch_dir / f"{sample_name}"
                else:
                    sample_dir = batch_dir / f"sample_00"
                sample_dir.mkdir(exist_ok=True)
                sample_points = (
                    point_clouds[:num_points_per_sample].detach().cpu().numpy()
                )
                sample_preds = (
                    predictions[:num_points_per_sample].detach().cpu().numpy()
                )
                sample_gt = ground_truth[:num_points_per_sample].detach().cpu().numpy()
                if query_points.dim() == 1 and query_points.shape[0] == 3:
                    sample_query_point = query_points.detach().cpu().numpy()
                elif query_points.dim() == 2:
                    sample_query_point = query_points[0].detach().cpu().numpy()
                else:
                    sample_query_point = query_points.detach().cpu().numpy()
                if query_labels.dim() == 0:
                    sample_query_label = query_labels.item()
                else:
                    sample_query_label = (
                        query_labels[0].item()
                        if query_labels.dim() > 0
                        else query_labels.item()
                    )
                save_segmentation_results(
                    points=sample_points,
                    predictions=sample_preds,
                    ground_truth=sample_gt,
                    query_point=sample_query_point,
                    query_label=sample_query_label,
                    save_dir=sample_dir,
                    num_classes=2,
                    prefix=f"{sample_name}_" if filenames else "",
                )
                if filenames and len(filenames) > 0:
                    self.logger.debug(
                        f"Saved {phase} sample: {filenames[0]} -> {sample_dir}"
                    )
        except Exception as e:
            self.logger.error(
                f"Failed to save {phase} sample {filenames[0] if filenames else 'unknown'}: {str(e)}"
            )

    def _clean_filename(self, filename: str) -> str:
        import re

        clean_name = filename.replace(".ply", "").replace(".", "_")
        clean_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", clean_name)
        return clean_name[:50] if len(clean_name) > 50 else clean_name

    def generate_training_plots(self):
        if not self.val_history["epoch"]:
            return
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Training Analysis - {self.experiment_dir.name}", fontsize=16)
        epochs = self.val_history["epoch"]
        axes[0, 0].plot(
            epochs,
            self.val_history["loss"],
            "b-o",
            label="Validation Loss",
            markersize=4,
        )
        if self.train_history["epoch"]:
            train_epochs = []
            train_losses = []
            current_epoch = -1
            epoch_losses = []
            for i, epoch in enumerate(self.train_history["epoch"]):
                if epoch != current_epoch:
                    if epoch_losses:
                        train_epochs.append(current_epoch)
                        train_losses.append(np.mean(epoch_losses))
                    current_epoch = epoch
                    epoch_losses = [self.train_history["loss"][i]]
                else:
                    epoch_losses.append(self.train_history["loss"][i])
            if epoch_losses:
                train_epochs.append(current_epoch)
                train_losses.append(np.mean(epoch_losses))
            axes[0, 0].plot(
                train_epochs, train_losses, "r-s", label="Training Loss", markersize=4
            )
        axes[0, 0].set_title("Loss Curves")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 1].plot(
            epochs, self.val_history["accuracy"], "g-o", label="Accuracy", markersize=4
        )
        axes[0, 1].plot(
            epochs, self.val_history["iou"], "b-s", label="IoU", markersize=4
        )
        axes[0, 1].plot(epochs, self.val_history["f1"], "r-^", label="F1", markersize=4)
        axes[0, 1].set_title("Validation Metrics")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Score")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 2].plot(
            epochs,
            self.val_history["precision"],
            "c-o",
            label="Precision",
            markersize=4,
        )
        axes[0, 2].plot(
            epochs, self.val_history["recall"], "m-s", label="Recall", markersize=4
        )
        axes[0, 2].set_title("Precision & Recall")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("Score")
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        best_iou_idx = np.argmax(self.val_history["iou"])
        best_loss_idx = np.argmin(self.val_history["loss"])
        axes[1, 0].bar(
            ["Best Loss Epoch", "Best IoU Epoch"],
            [epochs[best_loss_idx], epochs[best_iou_idx]],
            color=["lightcoral", "lightgreen"],
        )
        axes[1, 0].set_title("Best Performance Epochs")
        axes[1, 0].set_ylabel("Epoch")
        stats_text = f"""
Best Validation IoU: {max(self.val_history['iou']):.4f} (Epoch {epochs[best_iou_idx]})
Best Validation Loss: {min(self.val_history['loss']):.4f} (Epoch {epochs[best_loss_idx]})
Final IoU: {self.val_history['iou'][-1]:.4f}
Final Loss: {self.val_history['loss'][-1]:.4f}
Training Duration: {datetime.datetime.now() - self.start_time}
        """
        axes[1, 1].text(
            0.1,
            0.5,
            stats_text,
            transform=axes[1, 1].transAxes,
            fontsize=10,
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        axes[1, 1].set_title("Performance Summary")
        axes[1, 1].axis("off")
        if self.train_history["learning_rate"]:
            train_steps = list(range(len(self.train_history["learning_rate"])))
            axes[1, 2].plot(train_steps, self.train_history["learning_rate"], "purple")
            axes[1, 2].set_title("Learning Rate Schedule")
            axes[1, 2].set_xlabel("Training Step")
            axes[1, 2].set_ylabel("Learning Rate")
            axes[1, 2].grid(True)
        else:
            axes[1, 2].axis("off")
        plt.tight_layout()
        plot_path = self.analysis_dir / "training_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    def generate_final_report(self):
        end_time = datetime.datetime.now()
        duration = end_time - self.start_time
        self.generate_training_plots()
        report = {
            "experiment_info": {
                "name": self.experiment_dir.name,
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration": str(duration),
                "config": self.config,
            },
            "training_summary": {
                "total_epochs": (
                    max(self.val_history["epoch"]) if self.val_history["epoch"] else 0
                ),
                "total_training_steps": len(self.train_history["loss"]),
            },
            "best_performance": self.best_metrics,
            "final_performance": {
                "val_loss": (
                    self.val_history["loss"][-1]
                    if self.val_history["loss"]
                    else float("inf")
                ),
                "val_accuracy": (
                    self.val_history["accuracy"][-1]
                    if self.val_history["accuracy"]
                    else 0.0
                ),
                "val_iou": (
                    self.val_history["iou"][-1] if self.val_history["iou"] else 0.0
                ),
                "val_f1": self.val_history["f1"][-1] if self.val_history["f1"] else 0.0,
            },
            "training_history": {
                "train_history": self.train_history,
                "val_history": self.val_history,
            },
        }
        json_path = self.analysis_dir / "training_report.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        self._generate_markdown_report(report)
        return report

    def _generate_markdown_report(self, report: Dict[str, Any]):
        md_content = f"""# Training Report: {report['experiment_info']['name']}


## Experiment Overview
- **Start Time**: {report['experiment_info']['start_time']}
- **End Time**: {report['experiment_info']['end_time']}
- **Duration**: {report['experiment_info']['duration']}
- **Total Epochs**: {report['training_summary']['total_epochs']}
- **Total Training Steps**: {report['training_summary']['total_training_steps']}
## Best Performance
- **Best Validation IoU**: {report['best_performance']['val_iou']:.4f} (Epoch {report['best_performance']['epoch']})
- **Best Validation Accuracy**: {report['best_performance']['val_accuracy']:.4f}
- **Best Validation Loss**: {report['best_performance']['val_loss']:.4f}
## Final Performance
- **Final Validation IoU**: {report['final_performance']['val_iou']:.4f}
- **Final Validation Accuracy**: {report['final_performance']['val_accuracy']:.4f}
- **Final Validation Loss**: {report['final_performance']['val_loss']:.4f}
- **Final Validation F1**: {report['final_performance']['val_f1']:.4f}
## Training Configuration
```yaml
{self._dict_to_yaml_str(report['experiment_info']['config'])}
```
## Training Curves
![Training Analysis](training_analysis.png)
## Files Generated
- 🔧 **Configuration**: `config/config.yaml`
- 💾 **Checkpoints**: `checkpoints/`
  - `best_checkpoint.pth` - Best model based on validation IoU
  - `latest_checkpoint.pth` - Most recent model state
- 📊 **Analysis**: `analysis/`
  - `training_analysis.png` - Training curves and metrics
  - `training_report.json` - Detailed metrics in JSON format
- 🎯 **Samples**: `training/` and `validation/`
  - Sample predictions saved as colored PLY files
- 📝 **Logs**: `logs/`
  - `training.log` - Detailed training logs
  - `tensorboard/` - TensorBoard events


---
*Report generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        md_path = self.analysis_dir / "training_report.md"
        with open(md_path, "w") as f:
            f.write(md_content)

    def _dict_to_yaml_str(self, d: dict, indent: int = 0) -> str:
        yaml_str = ""
        for key, value in d.items():
            yaml_str += "  " * indent + f"{key}:"
            if isinstance(value, dict):
                yaml_str += "\n" + self._dict_to_yaml_str(value, indent + 1)
            else:
                yaml_str += f" {value}\n"
        return yaml_str

    def cleanup(self):
        if self.writer:
            self.writer.close()
        self.generate_final_report()
        try:
            finished_dir = self.experiment_dir.parent / (
                self.experiment_dir.name + "_finish"
            )
            if not finished_dir.exists():
                self.experiment_dir.rename(finished_dir)
                self.experiment_dir = finished_dir
        except Exception as e:
            print(f"Warning: Failed to rename experiment directory: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
