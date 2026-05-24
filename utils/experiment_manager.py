"""
Experiment Management System for Point Cloud Segmentation
Provides comprehensive tracking, logging, and output management for training experiments.
"""

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
    """
    Comprehensive experiment management system for point cloud segmentation training.

    Features:
    - Automatic timestamped experiment directories
    - TensorBoard integration for real-time metrics
    - Structured output organization
    - Training history tracking
    - Automatic backup and recovery
    - Comprehensive reporting
    """

    def __init__(self, config: Dict[str, Any], experiment_name: Optional[str] = None):
        """
        Initialize experiment manager.

        Args:
            config: Training configuration dictionary
            experiment_name: Optional custom experiment name
        """
        self.config = config
        self.start_time = datetime.datetime.now()

        # Create experiment directory with timestamp
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            exp_name = f"{experiment_name}_{timestamp}"
        else:
            exp_name = f"{config.get('experiment', {}).get('name', 'ptv3_seg')}_{timestamp}"

        # Create main experiment directory
        # Support custom experiment base path from config
        experiments_base_path = config.get('experiment', {}).get('base_path', 'experiments')
        self.experiment_dir = Path(experiments_base_path) / exp_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.config_dir = self.experiment_dir / "config"
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        self.training_dir = self.experiment_dir / "training"
        self.validation_dir = self.experiment_dir / "validation"
        self.analysis_dir = self.experiment_dir / "analysis"
        self.logs_dir = self.experiment_dir / "logs"

        for dir_path in [self.config_dir, self.checkpoints_dir, self.training_dir,
                        self.validation_dir, self.analysis_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)

        # Save configuration
        self._save_config()

        # Initialize logging
        self._setup_logging()

        # Initialize TensorBoard
        self.writer = None
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=str(self.logs_dir / "tensorboard"))

        # Initialize history tracking
        self.train_history = {
            'epoch': [], 'batch': [], 'loss': [], 'accuracy': [],
            'learning_rate': [], 'time': [], 'precision': [], 'recall': [], 'mean_gt': []
        }
        # Per-epoch training history (aggregated from batch metrics)
        self.train_epoch_history = {
            'epoch': [], 'loss': [], 'accuracy': [], 'learning_rate': [],
            'time': [], 'precision': [], 'recall': [], 'mean_gt': []
        }
        self.val_history = {
            'epoch': [], 'loss': [], 'accuracy': [], 'iou': [], 'f1': [],
            'precision': [], 'recall': [], 'mean_gt': [], 'time': []
        }

        # Best model tracking
        self.best_metrics = {
            'val_loss': float('inf'),
            'val_accuracy': 0.0,
            'val_iou': 0.0,
            'epoch': 0
        }

        # Backup git tracked files
        self._backup_git_code()

        # Experiment initialized silently

    def _save_config(self):
        """Save experiment configuration."""
        config_path = self.config_dir / "config.yaml"

        # Add experiment metadata
        config_with_meta = {
            **self.config,
            'experiment_metadata': {
                'start_time': self.start_time.isoformat(),
                'experiment_dir': str(self.experiment_dir),
                'git_commit': self._get_git_commit(),
            }
        }

        with open(config_path, 'w') as f:
            # Use yaml if available, otherwise json
            try:
                import yaml
                yaml.dump(config_with_meta, f, default_flow_style=False, indent=2)
            except ImportError:
                json.dump(config_with_meta, f, indent=2)

    def _setup_logging(self):
        """Setup structured logging."""
        log_file = self.logs_dir / "training.log"

        # Create logger
        self.logger = logging.getLogger(f"experiment_{self.experiment_dir.name}")
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, cwd=Path.cwd()
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def _backup_git_code(self):
        """Backup all git tracked files to experiment directory."""
        try:
            # Create code backup directory
            code_backup_dir = self.experiment_dir / "code_backup"
            code_backup_dir.mkdir(exist_ok=True)

            # Get git root directory
            try:
                git_root = subprocess.check_output(
                    ['git', 'rev-parse', '--show-toplevel'],
                    cwd=os.getcwd(),
                    text=True
                ).strip()
            except subprocess.CalledProcessError:
                print("Warning: Not in a git repository, skipping code backup")
                return

            # Get git tracked files
            try:
                tracked_files = subprocess.check_output(
                    ['git', 'ls-files'],
                    cwd=git_root,
                    text=True
                ).strip().split('\n')
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to get git files: {e}")
                return

            # Filter out empty lines and large files
            tracked_files = [f for f in tracked_files if f.strip()]

            # Exclude patterns
            exclude_patterns = ['.pyc', '__pycache__', '.git', '.pytest_cache',
                              'node_modules', '.vscode', 'experiments/', '*.log']

            # Copy files with structure preservation
            copied_count = 0
            total_size = 0
            backup_info = []

            for file_path in tracked_files:
                source_path = os.path.join(git_root, file_path)

                # Check exclude patterns
                should_exclude = any(pattern in file_path for pattern in exclude_patterns)
                if should_exclude:
                    continue

                if not os.path.exists(source_path):
                    continue

                try:
                    # Get file size
                    file_size = os.path.getsize(source_path)
                    if file_size > 10 * 1024 * 1024:  # Skip files > 10MB
                        continue

                    # Preserve directory structure
                    dest_path = code_backup_dir / file_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)

                    # Copy file
                    shutil.copy2(source_path, dest_path)
                    copied_count += 1
                    total_size += file_size
                    backup_info.append({
                        'file': file_path,
                        'size': file_size,
                        'modified': datetime.datetime.fromtimestamp(os.path.getmtime(source_path)).isoformat()
                    })

                except Exception as e:
                    print(f"Warning: Failed to backup {file_path}: {e}")

            # Save backup info
            backup_meta = {
                'backup_time': datetime.datetime.now().isoformat(),
                'git_root': git_root,
                'git_commit': self._get_git_commit(),
                'copied_files': copied_count,
                'total_size_mb': total_size / (1024 * 1024),
                'files': backup_info
            }

            with open(code_backup_dir / 'backup_info.json', 'w') as f:
                json.dump(backup_meta, f, indent=2)

            print(f"Code backup completed: {copied_count} files ({total_size / (1024 * 1024):.1f} MB)")

        except Exception as e:
            print(f"Warning: Code backup failed: {e}")

    def log_training_step(self, epoch: int, batch_idx: int, loss: float,
                         accuracy: float, learning_rate: float, step: int,
                         precision: float = 0.0, recall: float = 0.0, mean_gt: float = 0.0):
        """Log training step metrics."""
        current_time = datetime.datetime.now()

        # Update history
        self.train_history['epoch'].append(epoch)
        self.train_history['batch'].append(batch_idx)
        self.train_history['loss'].append(loss)
        self.train_history['accuracy'].append(accuracy)
        self.train_history['learning_rate'].append(learning_rate)
        self.train_history['precision'].append(precision)
        self.train_history['recall'].append(recall)
        self.train_history['mean_gt'].append(mean_gt)
        self.train_history['time'].append(current_time.isoformat())

        # TensorBoard logging
        if self.writer:
            self.writer.add_scalar('Train/Loss', loss, step)
            self.writer.add_scalar('Train/Accuracy', accuracy, step)
            self.writer.add_scalar('Train/LearningRate', learning_rate, step)
            self.writer.add_scalar('Train/Precision', precision, step)
            self.writer.add_scalar('Train/Recall', recall, step)
            self.writer.add_scalar('Train/MeanGT', mean_gt, step)

    def log_training_epoch(self, epoch: int):
        """Log training epoch metrics (aggregated from per-batch data).

        This method automatically calculates epoch-level metrics from the
        batch-level data stored in train_history for the given epoch.
        """
        import numpy as np

        current_time = datetime.datetime.now()

        # Find all batch records for this epoch
        epoch_indices = [i for i, e in enumerate(self.train_history['epoch']) if e == epoch]

        if not epoch_indices:
            self.logger.warning(f"No training data found for epoch {epoch}")
            return

        # Calculate average metrics for this epoch
        avg_loss = np.mean([self.train_history['loss'][i] for i in epoch_indices])
        avg_accuracy = np.mean([self.train_history['accuracy'][i] for i in epoch_indices])
        avg_precision = np.mean([self.train_history['precision'][i] for i in epoch_indices])
        avg_recall = np.mean([self.train_history['recall'][i] for i in epoch_indices])
        avg_mean_gt = np.mean([self.train_history['mean_gt'][i] for i in epoch_indices])
        # Learning rate should be the last one from this epoch (most recent)
        learning_rate = self.train_history['learning_rate'][epoch_indices[-1]]

        # Update epoch history
        self.train_epoch_history['epoch'].append(epoch)
        self.train_epoch_history['loss'].append(avg_loss)
        self.train_epoch_history['accuracy'].append(avg_accuracy)
        self.train_epoch_history['precision'].append(avg_precision)
        self.train_epoch_history['recall'].append(avg_recall)
        self.train_epoch_history['mean_gt'].append(avg_mean_gt)
        self.train_epoch_history['learning_rate'].append(learning_rate)
        self.train_epoch_history['time'].append(current_time.isoformat())

        # TensorBoard logging (per-epoch)
        if self.writer:
            self.writer.add_scalar('Train_Epoch/Loss', avg_loss, epoch)
            self.writer.add_scalar('Train_Epoch/Accuracy', avg_accuracy, epoch)
            self.writer.add_scalar('Train_Epoch/Precision', avg_precision, epoch)
            self.writer.add_scalar('Train_Epoch/Recall', avg_recall, epoch)
            self.writer.add_scalar('Train_Epoch/MeanGT', avg_mean_gt, epoch)
            self.writer.add_scalar('Train_Epoch/LearningRate', learning_rate, epoch)

    def log_validation_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log validation epoch metrics."""
        current_time = datetime.datetime.now()

        # Update history
        self.val_history['epoch'].append(epoch)
        self.val_history['loss'].append(metrics.get('loss', 0.0))
        self.val_history['accuracy'].append(metrics.get('accuracy', 0.0))
        self.val_history['iou'].append(metrics.get('iou', 0.0))
        self.val_history['f1'].append(metrics.get('f1', 0.0))
        self.val_history['precision'].append(metrics.get('precision', 0.0))
        self.val_history['recall'].append(metrics.get('recall', 0.0))
        self.val_history['mean_gt'].append(metrics.get('mean_gt', 0.0))
        self.val_history['time'].append(current_time.isoformat())

        # TensorBoard logging
        if self.writer:
            for metric_name, value in metrics.items():
                self.writer.add_scalar(f'Validation/{metric_name.title()}', value, epoch)

        # Update best metrics
        is_best = False
        if metrics.get('iou', 0.0) > self.best_metrics['val_iou']:
            self.best_metrics.update({
                'val_loss': metrics.get('loss', float('inf')),
                'val_accuracy': metrics.get('accuracy', 0.0),
                'val_iou': metrics.get('iou', 0.0),
                'epoch': epoch
            })
            is_best = True

        # Validation metrics logged silently

        return is_best

    def save_checkpoint(self, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       metrics: Dict[str, float], is_best: bool = False,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
        """Save model checkpoint with metadata."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'train_history': self.train_history,
            'train_epoch_history': self.train_epoch_history,
            'val_history': self.val_history,
            'best_metrics': self.best_metrics,
            'timestamp': datetime.datetime.now().isoformat()
        }

        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # Save latest checkpoint
        latest_path = self.checkpoints_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)

        # Save periodic checkpoint
        if epoch % self.config.get('training', {}).get('save_freq', 10) == 0:
            epoch_path = self.checkpoints_dir / f"checkpoint_epoch_{epoch:03d}.pth"
            torch.save(checkpoint, epoch_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoints_dir / "best_checkpoint.pth"
            torch.save(checkpoint, best_path)
            # Best checkpoint saved silently

    def save_sample_predictions(self, epoch: int, batch_idx: int,
                              point_clouds: torch.Tensor, predictions: torch.Tensor,
                              ground_truth: torch.Tensor, query_points: torch.Tensor,
                              query_labels: torch.Tensor, phase: str = "train",
                              max_samples: int = 2, filenames: list = None):
        """Save sample predictions as point clouds for visualization."""
        from .pointcloud_io import save_segmentation_results

        try:
            # Only log errors, not debug info

            # Create batch directory
            if phase == "train":
                phase_dir = self.training_dir
            elif phase == "validation":
                phase_dir = self.validation_dir
            else:
                phase_dir = self.validation_dir  # Default to validation for other phases
            # Use filename for better organization if available
            if filenames and len(filenames) > 0:
                # Clean filename for directory name
                clean_filename = self._clean_filename(filenames[0])
                batch_dir = phase_dir / f"epoch_{epoch:03d}" / f"batch_{batch_idx:03d}_{clean_filename}"
            else:
                batch_dir = phase_dir / f"epoch_{epoch:03d}" / f"batch_{batch_idx:03d}"

            batch_dir.mkdir(parents=True, exist_ok=True)

            # Handle batch processing - save first sample from concatenated data
            if hasattr(point_clouds, 'shape') and len(point_clouds.shape) >= 2:
                num_points_per_sample = point_clouds.shape[0]
                # Use filename for sample directory if available
                if filenames and len(filenames) > 0:
                    sample_name = self._clean_filename(filenames[0])
                    sample_dir = batch_dir / f"{sample_name}"
                else:
                    sample_dir = batch_dir / f"sample_00"

                sample_dir.mkdir(exist_ok=True)

                # Extract sample data (first N points) - ensure all tensors are moved to CPU
                sample_points = point_clouds[:num_points_per_sample].detach().cpu().numpy()
                sample_preds = predictions[:num_points_per_sample].detach().cpu().numpy()
                sample_gt = ground_truth[:num_points_per_sample].detach().cpu().numpy()

                # Handle query points and labels - ensure all tensors are moved to CPU
                if query_points.dim() == 1 and query_points.shape[0] == 3:
                    sample_query_point = query_points.detach().cpu().numpy()
                elif query_points.dim() == 2:
                    sample_query_point = query_points[0].detach().cpu().numpy()
                else:
                    sample_query_point = query_points.detach().cpu().numpy()

                if query_labels.dim() == 0:
                    sample_query_label = query_labels.item()
                else:
                    sample_query_label = query_labels[0].item() if query_labels.dim() > 0 else query_labels.item()

                # Save predictions as colored PLY
                save_segmentation_results(
                    points=sample_points,
                    predictions=sample_preds,
                    ground_truth=sample_gt,
                    query_point=sample_query_point,
                    query_label=sample_query_label,
                    save_dir=sample_dir,
                    num_classes=2,  # Assuming binary segmentation
                    prefix=f"{sample_name}_" if filenames else ""
                )

                # Log success with meaningful information (to file only, not console)
                if filenames and len(filenames) > 0:
                    self.logger.debug(f"Saved {phase} sample: {filenames[0]} -> {sample_dir}")

        except Exception as e:
            # Only log errors to console
            self.logger.error(f"Failed to save {phase} sample {filenames[0] if filenames else 'unknown'}: {str(e)}")

    def _clean_filename(self, filename: str) -> str:
        """Clean filename for use in directory names."""
        import re
        # Remove file extension and clean for filesystem use
        clean_name = filename.replace('.ply', '').replace('.', '_')
        # Keep only alphanumeric, underscore, and hyphen
        clean_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', clean_name)
        # Limit length
        return clean_name[:50] if len(clean_name) > 50 else clean_name

    def generate_training_plots(self):
        """Generate comprehensive training analysis plots."""
        if not self.val_history['epoch']:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Training Analysis - {self.experiment_dir.name}', fontsize=16)

        epochs = self.val_history['epoch']

        # Loss curves
        axes[0, 0].plot(epochs, self.val_history['loss'], 'b-o', label='Validation Loss', markersize=4)
        if self.train_history['epoch']:
            # Aggregate training loss by epoch
            train_epochs = []
            train_losses = []
            current_epoch = -1
            epoch_losses = []

            for i, epoch in enumerate(self.train_history['epoch']):
                if epoch != current_epoch:
                    if epoch_losses:
                        train_epochs.append(current_epoch)
                        train_losses.append(np.mean(epoch_losses))
                    current_epoch = epoch
                    epoch_losses = [self.train_history['loss'][i]]
                else:
                    epoch_losses.append(self.train_history['loss'][i])

            if epoch_losses:
                train_epochs.append(current_epoch)
                train_losses.append(np.mean(epoch_losses))

            axes[0, 0].plot(train_epochs, train_losses, 'r-s', label='Training Loss', markersize=4)

        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy curves
        axes[0, 1].plot(epochs, self.val_history['accuracy'], 'g-o', label='Accuracy', markersize=4)
        axes[0, 1].plot(epochs, self.val_history['iou'], 'b-s', label='IoU', markersize=4)
        axes[0, 1].plot(epochs, self.val_history['f1'], 'r-^', label='F1', markersize=4)
        axes[0, 1].set_title('Validation Metrics')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Precision & Recall
        axes[0, 2].plot(epochs, self.val_history['precision'], 'c-o', label='Precision', markersize=4)
        axes[0, 2].plot(epochs, self.val_history['recall'], 'm-s', label='Recall', markersize=4)
        axes[0, 2].set_title('Precision & Recall')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        # Best performance markers
        best_iou_idx = np.argmax(self.val_history['iou'])
        best_loss_idx = np.argmin(self.val_history['loss'])

        axes[1, 0].bar(['Best Loss Epoch', 'Best IoU Epoch'],
                      [epochs[best_loss_idx], epochs[best_iou_idx]],
                      color=['lightcoral', 'lightgreen'])
        axes[1, 0].set_title('Best Performance Epochs')
        axes[1, 0].set_ylabel('Epoch')

        # Performance summary
        stats_text = f"""
Best Validation IoU: {max(self.val_history['iou']):.4f} (Epoch {epochs[best_iou_idx]})
Best Validation Loss: {min(self.val_history['loss']):.4f} (Epoch {epochs[best_loss_idx]})
Final IoU: {self.val_history['iou'][-1]:.4f}
Final Loss: {self.val_history['loss'][-1]:.4f}
Training Duration: {datetime.datetime.now() - self.start_time}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        axes[1, 1].set_title('Performance Summary')
        axes[1, 1].axis('off')

        # Learning rate schedule
        if self.train_history['learning_rate']:
            train_steps = list(range(len(self.train_history['learning_rate'])))
            axes[1, 2].plot(train_steps, self.train_history['learning_rate'], 'purple')
            axes[1, 2].set_title('Learning Rate Schedule')
            axes[1, 2].set_xlabel('Training Step')
            axes[1, 2].set_ylabel('Learning Rate')
            axes[1, 2].grid(True)
        else:
            axes[1, 2].axis('off')

        plt.tight_layout()
        plot_path = self.analysis_dir / "training_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Training plots saved silently

    def generate_final_report(self):
        """Generate comprehensive final training report."""
        end_time = datetime.datetime.now()
        duration = end_time - self.start_time

        # Generate plots first
        self.generate_training_plots()

        # Create report data
        report = {
            "experiment_info": {
                "name": self.experiment_dir.name,
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration": str(duration),
                "config": self.config
            },
            "training_summary": {
                "total_epochs": max(self.val_history['epoch']) if self.val_history['epoch'] else 0,
                "total_training_steps": len(self.train_history['loss']),
            },
            "best_performance": self.best_metrics,
            "final_performance": {
                "val_loss": self.val_history['loss'][-1] if self.val_history['loss'] else float('inf'),
                "val_accuracy": self.val_history['accuracy'][-1] if self.val_history['accuracy'] else 0.0,
                "val_iou": self.val_history['iou'][-1] if self.val_history['iou'] else 0.0,
                "val_f1": self.val_history['f1'][-1] if self.val_history['f1'] else 0.0
            },
            "training_history": {
                "train_history": self.train_history,
                "val_history": self.val_history
            }
        }

        # Save JSON report
        json_path = self.analysis_dir / "training_report.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Generate Markdown report
        self._generate_markdown_report(report)

        # Final report generated silently
        return report

    def _generate_markdown_report(self, report: Dict[str, Any]):
        """Generate Markdown format training report."""
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
        with open(md_path, 'w') as f:
            f.write(md_content)

    def _dict_to_yaml_str(self, d: dict, indent: int = 0) -> str:
        """Convert dictionary to YAML-like string."""
        yaml_str = ""
        for key, value in d.items():
            yaml_str += "  " * indent + f"{key}:"
            if isinstance(value, dict):
                yaml_str += "\n" + self._dict_to_yaml_str(value, indent + 1)
            else:
                yaml_str += f" {value}\n"
        return yaml_str

    def cleanup(self):
        """Clean up resources."""
        if self.writer:
            self.writer.close()

        # Generate final report
        self.generate_final_report()

        # Rename experiment directory to add "_finish" suffix
        try:
            finished_dir = self.experiment_dir.parent / (self.experiment_dir.name + "_finish")
            if not finished_dir.exists():
                self.experiment_dir.rename(finished_dir)
                self.experiment_dir = finished_dir
        except Exception as e:
            print(f"Warning: Failed to rename experiment directory: {e}")

        # Experiment completed silently

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()