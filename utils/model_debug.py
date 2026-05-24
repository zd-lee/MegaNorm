"""
Enhanced Model Debug System
Enhanced model logger with integrated NaN detection for comprehensive debugging

This module provides:
1. ModelLogger - Singleton logger for model debugging (original functionality)
2. NaNDetector - Comprehensive NaN detection for parameters, gradients, and activations
3. Proper file organization within experiment directory structure
"""

import logging
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import torch
import torch.nn as nn


class ModelLogger:
    """
    Enhanced singleton model logger with support for multiple log files
    Maintains backward compatibility with original ModelLogger
    """
    _instance = None
    _loggers = {}  # Multiple loggers for different purposes
    _base_log_dir = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLogger, cls).__new__(cls)
        return cls._instance

    @classmethod
    def setup(cls, log_dir=None, log_file='model_debug.log', level=logging.WARNING,
              enable_nan_detection=True):
        """
        Setup enhanced logging system with multiple specialized loggers

        Args:
            log_dir: Base log directory (usually experiment_dir/logs)
            log_file: Main debug log file name
            level: Logging level
            enable_nan_detection: Whether to create NaN detection specific loggers
        """
        if cls._loggers:
            return  # Already setup

        cls._base_log_dir = Path(log_dir) if log_dir else Path("logs")
        cls._base_log_dir.mkdir(parents=True, exist_ok=True)

        # Main debug logger
        cls._loggers['main'] = cls._create_logger(
            name='ModelDebug',
            log_file=log_file,
            level=level
        )

        # NaN detection logger
        if enable_nan_detection:
            cls._loggers['nan_detection'] = cls._create_logger(
                name='NaNDetection',
                log_file='nan_detection.log',
                level=logging.WARNING
            )

        # Parameter statistics logger
        if enable_nan_detection:
            cls._loggers['parameter_stats'] = cls._create_logger(
                name='ParameterStats',
                log_file='parameter_stats.log',
                level=logging.INFO
            )

        # Gradient statistics logger
        if enable_nan_detection:
            cls._loggers['gradient_stats'] = cls._create_logger(
                name='GradientStats',
                log_file='gradient_stats.log',
                level=logging.INFO
            )

    @classmethod
    def _create_logger(cls, name: str, log_file: str, level: int):
        """Create a logger with both console and file output"""
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Clear existing handlers
        logger.handlers = []

        # Console handler (ERROR and above only)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler (WARNING and above, or specified level)
        file_handler = logging.FileHandler(cls._base_log_dir / log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        return logger

    @classmethod
    def get_logger(cls, logger_type: str = 'main'):
        """Get specific logger by type"""
        if cls._loggers is None:
            # Fallback to simple logger
            logger = logging.getLogger('ModelDebug')
            logger.setLevel(logging.WARNING)
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
                logger.addHandler(handler)
            return logger

        if logger_type not in cls._loggers:
            return cls._loggers['main']
        return cls._loggers[logger_type]

    @classmethod
    def log_to_file(cls, message: str, logger_type: str = 'main', level: str = 'warning'):
        """Log message to specific logger file"""
        logger = cls.get_logger(logger_type)
        level_map = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }
        logger.log(level_map.get(level, logging.WARNING), message)


class NaNDetector:
    """
    Comprehensive NaN detection system for model parameters, gradients, and activations
    """

    def __init__(self, base_log_dir: Optional[Path] = None, config: Optional[Dict] = None):
        self.base_log_dir = base_log_dir
        self.config = config or {}
        self.nan_counts = {}  # Track NaN occurrences
        self.last_check_time = time.time()

    @classmethod
    def setup(cls, base_log_dir: Union[str, Path], config: Optional[Dict] = None):
        """Setup NaN detector with logging directory and configuration"""
        base_log_dir = Path(base_log_dir)
        base_log_dir.mkdir(parents=True, exist_ok=True)
        return cls(base_log_dir, config)

    def check_tensor_nan(self, tensor: torch.Tensor, name: str,
                        context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Check if tensor contains NaN values and return detailed information

        Args:
            tensor: Tensor to check
            name: Name of the tensor for logging
            context: Additional context information

        Returns:
            Dictionary with NaN detection results
        """
        result = {
            'name': name,
            'has_nan': False,
            'nan_count': 0,
            'total_elements': tensor.numel(),
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'device': str(tensor.device)
        }

        if tensor.numel() == 0:
            return result

        # Check for NaN
        nan_mask = torch.isnan(tensor)
        result['nan_count'] = nan_mask.sum().item()
        result['has_nan'] = result['nan_count'] > 0

        # Get statistics for non-NaN elements
        if result['nan_count'] < tensor.numel():
            valid_tensor = tensor[~nan_mask]
            if valid_tensor.numel() > 0:
                result.update({
                    'min': valid_tensor.min().item(),
                    'max': valid_tensor.max().item(),
                    'mean': valid_tensor.mean().item(),
                    'std': valid_tensor.std().item()
                })

        # Log if NaN detected
        if result['has_nan']:
            self._log_nan_detection(result, context)

            # Update tracking
            if name not in self.nan_counts:
                self.nan_counts[name] = 0
            self.nan_counts[name] += 1

        return result

    def check_model_parameters_nan(self, model: nn.Module,
                                  context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Check all model parameters for NaN values

        Args:
            model: PyTorch model to check
            context: Additional context (epoch, batch, etc.)

        Returns:
            Summary of parameter NaN detection results
        """
        results = {
            'total_parameters': 0,
            'nan_parameters': 0,
            'nan_layers': [],
            'layer_details': {}
        }

        for name, param in model.named_parameters():
            if param.requires_grad:  # Only check trainable parameters
                param_result = self.check_tensor_nan(
                    param.data, f"param.{name}", context
                )

                results['total_parameters'] += param_result['total_elements']
                results['layer_details'][name] = param_result

                if param_result['has_nan']:
                    results['nan_parameters'] += param_result['nan_count']
                    results['nan_layers'].append(name)

        # Log summary
        if results['nan_layers']:
            self._log_parameter_nan_summary(results, context)

        return results

    def check_model_gradients_nan(self, model: nn.Module,
                                 context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Check all model gradients for NaN values

        Args:
            model: PyTorch model to check
            context: Additional context

        Returns:
            Summary of gradient NaN detection results
        """
        results = {
            'total_gradients': 0,
            'nan_gradients': 0,
            'nan_layers': [],
            'layer_details': {}
        }

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_result = self.check_tensor_nan(
                    param.grad.data, f"grad.{name}", context
                )

                results['total_gradients'] += grad_result['total_elements']
                results['layer_details'][name] = grad_result

                if grad_result['has_nan']:
                    results['nan_gradients'] += grad_result['nan_count']
                    results['nan_layers'].append(name)

        # Log summary
        if results['nan_layers']:
            self._log_gradient_nan_summary(results, context)

        return results

    def log_parameter_statistics(self, model: nn.Module, epoch: int, batch: int):
        """Log detailed parameter statistics"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        stats = {
            'timestamp': timestamp,
            'epoch': epoch,
            'batch': batch,
            'parameters': {}
        }

        for name, param in model.named_parameters():
            if param.requires_grad:
                param_data = param.data
                valid_mask = ~torch.isnan(param_data)

                param_stats = {
                    'shape': list(param_data.shape),
                    'total_elements': param_data.numel(),
                    'nan_count': (~valid_mask).sum().item(),
                }

                if valid_mask.any():
                    valid_data = param_data[valid_mask]
                    param_stats.update({
                        'min': valid_data.min().item(),
                        'max': valid_data.max().item(),
                        'mean': valid_data.mean().item(),
                        'std': valid_data.std().item(),
                        'l2_norm': torch.norm(valid_data).item()
                    })

                stats['parameters'][name] = param_stats

        # Log to parameter stats file
        ModelLogger.log_to_file(
            f"Parameter Statistics - Epoch {epoch}, Batch {batch}:\n"
            f"{json.dumps(stats, indent=2)}\n",
            logger_type='parameter_stats',
            level='info'
        )

    def log_gradient_statistics(self, model: nn.Module, epoch: int, batch: int):
        """Log detailed gradient statistics"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        stats = {
            'timestamp': timestamp,
            'epoch': epoch,
            'batch': batch,
            'gradients': {}
        }

        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_data = param.grad.data
                valid_mask = ~torch.isnan(grad_data)

                grad_stats = {
                    'shape': list(grad_data.shape),
                    'total_elements': grad_data.numel(),
                    'nan_count': (~valid_mask).sum().item(),
                }

                if valid_mask.any():
                    valid_data = grad_data[valid_mask]
                    param_norm = torch.norm(valid_data).item()
                    grad_stats.update({
                        'min': valid_data.min().item(),
                        'max': valid_data.max().item(),
                        'mean': valid_data.mean().item(),
                        'std': valid_data.std().item(),
                        'l2_norm': param_norm
                    })
                    total_norm += param_norm ** 2

                stats['gradients'][name] = grad_stats

        stats['total_gradient_norm'] = total_norm ** 0.5

        # Log to gradient stats file
        ModelLogger.log_to_file(
            f"Gradient Statistics - Epoch {epoch}, Batch {batch}:\n"
            f"{json.dumps(stats, indent=2)}\n",
            logger_type='gradient_stats',
            level='info'
        )

    def get_nan_summary(self) -> Dict[str, Any]:
        """Get summary of NaN detections since last reset"""
        current_time = time.time()
        time_elapsed = current_time - self.last_check_time

        return {
            'nan_counts': self.nan_counts.copy(),
            'total_nan_detections': sum(self.nan_counts.values()),
            'time_elapsed_seconds': time_elapsed,
            'last_check_time': self.last_check_time
        }

    def reset_nan_tracking(self):
        """Reset NaN tracking counters"""
        self.nan_counts.clear()
        self.last_check_time = time.time()

    def _log_nan_detection(self, tensor_result: Dict, context: Optional[Dict]):
        """Log detailed NaN detection information"""
        context_str = ""
        if context:
            context_parts = []
            for key, value in context.items():
                context_parts.append(f"{key}={value}")
            context_str = f", Context: {', '.join(context_parts)}"

        message = (
            f"[NaN DETECTED] {tensor_result['name']}\n"
            f"  - Shape: {tensor_result['shape']}\n"
            f"  - NaN Count: {tensor_result['nan_count']:,} / {tensor_result['total_elements']:,} "
            f"({100 * tensor_result['nan_count'] / tensor_result['total_elements']:.4f}%)\n"
        )

        if 'min' in tensor_result:
            message += (
                f"  - Non-NaN Range: [{tensor_result['min']:.6f}, {tensor_result['max']:.6f}]\n"
                f"  - Non-NaN Mean: {tensor_result['mean']:.6f}, Std: {tensor_result['std']:.6f}\n"
            )

        message += f"  - Device: {tensor_result['device']}, Dtype: {tensor_result['dtype']}{context_str}\n"

        # Only log to file if logger is properly set up
        if ModelLogger._loggers and 'nan_detection' in ModelLogger._loggers:
            ModelLogger.log_to_file(message, logger_type='nan_detection', level='warning')
        else:
            # Fallback to console output if logger not set up
            print(f"NaN DETECTION WARNING: {message.strip()}")

    def _log_parameter_nan_summary(self, results: Dict, context: Optional[Dict]):
        """Log parameter NaN detection summary"""
        context_str = f" (Epoch {context.get('epoch', '?')}, Batch {context.get('batch', '?')})" if context else ""

        message = (
            f"[PARAMETER NAN SUMMARY]{context_str}\n"
            f"  - Total Parameters: {results['total_parameters']:,}\n"
            f"  - NaN Parameters: {results['nan_parameters']:,}\n"
            f"  - Affected Layers: {len(results['nan_layers'])}\n"
            f"  - Layers with NaN: {', '.join(results['nan_layers'])}\n"
        )

        # Only log to file if logger is properly set up
        if ModelLogger._loggers and 'nan_detection' in ModelLogger._loggers:
            ModelLogger.log_to_file(message, logger_type='nan_detection', level='error')
        else:
            # Fallback to console output if logger not set up
            print(f"PARAMETER NAN WARNING: {message.strip()}")

    def _log_gradient_nan_summary(self, results: Dict, context: Optional[Dict]):
        """Log gradient NaN detection summary"""
        context_str = f" (Epoch {context.get('epoch', '?')}, Batch {context.get('batch', '?')})" if context else ""

        message = (
            f"[GRADIENT NAN SUMMARY]{context_str}\n"
            f"  - Total Gradients: {results['total_gradients']:,}\n"
            f"  - NaN Gradients: {results['nan_gradients']:,}\n"
            f"  - Affected Layers: {len(results['nan_layers'])}\n"
            f"  - Layers with NaN: {', '.join(results['nan_layers'])}\n"
        )

        # Only log to file if logger is properly set up
        if ModelLogger._loggers and 'nan_detection' in ModelLogger._loggers:
            ModelLogger.log_to_file(message, logger_type='nan_detection', level='error')
        else:
            # Fallback to console output if logger not set up
            print(f"GRADIENT NAN WARNING: {message.strip()}")


# Convenience functions for backward compatibility and easy access
def get_model_logger(logger_type: str = 'main'):
    """Get model logger by type"""
    return ModelLogger.get_logger(logger_type)


def setup_model_debug(log_dir=None, log_file='model_debug.log', level=logging.WARNING,
                     enable_nan_detection=True):
    """Setup enhanced model debugging system"""
    ModelLogger.setup(log_dir, log_file, level, enable_nan_detection)


def create_nan_detector(base_log_dir: Union[str, Path], config: Optional[Dict] = None):
    """Create NaN detector instance"""
    return NaNDetector.setup(base_log_dir, config)


# Global nan detector instance (can be set by training script)
_global_nan_detector = None


def get_global_nan_detector():
    """Get global NaN detector instance"""
    return _global_nan_detector


def set_global_nan_detector(nan_detector):
    """Set global NaN detector instance"""
    global _global_nan_detector
    _global_nan_detector = nan_detector