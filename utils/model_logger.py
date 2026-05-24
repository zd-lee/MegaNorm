"""
Model Logger
用于模型内部的 NaN 检测和调试信息记录
"""

import logging
import os
from pathlib import Path


class ModelLogger:
    """
    单例模式的模型日志器
    在训练脚本中初始化一次，模型中的所有组件共享使用
    """
    _instance = None
    _logger = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLogger, cls).__new__(cls)
        return cls._instance

    @classmethod
    def setup(cls, log_dir=None, log_file='model_debug.log', level=logging.WARNING):
        """
        设置日志系统

        Args:
            log_dir: 日志目录
            log_file: 日志文件名
            level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
        """
        if cls._logger is not None:
            return cls._logger

        cls._logger = logging.getLogger('ModelDebug')
        cls._logger.setLevel(level)

        # 清除已有的 handlers
        cls._logger.handlers = []

        # 控制台 handler（只显示 ERROR 及以上）
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        console_formatter = logging.Formatter(
            '[%(levelname)s] %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        cls._logger.addHandler(console_handler)

        # 文件 handler（记录所有 WARNING 及以上）
        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / log_file)
            file_handler.setLevel(logging.WARNING)
            file_formatter = logging.Formatter(
                '%(asctime)s - [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            cls._logger.addHandler(file_handler)

        return cls._logger

    @classmethod
    def get_logger(cls):
        """获取日志器实例"""
        if cls._logger is None:
            # 如果未初始化，创建默认logger（只输出到控制台）
            cls._logger = logging.getLogger('ModelDebug')
            cls._logger.setLevel(logging.WARNING)
            if not cls._logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
                cls._logger.addHandler(handler)
        return cls._logger


# 便捷函数
def get_model_logger():
    """获取模型日志器"""
    return ModelLogger.get_logger()
