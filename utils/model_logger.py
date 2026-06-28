import logging
import os
from pathlib import Path


class ModelLogger:
    _instance = None
    _logger = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLogger, cls).__new__(cls)
        return cls._instance

    @classmethod
    def setup(cls, log_dir=None, log_file="model_debug.log", level=logging.WARNING):
        if cls._logger is not None:
            return cls._logger
        cls._logger = logging.getLogger("ModelDebug")
        cls._logger.setLevel(level)
        cls._logger.handlers = []
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
        console_handler.setFormatter(console_formatter)
        cls._logger.addHandler(console_handler)
        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / log_file)
            file_handler.setLevel(logging.WARNING)
            file_formatter = logging.Formatter(
                "%(asctime)s - [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)
            cls._logger.addHandler(file_handler)
        return cls._logger

    @classmethod
    def get_logger(cls):
        if cls._logger is None:
            cls._logger = logging.getLogger("ModelDebug")
            cls._logger.setLevel(logging.WARNING)
            if not cls._logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
                cls._logger.addHandler(handler)
        return cls._logger


def get_model_logger():
    return ModelLogger.get_logger()
