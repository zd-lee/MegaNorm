import os
import sys
import tempfile
from pathlib import Path

import torch
import yaml
from lightning.pytorch.cli import LightningCLI

torch.set_float32_matmul_precision("medium")
os.makedirs("pl_logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)


class CustomLightningCLI(LightningCLI):
    def _deep_merge_dicts(self, base, override):
        result = dict(base)
        for key, value in override.items():
            if isinstance(result.get(key), dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        return result

    def _process_base_configs(self, config_dict):
        base_paths = config_dict.pop("base_configs", None)
        if not base_paths:
            return config_dict
        if isinstance(base_paths, str):
            base_paths = [base_paths]
        merged = {}
        for base_path in base_paths:
            path = Path(base_path)
            if not path.is_absolute():
                path = Path.cwd() / path
            if not path.exists():
                print(f"Warning: Base config not found: {path}")
                continue
            with open(path, "r", encoding="utf-8") as f:
                base_config = yaml.safe_load(f) or {}
            merged = self._deep_merge_dicts(merged, base_config)
        return self._deep_merge_dicts(merged, config_dict)

    def parse_arguments(self, parser, args):
        use_sys_argv = args is None
        args = list(sys.argv[1:] if use_sys_argv else args)
        config_path = None
        for idx, arg in enumerate(args):
            if arg in ("--config", "-c") and idx + 1 < len(args):
                config_path = args[idx + 1]
                break
            if arg.startswith("--config="):
                config_path = arg.split("=", 1)[1]
                break
        if config_path:
            path = Path(config_path)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                if "base_configs" in config:
                    merged = self._process_base_configs(config)
                    logger_args = (
                        merged.get("trainer", {})
                        .get("logger", {})
                        .get("init_args", {})
                    )
                    if logger_args:
                        logger_args["name"] = path.stem
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
                    ) as tmp:
                        yaml.safe_dump(merged, tmp, sort_keys=False)
                        tmp_path = tmp.name
                    for idx, arg in enumerate(args):
                        if arg == config_path:
                            args[idx] = tmp_path
                            break
                        if (
                            arg in ("--config", "-c")
                            and idx + 1 < len(args)
                            and args[idx + 1] == config_path
                        ):
                            args[idx + 1] = tmp_path
                            break
                        if arg.startswith("--config=") and arg.split("=", 1)[1] == config_path:
                            args[idx] = f"--config={tmp_path}"
                            break
        if use_sys_argv:
            sys.argv = [sys.argv[0]] + args
            return super().parse_arguments(parser, None)
        return super().parse_arguments(parser, args)


if __name__ == "__main__":
    CustomLightningCLI(seed_everything_default=42)
