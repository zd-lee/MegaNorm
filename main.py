import torch
import sys
import os
from pathlib import Path
from lightning.pytorch.cli import LightningCLI
import yaml

torch.set_float32_matmul_precision("medium")

os.makedirs("pl_logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

class CustomLightningCLI(LightningCLI):
    def _deep_merge_dicts(self, base: dict, override: dict) -> dict:
        """Deep merge two dictionaries. override values take precedence."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        return result

    def _process_base_configs(self, config_dict: dict) -> dict:
        """Process base_configs field and merge configurations."""
        if 'base_configs' not in config_dict:
            return config_dict

        base_paths = config_dict['base_configs']
        if isinstance(base_paths, str):
            base_paths = [base_paths]

        merged_config = {}
        for base_path in base_paths:
            base_path = Path(base_path)
            if not base_path.is_absolute():
                base_path = Path.cwd() / base_path

            if base_path.exists():
                with open(base_path, 'r') as f:
                    base_config = yaml.safe_load(f)
                    if base_config:
                        merged_config = self._deep_merge_dicts(merged_config, base_config)
            else:
                print(f"Warning: Base config not found: {base_path}")

        del config_dict['base_configs']
        return self._deep_merge_dicts(merged_config, config_dict)

    def parse_arguments(self, parser, args):
        """Override to process base_configs before parsing."""
        if args is None:
            args = sys.argv[1:]

        config_path = None
        for i, arg in enumerate(args):
            if arg in ['--config', '-c'] and i + 1 < len(args):
                config_path = args[i + 1]
                break
            elif arg.startswith('--config='):
                config_path = arg.split('=', 1)[1]
                break

        if config_path:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_dict = yaml.safe_load(f)
                    if config_dict and 'base_configs' in config_dict:
                        merged = self._process_base_configs(config_dict)

                        config_name = config_file.stem
                        if 'trainer' in merged and 'logger' in merged['trainer']:
                            if 'init_args' in merged['trainer']['logger']:
                                merged['trainer']['logger']['init_args']['name'] = config_name

                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
                            yaml.dump(merged, tmp)
                            tmp_path = tmp.name

                        for i, arg in enumerate(args):
                            if arg == config_path:
                                args[i] = tmp_path
                                break
                            elif arg == '--config' and i + 1 < len(args) and args[i + 1] == config_path:
                                args[i + 1] = tmp_path
                                break
                            elif arg.startswith('--config=') and arg.split('=', 1)[1] == config_path:
                                args[i] = f'--config={tmp_path}'
                                break

        return super().parse_arguments(parser, args)

cli = CustomLightningCLI(seed_everything_default=42)
