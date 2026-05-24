import yaml
import os

def load_config(config_path):
    """
    Load configuration from YAML file with base config inheritance support

    Args:
        config_path: Path to the configuration file

    Returns:
        Merged configuration dictionary

    Raises:
        ValueError: If current config has keys not present in base config
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Check if base_config_url is specified
    base_config_url = config.get('base_config_url', None)

    if base_config_url is not None:
        # Load base configuration
        if not os.path.exists(base_config_url):
            raise FileNotFoundError(f"Base config file not found: {base_config_url}")

        with open(base_config_url, 'r') as f:
            base_config = yaml.safe_load(f)

        # Validate: check if all keys in current config exist in base config
        validate_config_keys(config, base_config, prefix='')

        # Merge configs: current config overrides base config
        merged_config = deep_merge_configs(base_config, config)
        return merged_config

    return config


def validate_config_keys(current_config, base_config, prefix=''):
    """
    Recursively validate that all keys in current_config exist in base_config

    Args:
        current_config: Current configuration dictionary
        base_config: Base configuration dictionary
        prefix: Key prefix for error messages (used in recursion)

    Raises:
        ValueError: If a key in current_config is not found in base_config
    """
    if not isinstance(current_config, dict):
        return

    for key, value in current_config.items():
        # Skip base_config_url itself (this is allowed to be new)
        if key == 'base_config_url':
            continue

        current_key_path = f"{prefix}.{key}" if prefix else key

        # Check if key exists in base config
        if key not in base_config:
            raise ValueError(
                f"Configuration key '{current_key_path}' not found in base config. "
                f"All keys in the current config must exist in the base config."
            )

        # Recursively validate nested dictionaries
        if isinstance(value, dict) and isinstance(base_config[key], dict):
            validate_config_keys(value, base_config[key], current_key_path)


def deep_merge_configs(base_config, override_config):
    """
    Deep merge two configuration dictionaries

    Args:
        base_config: Base configuration (will be overridden)
        override_config: Override configuration (takes precedence)

    Returns:
        Merged configuration dictionary
    """
    import copy
    merged = copy.deepcopy(base_config)

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = deep_merge_configs(merged[key], value)
        else:
            # Override the value
            merged[key] = value

    return merged


def load_and_merge_feat_extraction_config(config):
    """
    加载并合并特征提取配置

    如果 config 包含 feat_extraction_config 字段，加载该配置文件，
    并用其中的字段完全替换主配置中的对应字段。

    Args:
        config: 主配置字典

    Returns:
        合并后的配置（深拷贝，不修改原配置）
    """
    if 'feat_extraction_config' not in config:
        return config

    import copy
    result = copy.deepcopy(config)
    feat_config = load_config(config['feat_extraction_config'])

    # 完全替换这些字段
    for key in ['patch_extraction', 'backbone', 'patch_iterative',
                'feature_extraction', 'm2_iterative', 'grid_size', 'pca_max_nn']:
        if key in feat_config:
            result[key] = feat_config[key]

    return result