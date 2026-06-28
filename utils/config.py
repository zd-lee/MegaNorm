import yaml

import os


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    base_config_url = config.get("base_config_url", None)
    if base_config_url is not None:
        if not os.path.exists(base_config_url):
            raise FileNotFoundError(f"Base config file not found: {base_config_url}")
        with open(base_config_url, "r") as f:
            base_config = yaml.safe_load(f)
        validate_config_keys(config, base_config, prefix="")
        merged_config = deep_merge_configs(base_config, config)
        return merged_config
    return config


def validate_config_keys(current_config, base_config, prefix=""):
    if not isinstance(current_config, dict):
        return
    for key, value in current_config.items():
        if key == "base_config_url":
            continue
        current_key_path = f"{prefix}.{key}" if prefix else key
        if key not in base_config:
            raise ValueError(
                f"Configuration key '{current_key_path}' not found in base config. "
                f"All keys in the current config must exist in the base config."
            )
        if isinstance(value, dict) and isinstance(base_config[key], dict):
            validate_config_keys(value, base_config[key], current_key_path)


def deep_merge_configs(base_config, override_config):
    import copy

    merged = copy.deepcopy(base_config)
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_and_merge_feat_extraction_config(config):
    if not config.get("feat_extraction_config"):
        return config

    import copy

    result = copy.deepcopy(config)
    feat_config = load_config(config["feat_extraction_config"])
    for key in [
        "patch_extraction",
        "backbone",
        "patch_iterative",
        "feature_extraction",
        "use_inverse_features",
        "augmentation",
        "grid_size",
        "pca_max_nn",
    ]:
        if key in feat_config:
            result[key] = feat_config[key]
    return result
