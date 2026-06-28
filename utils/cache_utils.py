import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Set

import yaml


def compute_cache_config_hash(config: Dict, backbone_config: Dict) -> str:
    if not "scales" in config["patch_extraction"]:
        print(f"No scales in config['patch_extraction']")
        return ""
    patch_iter_config = config.get(
        "patch_iterative", backbone_config.get("iterative", {})
    )
    canonical_config = {
        "backbone": {
            "checkpoint": config["backbone"]["checkpoint"],
        },
        "patch_iterative": {
            "num_iterations": patch_iter_config.get("num_iterations", 3),
            "use_confidence": patch_iter_config.get("use_confidence", True),
        },
        "scales": sorted(
            config["patch_extraction"]["scales"],
            key=lambda x: json.dumps(x, sort_keys=True),
        ),
        "grid_size": config.get("grid_size", 0.02),
        "pca_max_nn": config.get("pca_max_nn", 30),
        "feature_extraction": {
            "use_encoder_features": config.get("feature_extraction", {}).get(
                "use_encoder_features", False
            ),
            "pooling_method": config.get("feature_extraction", {}).get(
                "pooling_method", "mean"
            ),
        },
        "use_inverse_features": config.get("use_inverse_features", False),
    }
    augmentation_config = config.get("augmentation", {})
    if augmentation_config and augmentation_config.get("enabled", False):
        canonical_config["augmentation"] = {
            "enabled": True,
            "rotations_z_deg": augmentation_config.get("rotations_z_deg", [0]),
            "downsample_rates": augmentation_config.get("downsample_rates", [1.0]),
            "output_subfolder": augmentation_config.get(
                "output_subfolder", "train_augmented"
            ),
        }
    config_str = json.dumps(canonical_config, sort_keys=True)
    hash_obj = hashlib.md5(config_str.encode())
    return hash_obj.hexdigest()[:8]


def get_global_flip_cache_dir(
    config: Dict,
    backbone_config: Optional[Dict] = None,
    auto_load_backbone_config: bool = True,
) -> str:
    if "feat_extraction_config" in config:
        from utils.config import load_and_merge_feat_extraction_config

        config = load_and_merge_feat_extraction_config(config)
    if not "scales" in config.get("patch_extraction", {}):
        print(f"No scales in config['patch_extraction']")
        return os.path.join(config["data"]["root"], f"global_flip_cache")
    if backbone_config is None and auto_load_backbone_config:
        from utils.config import load_config

        backbone_config_path = config["backbone"]["config"]
        backbone_config = load_config(backbone_config_path)
    config_hash = compute_cache_config_hash(config, backbone_config)
    data_root = config["data"]["root"]
    cache_dir = os.path.join(data_root, f"global_flip_cache_{config_hash}")
    return cache_dir


def save_cache_metadata(
    cache_root: str,
    config: Dict,
    backbone_config: Dict,
    dataset_stats: Optional[Dict] = None,
):
    os.makedirs(cache_root, exist_ok=True)
    config_hash = compute_cache_config_hash(config, backbone_config)
    patch_iter_config = config.get(
        "patch_iterative", backbone_config.get("iterative", {})
    )
    metadata = {
        "cache_version": "1.0",
        "config_hash": config_hash,
        "created_at": datetime.now().isoformat(),
        "backbone": {
            "checkpoint": config["backbone"]["checkpoint"],
            "config_path": config["backbone"]["config"],
        },
        "patch_iterative": {
            "num_iterations": patch_iter_config.get("num_iterations", 3),
            "use_confidence": patch_iter_config.get("use_confidence", True),
        },
        "use_inverse_features": config.get("use_inverse_features", False),
        "augmentation": config.get("augmentation", {}),
        "patch_extraction": {
            "num_scales": len(config["patch_extraction"]["scales"]),
            "scales": config["patch_extraction"]["scales"],
        },
        "feature_extraction": {
            "use_encoder_features": config.get("feature_extraction", {}).get(
                "use_encoder_features", False
            ),
            "pooling_method": config.get("feature_extraction", {}).get(
                "pooling_method", "mean"
            ),
        },
        "processing_params": {
            "grid_size": config.get("grid_size", 0.02),
            "pca_max_nn": config.get("pca_max_nn", 30),
            "batch_size": config.get("batch_size", 8),
        },
        "dataset_info": {"data_root": config["data"]["root"]},
    }
    if dataset_stats:
        metadata["dataset_info"].update(dataset_stats)
    metadata_path = os.path.join(cache_root, "cache_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved cache metadata to {metadata_path}")
    precompute_config_path = os.path.join(cache_root, "precompute_config.yaml")
    with open(precompute_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    backbone_config_path = os.path.join(cache_root, "backbone_config.yaml")
    with open(backbone_config_path, "w") as f:
        yaml.dump(backbone_config, f, default_flow_style=False)
    print(f"Saved config backups to {cache_root}/")


def load_and_verify_cache_metadata(
    cache_root: str, config: Dict, backbone_config: Dict, verbose: bool = True
) -> Tuple[bool, Optional[Dict]]:
    metadata_path = os.path.join(cache_root, "cache_metadata.json")
    if not "scales" in config["patch_extraction"]:
        print(f"No scales in config['patch_extraction']")
        return True, None
    if not os.path.exists(cache_root):
        if verbose:
            print(f"Cache directory does not exist: {cache_root}")
        return False, None
    if not os.path.exists(metadata_path):
        if verbose:
            print(f"Cache metadata not found: {metadata_path}")
        return False, None
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except Exception as e:
        if verbose:
            print(f"Failed to load cache metadata: {e}")
        return False, None
    current_hash = compute_cache_config_hash(config, backbone_config)
    cached_hash = metadata.get("config_hash", "")
    if current_hash != cached_hash:
        if verbose:
            print(f"Cache configuration mismatch!")
            print(f"  Current config hash: {current_hash}")
            print(f"  Cached config hash:  {cached_hash}")
            print(f"\nPossible reasons:")
            print(f"  - Backbone checkpoint changed")
            print(f"  - Scales configuration changed")
            print(f"  - grid_size or pca_max_nn changed")
            print(
                f"\nPlease run precompute_patch_features_multiscale.py with the current config."
            )
        return False, metadata
    if verbose:
        print(f"✓ Cache metadata verified (hash: {current_hash})")
        print(f"  Created at: {metadata.get('created_at', 'unknown')}")
        print(f"  Backbone: {metadata['backbone']['checkpoint']}")
        print(f"  Scales: {metadata['patch_extraction']['num_scales']}")
        if "dataset_info" in metadata:
            splits = metadata["dataset_info"].get("splits_processed", [])
            if splits:
                print(f"  Splits: {', '.join(splits)}")
    return True, metadata


def compute_inference_cache_hash(config: Dict, m1_checkpoint: str) -> str:
    patch_config = config["patch_extraction"]
    inference_config = config.get("inference", {})
    iterative_config = config.get("iterative", {})
    canonical_config = {
        "m1_checkpoint": m1_checkpoint,
        "patch_extraction": {
            "method": patch_config["method"],
            "max_points_per_patch": patch_config["max_points_per_patch"],
            "overlap_rate": patch_config.get("overlap_rate", 0.0),
        },
        "iterative": {
            "num_iterations": iterative_config.get("num_iterations", 1),
            "use_confidence": iterative_config.get("use_confidence", False),
        },
        "inference": {
            "grid_size": inference_config.get("grid_size", 0.02),
            "pca_max_nn": inference_config.get("pca_max_nn", 30),
        },
    }
    config_str = json.dumps(canonical_config, sort_keys=True)
    hash_obj = hashlib.md5(config_str.encode())
    return hash_obj.hexdigest()[:8]


def get_inference_cache_dir(output_dir: str, config: Dict, m1_checkpoint: str) -> str:
    config_hash = compute_inference_cache_hash(config, m1_checkpoint)
    parent_dir = Path(output_dir).parent.parent
    cache_dir = parent_dir / "cache" / f"inference_cache_{config_hash}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)


def save_inference_cache(
    cache_dir: str,
    model_name: str,
    coords,
    patches,
    patch_normals,
    patch_centers,
    patch_features,
    m2_enabled: bool = False,
):
    import numpy as np
    import torch

    os.makedirs(cache_dir, exist_ok=True)
    patch_sizes = [len(p) for p in patches]
    max_patch_size = max(patch_sizes)
    patches_array = np.full((len(patches), max_patch_size), -1, dtype=np.int64)
    for i, patch in enumerate(patches):
        patch_np = patch.cpu().numpy() if isinstance(patch, torch.Tensor) else patch
        patches_array[i, : len(patch_np)] = patch_np
    all_normals = []
    for normals in patch_normals:
        normals_np = (
            normals.cpu().numpy() if isinstance(normals, torch.Tensor) else normals
        )
        all_normals.append(normals_np)
    centers_list = [
        c.cpu().numpy() if isinstance(c, torch.Tensor) else c for c in patch_centers
    ]
    features_list = [
        f.cpu().numpy() if isinstance(f, torch.Tensor) else f for f in patch_features
    ]
    cache_data = {
        "coords": coords,
        "num_patches": len(patches),
        "m2_enabled": m2_enabled,
        "patch_sizes": np.array(patch_sizes, dtype=np.int32),
        "patches": patches_array,
        "patch_normals": np.concatenate(all_normals, axis=0),
        "patch_centers": np.stack(centers_list, axis=0),
        "patch_features": np.stack(features_list, axis=0),
    }
    cache_path = os.path.join(cache_dir, f"{model_name}.npz")
    np.savez_compressed(cache_path, **cache_data)


def load_inference_cache(cache_dir: str, model_name: str, device: str = "cuda"):
    import numpy as np
    import torch

    cache_path = os.path.join(cache_dir, f"{model_name}.npz")
    if not os.path.exists(cache_path):
        return None
    try:
        data = np.load(cache_path)
        coords = data["coords"]
        num_patches = int(data["num_patches"])
        patch_sizes = data["patch_sizes"]
        patches_array = data["patches"]
        m2_enabled = bool(data["m2_enabled"])
        patches = []
        for i in range(num_patches):
            size = patch_sizes[i]
            patch_indices = patches_array[i, :size]
            patches.append(torch.from_numpy(patch_indices).to(device))
        patch_normals_flat = data["patch_normals"]
        patch_normals = []
        offset = 0
        for size in patch_sizes:
            normals = patch_normals_flat[offset : offset + size]
            patch_normals.append(torch.from_numpy(normals).float())
            offset += size
        centers_array = data["patch_centers"]
        features_array = data["patch_features"]
        patch_centers = [
            torch.from_numpy(centers_array[i]).float() for i in range(num_patches)
        ]
        patch_features = [
            torch.from_numpy(features_array[i]).float() for i in range(num_patches)
        ]
        return coords, patches, patch_normals, patch_centers, patch_features, m2_enabled
    except Exception as e:
        print(f"Warning: Failed to load cache for {model_name}: {e}")
        return None


def save_inference_cache_metadata(cache_dir: str, config: Dict, m1_checkpoint: str):
    os.makedirs(cache_dir, exist_ok=True)
    metadata = {
        "cache_version": "1.0",
        "cache_type": "inference",
        "config_hash": compute_inference_cache_hash(config, m1_checkpoint),
        "created_at": datetime.now().isoformat(),
        "m1_checkpoint": m1_checkpoint,
        "patch_extraction": config["patch_extraction"],
        "iterative": config.get("iterative", {}),
        "inference": config.get("inference", {}),
        "use_inverse_features": config.get("use_inverse_features", False),
    }
    metadata_path = os.path.join(cache_dir, "cache_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
