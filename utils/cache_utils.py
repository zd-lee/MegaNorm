"""
Cache Management Utilities for Global Flip Optimization
自动管理基于配置hash的缓存路径和元信息
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Set
import yaml


def compute_cache_config_hash(config: Dict, backbone_config: Dict) -> str:
    """
    计算缓存配置的hash值

    基于影响特征计算的关键配置生成8位MD5 hash

    Args:
        config: 全局配置（来自 configs/global_flip/xxx.yaml）
        backbone_config: backbone配置（来自 configs/direct_orientation/xxx.yaml）

    Returns:
        8位hash字符串
    """
    if not 'scales' in config['patch_extraction']:
        print(f"No scales in config['patch_extraction']")
        return ''

    # M1 预计算迭代配置（总是包含在 hash 中）
    patch_iter_config = config.get('patch_iterative', backbone_config.get('iterative', {}))

    # 构建规范化的配置字典（只包含影响特征计算的关键参数）
    canonical_config = {
        'backbone': {
            'checkpoint': config['backbone']['checkpoint'],  # 直接使用原始路径
        },
        'patch_iterative': {
            'num_iterations': patch_iter_config.get('num_iterations', 3),
            'use_confidence': patch_iter_config.get('use_confidence', True)
        },
        'scales': sorted(
            config['patch_extraction']['scales'],
            key=lambda x: json.dumps(x, sort_keys=True)  # 按字典序排序
        ),
        'grid_size': config.get('grid_size', 0.02),
        'pca_max_nn': config.get('pca_max_nn', 30),
        # 新增：特征提取配置
        'feature_extraction': {
            'use_encoder_features': config.get('feature_extraction', {}).get('use_encoder_features', False),
            'pooling_method': config.get('feature_extraction', {}).get('pooling_method', 'mean')
        }
    }

    # M2 迭代配置（只有 enabled=true 时才加入 hash，影响特征格式）
    m2_iter_config = config.get('m2_iterative', {})
    if m2_iter_config.get('enabled', False):
        canonical_config['m2_iterative'] = {
            'enabled': True  # 明确标记启用
        }

    # 生成JSON字符串（确保键排序）
    config_str = json.dumps(canonical_config, sort_keys=True)

    # 计算MD5 hash，取前8位
    hash_obj = hashlib.md5(config_str.encode())
    return hash_obj.hexdigest()[:8]


def get_global_flip_cache_dir(
    config: Dict,
    backbone_config: Optional[Dict] = None,
    auto_load_backbone_config: bool = True
) -> str:
    """
    获取全局flip优化的缓存根目录路径

    Args:
        config: 全局配置
        backbone_config: backbone配置（如果为None且auto_load=True，则自动加载）
        auto_load_backbone_config: 是否自动加载backbone配置

    Returns:
        缓存根目录的绝对路径
    """
    # 合并特征提取配置（如果存在）
    if 'feat_extraction_config' in config:
        from utils.config import load_and_merge_feat_extraction_config
        config = load_and_merge_feat_extraction_config(config)

    if not 'scales' in config.get('patch_extraction', {}):
        print(f"No scales in config['patch_extraction']")
        return os.path.join(config['data']['root'], f'global_flip_cache')
    # 如果没有提供backbone_config，尝试自动加载
    if backbone_config is None and auto_load_backbone_config:
        from utils.config import load_config
        backbone_config_path = config['backbone']['config']
        backbone_config = load_config(backbone_config_path)

    # 计算hash
    config_hash = compute_cache_config_hash(config, backbone_config)

    # 构建缓存目录路径
    data_root = config['data']['root']
    cache_dir = os.path.join(data_root, f'global_flip_cache_{config_hash}')

    return cache_dir


def save_cache_metadata(
    cache_root: str,
    config: Dict,
    backbone_config: Dict,
    dataset_stats: Optional[Dict] = None
):
    """
    保存缓存元信息和配置备份

    Args:
        cache_root: 缓存根目录
        config: 全局配置
        backbone_config: backbone配置
        dataset_stats: 数据集统计信息（可选）
            {
                'splits_processed': ['train', 'val'],
                'num_models': {'train': 100, 'val': 20},
                'total_patches': {'train': 5000, 'val': 1000},
                'accuracy_stats': {...}
            }
    """
    os.makedirs(cache_root, exist_ok=True)

    # 1. 保存元信息JSON
    config_hash = compute_cache_config_hash(config, backbone_config)

    # 读取配置
    patch_iter_config = config.get('patch_iterative', backbone_config.get('iterative', {}))
    m2_iter_config = config.get('m2_iterative', {})

    metadata = {
        'cache_version': '1.0',
        'config_hash': config_hash,
        'created_at': datetime.now().isoformat(),
        'backbone': {
            'checkpoint': config['backbone']['checkpoint'],
            'config_path': config['backbone']['config']
        },
        'patch_iterative': {
            'num_iterations': patch_iter_config.get('num_iterations', 3),
            'use_confidence': patch_iter_config.get('use_confidence', True)
        },
        'm2_iterative': {
            'enabled': m2_iter_config.get('enabled', False),
            'num_iterations': m2_iter_config.get('num_iterations', 1)
        },
        'patch_extraction': {
            'num_scales': len(config['patch_extraction']['scales']),
            'scales': config['patch_extraction']['scales']
        },
        'feature_extraction': {
            'use_encoder_features': config.get('feature_extraction', {}).get('use_encoder_features', False),
            'pooling_method': config.get('feature_extraction', {}).get('pooling_method', 'mean')
        },
        'processing_params': {
            'grid_size': config.get('grid_size', 0.02),
            'pca_max_nn': config.get('pca_max_nn', 30),
            'batch_size': config.get('batch_size', 8)
        },
        'dataset_info': {
            'data_root': config['data']['root']
        }
    }

    # 添加数据集统计信息
    if dataset_stats:
        metadata['dataset_info'].update(dataset_stats)

    metadata_path = os.path.join(cache_root, 'cache_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved cache metadata to {metadata_path}")

    # 2. 保存配置文件备份
    precompute_config_path = os.path.join(cache_root, 'precompute_config.yaml')
    with open(precompute_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    backbone_config_path = os.path.join(cache_root, 'backbone_config.yaml')
    with open(backbone_config_path, 'w') as f:
        yaml.dump(backbone_config, f, default_flow_style=False)

    print(f"Saved config backups to {cache_root}/")


def load_and_verify_cache_metadata(
    cache_root: str,
    config: Dict,
    backbone_config: Dict,
    verbose: bool = True
) -> Tuple[bool, Optional[Dict]]:
    """
    加载并验证缓存元信息

    Args:
        cache_root: 缓存根目录
        config: 当前配置
        backbone_config: 当前backbone配置
        verbose: 是否打印详细信息

    Returns:
        (is_valid, metadata): 缓存是否有效，元信息字典
    """
    metadata_path = os.path.join(cache_root, 'cache_metadata.json')
    if not 'scales' in config['patch_extraction']:
        print(f"No scales in config['patch_extraction']")
        return True, None

    # 检查缓存目录是否存在
    if not os.path.exists(cache_root):
        if verbose:
            print(f"Cache directory does not exist: {cache_root}")
        return False, None

    # 检查元信息文件是否存在
    if not os.path.exists(metadata_path):
        if verbose:
            print(f"Cache metadata not found: {metadata_path}")
        return False, None

    # 加载元信息
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        if verbose:
            print(f"Failed to load cache metadata: {e}")
        return False, None

    # 验证hash
    current_hash = compute_cache_config_hash(config, backbone_config)
    cached_hash = metadata.get('config_hash', '')

    if current_hash != cached_hash:
        if verbose:
            print(f"Cache configuration mismatch!")
            print(f"  Current config hash: {current_hash}")
            print(f"  Cached config hash:  {cached_hash}")
            print(f"\nPossible reasons:")
            print(f"  - Backbone checkpoint changed")
            print(f"  - Scales configuration changed")
            print(f"  - grid_size or pca_max_nn changed")
            print(f"\nPlease run precompute_patch_features_multiscale.py with the current config.")
        return False, metadata

    if verbose:
        print(f"✓ Cache metadata verified (hash: {current_hash})")
        print(f"  Created at: {metadata.get('created_at', 'unknown')}")
        print(f"  Backbone: {metadata['backbone']['checkpoint']}")
        print(f"  Scales: {metadata['patch_extraction']['num_scales']}")
        if 'dataset_info' in metadata:
            splits = metadata['dataset_info'].get('splits_processed', [])
            if splits:
                print(f"  Splits: {', '.join(splits)}")

    return True, metadata


def compute_inference_cache_hash(config: Dict, m1_checkpoint: str) -> str:
    """
    计算推理缓存的hash值

    基于影响M1推理的关键配置生成8位MD5 hash

    Args:
        config: 推理配置（来自 configs/inference/xxx.yaml）
        m1_checkpoint: M1模型checkpoint路径

    Returns:
        8位hash字符串
    """
    patch_config = config['patch_extraction']
    inference_config = config.get('inference', {})
    iterative_config = config.get('iterative', {})

    canonical_config = {
        'm1_checkpoint': m1_checkpoint,
        'patch_extraction': {
            'method': patch_config['method'],
            'max_points_per_patch': patch_config['max_points_per_patch'],
            'overlap_rate': patch_config.get('overlap_rate', 0.0)
        },
        'iterative': {
            'num_iterations': iterative_config.get('num_iterations', 1),
            'use_confidence': iterative_config.get('use_confidence', False)
        },
        'inference': {
            'grid_size': inference_config.get('grid_size', 0.02),
            'pca_max_nn': inference_config.get('pca_max_nn', 30)
        }
    }

    config_str = json.dumps(canonical_config, sort_keys=True)
    hash_obj = hashlib.md5(config_str.encode())
    return hash_obj.hexdigest()[:8]


def get_inference_cache_dir(output_dir: str, config: Dict, m1_checkpoint: str) -> str:
    """Get inference cache directory path: outputs/inference/cache/inference_cache_{hash}/"""
    config_hash = compute_inference_cache_hash(config, m1_checkpoint)
    parent_dir = Path(output_dir).parent.parent  # outputs/inference/output/{config} -> outputs/inference/
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
    m2_enabled: bool = False
):
    """Save inference cache to {cache_dir}/{model_name}.npz"""
    import numpy as np
    import torch

    os.makedirs(cache_dir, exist_ok=True)

    # Build cache data
    patch_sizes = [len(p) for p in patches]
    max_patch_size = max(patch_sizes)

    # Pad patches to uniform size
    patches_array = np.full((len(patches), max_patch_size), -1, dtype=np.int64)
    for i, patch in enumerate(patches):
        patch_np = patch.cpu().numpy() if isinstance(patch, torch.Tensor) else patch
        patches_array[i, :len(patch_np)] = patch_np

    # Concatenate patch normals
    all_normals = []
    for normals in patch_normals:
        normals_np = normals.cpu().numpy() if isinstance(normals, torch.Tensor) else normals
        all_normals.append(normals_np)

    # Stack centers and features
    centers_list = [c.cpu().numpy() if isinstance(c, torch.Tensor) else c for c in patch_centers]
    features_list = [f.cpu().numpy() if isinstance(f, torch.Tensor) else f for f in patch_features]

    cache_data = {
        'coords': coords,
        'num_patches': len(patches),
        'm2_enabled': m2_enabled,
        'patch_sizes': np.array(patch_sizes, dtype=np.int32),
        'patches': patches_array,
        'patch_normals': np.concatenate(all_normals, axis=0),
        'patch_centers': np.stack(centers_list, axis=0),
        'patch_features': np.stack(features_list, axis=0)
    }

    cache_path = os.path.join(cache_dir, f"{model_name}.npz")
    np.savez_compressed(cache_path, **cache_data)


def load_inference_cache(cache_dir: str, model_name: str, device: str = 'cuda'):
    """Load inference cache from {cache_dir}/{model_name}.npz. Returns None if not found."""
    import numpy as np
    import torch

    cache_path = os.path.join(cache_dir, f"{model_name}.npz")
    if not os.path.exists(cache_path):
        return None

    try:
        data = np.load(cache_path)
        coords = data['coords']
        num_patches = int(data['num_patches'])
        patch_sizes = data['patch_sizes']
        patches_array = data['patches']
        m2_enabled = bool(data['m2_enabled'])

        # Reconstruct patches
        patches = []
        for i in range(num_patches):
            size = patch_sizes[i]
            patch_indices = patches_array[i, :size]
            patches.append(torch.from_numpy(patch_indices).to(device))

        # Reconstruct patch normals
        patch_normals_flat = data['patch_normals']
        patch_normals = []
        offset = 0
        for size in patch_sizes:
            normals = patch_normals_flat[offset:offset+size]
            patch_normals.append(torch.from_numpy(normals).float())
            offset += size

        # Reconstruct centers and features
        centers_array = data['patch_centers']
        features_array = data['patch_features']
        patch_centers = [torch.from_numpy(centers_array[i]).float() for i in range(num_patches)]
        patch_features = [torch.from_numpy(features_array[i]).float() for i in range(num_patches)]

        return coords, patches, patch_normals, patch_centers, patch_features, m2_enabled

    except Exception as e:
        print(f"Warning: Failed to load cache for {model_name}: {e}")
        return None


def save_inference_cache_metadata(cache_dir: str, config: Dict, m1_checkpoint: str):
    """Save cache metadata to {cache_dir}/cache_metadata.json"""
    os.makedirs(cache_dir, exist_ok=True)

    metadata = {
        'cache_version': '1.0',
        'cache_type': 'inference',
        'config_hash': compute_inference_cache_hash(config, m1_checkpoint),
        'created_at': datetime.now().isoformat(),
        'm1_checkpoint': m1_checkpoint,
        'patch_extraction': config['patch_extraction'],
        'iterative': config.get('iterative', {}),
        'inference': config.get('inference', {}),
        'm2_iterative': config.get('m2_iterative', {})
    }

    metadata_path = os.path.join(cache_dir, 'cache_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
