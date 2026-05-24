"""
Multi-Scale Patch Dataset for Global Flip Optimization
动态分割点云成multi-scale patches，支持按patch或按模型索引访问
支持磁盘缓存以加速初始化
"""

import os
import json
import hashlib
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
import random

from dataset.patch_extractor import extract_patches_unified


class MultiScalePatchDataset(Dataset):
    """
    Multi-scale patch dataset that creates patches at multiple scales

    Inherits from PatchDataset but handles multiple scale configurations
    """

    def __init__(self,
                 data_root: str,
                 scales: List[Dict],
                 device: str = 'cuda',
                 grid_size: float = 0.02,
                 force_rebuild: bool = False,
                 transform=None):
        """
        Args:
            data_root: Root directory containing .ply files
            scales: List of scale configurations, each containing:
                - max_points_per_patch: int
                - method: str ('kdtree', 'bfs', 'grid', 'fps', 'knn')
                - k: Optional[int] (for knn-based methods)
                - patch_count: Optional[int]
            device: Device for patch extraction ('cuda' or 'cpu')
            grid_size: Grid size for voxelization/serialization
            force_rebuild: If True, ignore cache and rebuild all patches
            transform: Transform to apply to point_data (normalization, etc.)
        """
        self.data_root = data_root
        self.scales = scales
        self.device = device
        self.num_scales = len(scales)
        self.grid_size = grid_size
        self.force_rebuild = force_rebuild
        self.transform = transform

        # Scan all .ply files (reuse PatchDataset logic)
        self.model_files = sorted([
            f for f in os.listdir(data_root)
            if f.endswith('.ply')
        ])

        # Cache directory
        self.cache_dir = self._get_cache_dir()

        # Cache: model_idx → {scale_idx → (coords, normals, patch_list)}
        self.model_cache: Dict[int, Dict[int, Tuple]] = {}

        # Patch metadata with scale information
        self.patch_metadata: List[Dict] = []

        # Track patches: model_idx → {scale_idx → [global_patch_indices]}
        self.model_to_patches: Dict[int, Dict[int, List[int]]] = {}

        # Pre-build patch index across all scales (with caching)
        self._build_patch_index()

    def _compute_scales_hash(self) -> str:
        """Generate 8-char hash from canonical scale configs"""
        canonical = []
        for scale in self.scales:
            # Only include non-null values in sorted order
            config_dict = {
                'max_points_per_patch': scale['max_points_per_patch'],
                'method': scale['method'],
            }
            # Add optional params only if present
            if 'k' in scale and scale['k'] is not None:
                config_dict['k'] = scale['k']
            if 'patch_count' in scale and scale['patch_count'] is not None:
                config_dict['patch_count'] = scale['patch_count']
            if 'overlap_rate' in scale:
                config_dict['overlap_rate'] = scale.get('overlap_rate', 0.0)

            canonical.append(config_dict)

        json_str = json.dumps(canonical, sort_keys=True)
        hash_obj = hashlib.md5(json_str.encode())
        return hash_obj.hexdigest()[:8]

    def _get_cache_dir(self) -> Path:
        """Get cache directory path"""
        scales_hash = self._compute_scales_hash()
        cache_dir = Path(self.data_root).parent / f"patch_cache_{scales_hash}"
        return cache_dir

    def _get_cache_path(self, model_idx: int) -> Path:
        """Get cache file path for a specific model"""
        model_name = Path(self.model_files[model_idx]).stem
        return self.cache_dir / f"{model_name}.npz"

    def _is_cache_valid(self, model_idx: int) -> bool:
        """Check if cache file for a model is valid"""
        cache_path = self._get_cache_path(model_idx)

        if not cache_path.exists():
            return False

        try:
            data = np.load(cache_path)

            # Check required keys
            if 'num_scales' not in data or 'scale_configs' not in data:
                return False

            # Check scale configs match
            cached_configs = json.loads(str(data['scale_configs']))
            if len(cached_configs) != self.num_scales:
                return False

            # Validate each scale's data present
            for scale_idx in range(self.num_scales):
                prefix = f'scale_{scale_idx}_'
                required_keys = [
                    f'{prefix}patch_indices',
                    f'{prefix}patch_offsets',
                    f'{prefix}num_patches'
                ]
                if not all(key in data for key in required_keys):
                    return False

                # Check structural integrity
                indices = data[f'{prefix}patch_indices']
                offsets = data[f'{prefix}patch_offsets']
                if len(offsets) > 0 and offsets[-1] != len(indices):
                    return False

            return True

        except Exception as e:
            # print(f"Cache validation error for model {model_idx}: {e}")
            return False

    def _check_all_cache_status(self) -> Tuple[List[int], List[int]]:
        """Check which models have valid cache

        Returns:
            (valid_models, invalid_models): Lists of model indices
        """
        if not self.cache_dir.exists() or self.force_rebuild:
            # No cache or force rebuild
            return ([], list(range(len(self.model_files))))

        valid_models = []
        invalid_models = []

        for model_idx in range(len(self.model_files)):
            if self._is_cache_valid(model_idx):
                valid_models.append(model_idx)
            else:
                invalid_models.append(model_idx)

        return (valid_models, invalid_models)

    def _save_model_cache(self, model_idx: int, patches_by_scale: Dict[int, List[torch.Tensor]]):
        """Save extracted patches to cache file

        Args:
            model_idx: Model index
            patches_by_scale: Dict[scale_idx -> List[patch_indices_tensor]]
        """
        # Create cache directory if needed
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'num_scales': self.num_scales,
            'scale_configs': json.dumps([dict(s) for s in self.scales])
        }

        # Save each scale's data in flattened format
        for scale_idx, patch_list in patches_by_scale.items():
            prefix = f'scale_{scale_idx}_'

            # Flatten patch indices
            if len(patch_list) > 0:
                # Concatenate all patch indices
                all_indices = torch.cat(patch_list).numpy()

                # Compute offsets
                patch_sizes = [len(p) for p in patch_list]
                offsets = np.cumsum([0] + patch_sizes, dtype=np.int64)

                save_dict[f'{prefix}patch_indices'] = all_indices
                save_dict[f'{prefix}patch_offsets'] = offsets
                save_dict[f'{prefix}num_patches'] = len(patch_list)
            else:
                # Empty scale
                save_dict[f'{prefix}patch_indices'] = np.array([], dtype=np.int64)
                save_dict[f'{prefix}patch_offsets'] = np.array([0], dtype=np.int64)
                save_dict[f'{prefix}num_patches'] = 0

        # Save to disk
        cache_path = self._get_cache_path(model_idx)
        np.savez(cache_path, **save_dict)

    def _load_model_from_cache(self, model_idx: int) -> Dict[int, List[torch.Tensor]]:
        """Load patches from cache file

        Returns:
            Dict[scale_idx -> List[patch_indices_tensor]]
        """
        cache_path = self._get_cache_path(model_idx)
        data = np.load(cache_path)

        patches_by_scale = {}

        for scale_idx in range(self.num_scales):
            prefix = f'scale_{scale_idx}_'

            indices = data[f'{prefix}patch_indices']
            offsets = data[f'{prefix}patch_offsets']

            # Reconstruct patch list
            patch_list = []
            for i in range(len(offsets) - 1):
                start, end = offsets[i], offsets[i + 1]
                patch_indices = torch.from_numpy(indices[start:end].copy())
                patch_list.append(patch_indices)

            patches_by_scale[scale_idx] = patch_list

        return patches_by_scale

    def _build_patch_index(self):
        """Build patch index with disk caching support

        Three loading modes:
        1. Complete cache: Load all from disk (fast)
        2. No cache: Build all with progress bar (first run)
        3. Partial cache: Hybrid - load valid, rebuild invalid
        """
        valid_models, invalid_models = self._check_all_cache_status()

        if len(invalid_models) == len(self.model_files):
            # Mode 2: No cache or force rebuild - build all with progress bar
            print(f"Building patch cache for {len(self.model_files)} models...")
            self._build_and_cache_all(self.model_files)
        elif len(invalid_models) > 0:
            # Mode 3: Partial cache - hybrid approach
            print(f"Loading {len(valid_models)} models from cache...")
            self._load_from_cache(valid_models)
            print(f"Rebuilding {len(invalid_models)} invalid/missing models...")
            self._build_and_cache_all([self.model_files[i] for i in invalid_models],
                                     model_indices=invalid_models)
        else:
            # Mode 1: Complete cache - fast load
            print(f"Loading {len(valid_models)} models from cache...")
            self._load_from_cache(valid_models)

        print(f"Dataset ready: {len(self.patch_metadata)} total patches across {self.num_scales} scales")

    def _build_and_cache_all(self, model_files: List[str], model_indices: Optional[List[int]] = None):
        """Build patches for models with progress bar and save to cache

        Args:
            model_files: List of model filenames to process
            model_indices: Optional list of model indices (if not processing all models)
        """
        if model_indices is None:
            model_indices = list(range(len(model_files)))

        combined = list(zip(model_indices, model_files))

        rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', '0')))
        random.seed(42 + rank)
        random.shuffle(combined)

        model_indices, model_files = zip(*combined)
        model_indices = list(model_indices)
        model_files = list(model_files)

        pbar = tqdm(zip(model_indices, model_files), total=len(model_files),
                   desc="Building patch cache")

        for model_idx, model_file in pbar:
            if not self.force_rebuild and self._is_cache_valid(model_idx):
                start_meta_len = len(self.patch_metadata)
                self._load_from_cache([model_idx])
                total_patches = len(self.patch_metadata) - start_meta_len
                pbar.set_postfix({
                    'scales': self.num_scales,
                    'patches': total_patches,
                    'status': 'cached'
                })
                continue

            self.model_to_patches[model_idx] = {}
            self.model_cache[model_idx] = {}

            coords, gt_normals = self._load_model(model_idx)

            patches_by_scale = {}
            global_patch_idx = len(self.patch_metadata)

            for scale_idx, scale_config in enumerate(self.scales):
                patch_list = self._extract_patches_at_scale(coords, scale_config)

                self.model_cache[model_idx][scale_idx] = (coords, gt_normals, patch_list)
                patches_by_scale[scale_idx] = patch_list

                self.model_to_patches[model_idx][scale_idx] = []
                for patch_idx_in_scale, patch_indices in enumerate(patch_list):
                    self.patch_metadata.append({
                        'model_idx': model_idx,
                        'scale_idx': scale_idx,
                        'patch_idx_in_scale': patch_idx_in_scale,
                        'patch_indices': patch_indices,
                        'global_patch_idx': global_patch_idx,
                        'max_points': scale_config['max_points_per_patch'],
                        'method': scale_config['method']
                    })
                    self.model_to_patches[model_idx][scale_idx].append(global_patch_idx)
                    global_patch_idx += 1

            self._save_model_cache(model_idx, patches_by_scale)

            total_patches = sum(len(pl) for pl in patches_by_scale.values())
            pbar.set_postfix({
                'scales': self.num_scales,
                'patches': total_patches,
                'status': 'rebuilt'
            })

    def _load_from_cache(self, model_indices: List[int]):
        """Load patches from cache for specified models

        Args:
            model_indices: List of model indices to load
        """
        global_patch_idx = len(self.patch_metadata)  # Continue from current count

        for model_idx in model_indices:
            self.model_to_patches[model_idx] = {}
            self.model_cache[model_idx] = {}

            # Load from cache
            patches_by_scale = self._load_model_from_cache(model_idx)

            # Load coords and normals (still needed for training)
            coords, gt_normals = self._load_model(model_idx)

            for scale_idx in range(self.num_scales):
                patch_list = patches_by_scale[scale_idx]

                # Cache model data
                self.model_cache[model_idx][scale_idx] = (coords, gt_normals, patch_list)

                # Build metadata
                self.model_to_patches[model_idx][scale_idx] = []
                scale_config = self.scales[scale_idx]

                for patch_idx_in_scale, patch_indices in enumerate(patch_list):
                    self.patch_metadata.append({
                        'model_idx': model_idx,
                        'scale_idx': scale_idx,
                        'patch_idx_in_scale': patch_idx_in_scale,
                        'patch_indices': patch_indices,
                        'global_patch_idx': global_patch_idx,
                        'max_points': scale_config['max_points_per_patch'],
                        'method': scale_config['method']
                    })
                    self.model_to_patches[model_idx][scale_idx].append(global_patch_idx)
                    global_patch_idx += 1

    def _load_model(self, model_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a single model's coordinates and normals"""
        ply_path = os.path.join(self.data_root, self.model_files[model_idx])
        pcd = o3d.io.read_point_cloud(ply_path)

        # Extract coordinates and normals
        coords = torch.from_numpy(np.asarray(pcd.points).astype(np.float32))
        gt_normals = torch.from_numpy(np.asarray(pcd.normals).astype(np.float32))

        return coords, gt_normals

    def _extract_patches_at_scale(
        self,
        coords: torch.Tensor,
        scale_config: Dict
    ) -> List[torch.Tensor]:
        """Extract patches using specified scale configuration"""
        coords_device = coords.to(self.device)

        # Prepare parameters for extract_patches_unified
        method = scale_config['method']
        num_per_patch = scale_config['max_points_per_patch']
        k = scale_config.get('k', 10)
        patch_count = scale_config.get('patch_count', None)
        overlap_rate = scale_config.get('overlap_rate', 0.0)

        # Handle case where model is too small for the scale
        # If patch_count would be 0, create 1 patch with all points
        if patch_count is None:
            calculated_patch_count = int(len(coords) / num_per_patch)
            if calculated_patch_count == 0:
                # Model too small for this scale - create single patch with all points
                # Only return if it has more than 1 point to avoid BatchNorm errors
                if len(coords) > 1:
                    return [torch.arange(len(coords))]
                else:
                    return []
            patch_count = calculated_patch_count

        # Extract patches
        patch_indices_list, _ = extract_patches_unified(
            coords_device,
            method=method,
            num_per_patch=num_per_patch,
            k=k,
            patch_count=patch_count,
            overlap_rate=overlap_rate,
            device=self.device
        )

        # Convert to CPU and filter out single-point patches
        # BatchNorm requires at least 2 points to calculate variance during training
        patch_list = [indices.cpu() for indices in patch_indices_list if len(indices) > 1]

        return patch_list

    def __len__(self) -> int:
        return len(self.patch_metadata)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Return single patch with scale metadata

        Returns:
            point_data: Dict containing:
                - coord: (M, 3)
                - feat: (M, 3) (coords as features)
                - batch: (M,) all zeros
                - filename: str
                - patch_idx: int (within scale)
                - scale_idx: int (NEW)
                - max_points: int (NEW)
                - method: str (NEW)
            gt_normals: (M, 3)
        """
        metadata = self.patch_metadata[idx]
        model_idx = metadata['model_idx']
        scale_idx = metadata['scale_idx']
        patch_indices = metadata['patch_indices']

        # Get model data from cache
        coords, gt_normals, _ = self.model_cache[model_idx][scale_idx]

        # Extract patch data
        patch_coords = coords[patch_indices]
        patch_normals = gt_normals[patch_indices]

        # Build point_data dictionary (compatible with NormalEstimationDataset format)
        point_data = {
            'coord': patch_coords,
            'feat': patch_coords,  # Features are coordinates
            'batch': torch.zeros(len(patch_coords), dtype=torch.long),
            'grid_size': self.grid_size,  # Required by PTv3 backbone
            'filename': self.model_files[model_idx],
            'patch_idx': metadata['patch_idx_in_scale'],
            'scale_idx': scale_idx,  # NEW: scale information
            'max_points': metadata['max_points'],  # NEW: scale configuration
            'method': metadata['method']  # NEW: extraction method
        }

        # Apply transform for normalization if provided
        if self.transform is not None:
            try:
                result = self.transform(point_data, patch_normals)
                if result is None:
                    raise ValueError("Transform returned None")
                point_data, patch_normals = result
            except Exception as e:
                print(f"Error in transform: {e}")
                print(f"point_data keys: {point_data.keys()}")
                print(f"patch_normals shape: {patch_normals.shape if patch_normals is not None else None}")
                raise

        return point_data, patch_normals

    def get_model_patches_all_scales(self, model_idx: int) -> Dict:
        """Get all patches for a model across all scales

        Returns:
            Dict containing:
                - coords: (N, 3) full point cloud
                - gt_normals: (N, 3) full normals
                - patches_by_scale: Dict[scale_idx -> List[patch_indices]]
        """
        if model_idx not in self.model_cache:
            raise ValueError(f"Model {model_idx} not in cache")

        # Get coords and normals from first scale (same for all scales)
        coords, gt_normals, _ = self.model_cache[model_idx][0]

        # Collect patches by scale
        patches_by_scale = {}
        for scale_idx in range(self.num_scales):
            _, _, patch_list = self.model_cache[model_idx][scale_idx]
            patches_by_scale[scale_idx] = patch_list

        return {
            'coords': coords,
            'gt_normals': gt_normals,
            'patches_by_scale': patches_by_scale
        }

    def get_num_models(self) -> int:
        """Return total number of models"""
        return len(self.model_files)

    def get_model_name(self, model_idx: int) -> str:
        """Return model filename"""
        return self.model_files[model_idx]

    def get_num_scales(self) -> int:
        """Return number of scales"""
        return self.num_scales

    def get_scale_config(self, scale_idx: int) -> Dict:
        """Return configuration for a specific scale"""
        return self.scales[scale_idx]

    def get_patches_for_scale(self, model_idx: int, scale_idx: int) -> List[int]:
        """Get global patch indices for a specific model and scale"""
        return self.model_to_patches[model_idx][scale_idx]

    def clear_cache(self):
        """Clear the cache directory for this dataset"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            print(f"Cleared cache directory: {self.cache_dir}")
        else:
            print(f"Cache directory does not exist: {self.cache_dir}")

    def get_cache_info(self) -> Dict:
        """Get information about the cache

        Returns:
            Dict containing cache statistics
        """
        info = {
            'cache_dir': str(self.cache_dir),
            'cache_exists': self.cache_dir.exists(),
            'scales_hash': self._compute_scales_hash(),
            'num_models': len(self.model_files),
            'num_scales': self.num_scales,
        }

        if self.cache_dir.exists():
            cache_files = list(self.cache_dir.glob('*.npz'))
            info['cached_models'] = len(cache_files)
            info['cache_complete'] = len(cache_files) == len(self.model_files)

            # Estimate cache size
            total_size = sum(f.stat().st_size for f in cache_files)
            info['total_size_mb'] = total_size / (1024 * 1024)
        else:
            info['cached_models'] = 0
            info['cache_complete'] = False
            info['total_size_mb'] = 0

        return info
