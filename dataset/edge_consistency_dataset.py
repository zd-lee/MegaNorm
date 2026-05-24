import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional
from pathlib import Path

from dataset.global_optimization_dataset import NormalEstimationNormalize


def build_knn_graph(patch_centers, k=10):
    """
    Args:
        patch_centers: (P, 3) numpy array
        k: number of neighbors

    Returns:
        edges: (P*k, 2) numpy array of [node_A, node_B] indices
    """
    P = len(patch_centers)

    distances = np.linalg.norm(
        patch_centers[:, None, :] - patch_centers[None, :, :], axis=2
    )

    np.fill_diagonal(distances, np.inf)

    knn_indices = np.argsort(distances, axis=1)[:, :k]

    source_nodes = np.repeat(np.arange(P), k)
    target_nodes = knn_indices.flatten()
    edges = np.stack([source_nodes, target_nodes], axis=1)

    return edges


def generate_edge_samples(features, edges, gt_flip_status, patch_centers):
    """
    Generate edge samples from patch features.

    Args:
        features: (P, 2, 512) or (P, 512) - features for each patch
        edges: (P*k, 2) - edge list
        gt_flip_status: (P,) - ground truth flip for each patch
        patch_centers: (P, 3) - patch center coordinates

    Returns:
        edge_features_A: (num_samples, 512)
        edge_features_B: (num_samples, 512)
        edge_labels: (num_samples,)
        edge_centers_A: (num_samples, 3)
        edge_centers_B: (num_samples, 3)

    num_samples = P*k*4 if features.shape=(P,2,512), else P*k*1
    """
    num_edges = len(edges)

    # 判断是否有反向特征
    has_inv_features = (len(features.shape) == 3 and features.shape[1] == 2)

    if has_inv_features:
        # (P, 2, 512): 每条边生成4个样本
        edge_features_A = np.zeros((num_edges * 4, 512), dtype=np.float32)
        edge_features_B = np.zeros((num_edges * 4, 512), dtype=np.float32)
        edge_labels = np.zeros(num_edges * 4, dtype=np.float32)
        edge_centers_A = np.zeros((num_edges * 4, 3), dtype=np.float32)
        edge_centers_B = np.zeros((num_edges * 4, 3), dtype=np.float32)

        for i, (node_a, node_b) in enumerate(edges):
            base_idx = i * 4

            a_pos, a_neg = features[node_a, 0], features[node_a, 1]
            b_pos, b_neg = features[node_b, 0], features[node_b, 1]

            flip_a = gt_flip_status[node_a]
            flip_b = gt_flip_status[node_b]

            # All 4 samples share the same centers
            edge_centers_A[base_idx:base_idx+4] = patch_centers[node_a]
            edge_centers_B[base_idx:base_idx+4] = patch_centers[node_b]

            # 4 combinations
            edge_features_A[base_idx] = a_pos
            edge_features_B[base_idx] = b_pos
            edge_labels[base_idx] = flip_a ^ flip_b

            edge_features_A[base_idx + 1] = a_pos
            edge_features_B[base_idx + 1] = b_neg
            edge_labels[base_idx + 1] = flip_a ^ (1 - flip_b)

            edge_features_A[base_idx + 2] = a_neg
            edge_features_B[base_idx + 2] = b_pos
            edge_labels[base_idx + 2] = (1 - flip_a) ^ flip_b

            edge_features_A[base_idx + 3] = a_neg
            edge_features_B[base_idx + 3] = b_neg
            edge_labels[base_idx + 3] = (1 - flip_a) ^ (1 - flip_b)
    else:
        # (P, 512): 每条边生成1个样本（只用正向特征）
        edge_features_A = np.zeros((num_edges, 512), dtype=np.float32)
        edge_features_B = np.zeros((num_edges, 512), dtype=np.float32)
        edge_labels = np.zeros(num_edges, dtype=np.float32)
        edge_centers_A = np.zeros((num_edges, 3), dtype=np.float32)
        edge_centers_B = np.zeros((num_edges, 3), dtype=np.float32)

        for i, (node_a, node_b) in enumerate(edges):
            edge_features_A[i] = features[node_a]  # 正向特征
            edge_features_B[i] = features[node_b]  # 正向特征

            flip_a = gt_flip_status[node_a]
            flip_b = gt_flip_status[node_b]
            edge_labels[i] = flip_a ^ flip_b

            edge_centers_A[i] = patch_centers[node_a]
            edge_centers_B[i] = patch_centers[node_b]

    return edge_features_A, edge_features_B, edge_labels, edge_centers_A, edge_centers_B


class EdgeConsistencyDataset(Dataset):
    def __init__(self, cache_root: str, cache_subdir: str, k: int = 10,
                 scale_idx: Optional[int] = None, use_inv_features: bool = True):
        """
        Args:
            cache_root: Cache root directory
            cache_subdir: Cache subdirectory (relative to cache_root)
            k: Number of KNN neighbors
            scale_idx: Optional scale index for multi-scale datasets
            use_inv_features: Whether to use inverse features (m2_iterative.enabled)
        """
        self.cache_dir = os.path.join(cache_root, cache_subdir)
        self.k = k
        self.scale_idx = scale_idx
        self.use_inv_features = use_inv_features

        self.cache_files = sorted([f for f in os.listdir(self.cache_dir) if f.endswith('.npz')])

        self.samples = []
        for file_idx, cache_file in enumerate(self.cache_files):
            cache_path = os.path.join(self.cache_dir, cache_file)
            data = np.load(cache_path)

            if 'num_scales' in data:
                num_scales = int(data['num_scales'])
                if self.scale_idx is not None:
                    self.samples.append((file_idx, self.scale_idx))
                else:
                    for s_idx in range(num_scales):
                        self.samples.append((file_idx, s_idx))
            else:
                self.samples.append((file_idx, 0))

        print(f"Found {len(self.cache_files)} cached models, total samples={len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        file_idx, scale_idx = self.samples[idx]
        cache_path = os.path.join(self.cache_dir, self.cache_files[file_idx])
        data = np.load(cache_path)

        if 'num_scales' in data:
            prefix = f'scale_{scale_idx}_'
            patch_centers = data[prefix + 'patch_centers']
            o_features = data[prefix + 'features']
            gt_flip_status = data[prefix + 'gt_flip_status']

            if self.use_inv_features:
                inv_features = data[prefix + 'inv_features']
                features = np.concatenate([o_features[:, None, :],
                                          inv_features[:, None, :]], axis=1)  # (P, 2, 512)
            else:
                features = o_features  # (P, 512)
        else:
            patch_centers = data['patch_centers']
            o_features = data['features']
            gt_flip_status = data['gt_flip_status']

            if self.use_inv_features:
                inv_features = data['inv_features']
                features = np.concatenate([o_features[:, None, :],
                                          inv_features[:, None, :]], axis=1)  # (P, 2, 512)
            else:
                features = o_features  # (P, 512)

        P = len(patch_centers)
        if P < self.k:
            return None

        edges = build_knn_graph(patch_centers, k=self.k)


        feat_A, feat_B, labels, center_A, center_B = generate_edge_samples(
            features, edges, gt_flip_status, patch_centers
        )

        return {
            'features_A': torch.from_numpy(feat_A).float(),
            'features_B': torch.from_numpy(feat_B).float(),
            'edge_labels': torch.from_numpy(labels).float(),
            'patch_centers_A': torch.from_numpy(center_A).float(),
            'patch_centers_B': torch.from_numpy(center_B).float(),
            'num_edges': len(feat_A)
        }


def edge_consistency_collate(batch: List[Optional[Dict]]) -> Optional[Dict[str, torch.Tensor]]:
    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        return None

    all_feat_A = torch.cat([b['features_A'] for b in batch], dim=0)
    all_feat_B = torch.cat([b['features_B'] for b in batch], dim=0)
    all_labels = torch.cat([b['edge_labels'] for b in batch], dim=0)
    all_center_A = torch.cat([b['patch_centers_A'] for b in batch], dim=0)
    all_center_B = torch.cat([b['patch_centers_B'] for b in batch], dim=0)

    edge_counts = [b['num_edges'] for b in batch]
    batch_offsets = torch.cumsum(torch.tensor(edge_counts), dim=0)

    return {
        'features_A': all_feat_A,
        'features_B': all_feat_B,
        'edge_labels': all_labels,
        'patch_centers_A': all_center_A,
        'patch_centers_B': all_center_B,
        'batch_offsets': batch_offsets
    }
