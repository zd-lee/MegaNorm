"""
Point Cloud Segmentation Dataset Adaptor
点云分割数据集适配器
"""

import os
import numpy as np
import torch
from torch._refs import flip
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional, Tuple
import open3d as o3d

def estimate_normals_torch(inputpc, max_nn=10):
    from torch_cluster import knn_graph
    if not isinstance(inputpc, torch.Tensor):
        inputpc = torch.from_numpy(inputpc).float()
    if inputpc.shape[0] == 1:
        return torch.cat([inputpc[:,:3],torch.asarray([[1,0,0]])],dim=-1).numpy()

    # Validate max_nn
    num_points = inputpc.shape[0]
    if max_nn > inputpc.shape[0] - 1:
        max_nn = min(max_nn, inputpc.shape[0] - 1)
        print(f"Warning: max_nn is larger than number of points. Set max_nn to {max_nn}")
    # Get KNN edge indices
    edge_index = knn_graph(inputpc[:, :3], max_nn, loop=False)
    # Check edge count before reshape
    expected_elements = 2 * num_points * max_nn
    actual_elements = edge_index.numel()
    if actual_elements < expected_elements:
        raise RuntimeError(
            f"knn_graph returned insufficient edges: "
            f"expected {expected_elements}, got {actual_elements}"
        )
    elif actual_elements > expected_elements:
        edge_index = edge_index.flatten()[:expected_elements].view(2, -1)

    # Safe reshape
    knn = edge_index.view(2, num_points, max_nn)[0]

    x = inputpc[knn][:, :, :3]
    temp = x[:, :, :3] - x.mean(dim=1)[:, None, :3]
    # Fix: divide by number of neighbors (max_nn), not number of points
    cov = temp.transpose(1, 2) @ temp / max_nn
    # e, v = torch.symeig(cov, eigenvectors=True)
    e, v = torch.linalg.eigh(cov, UPLO='U')
    # Extract normal (eigenvector corresponding to smallest eigenvalue)
    n = v[:, :, 0]
    # Normalize to unit length to avoid NaN propagation
    n_norm = torch.norm(n, dim=1, keepdim=True).clamp(min=1e-8)  # Avoid division by zero
    n = n / n_norm
    # Check for NaN and replace with default normal [0, 0, 1]
    nan_mask = torch.isnan(n).any(dim=1)
    if nan_mask.any():
        n[nan_mask] = torch.tensor([0.0, 0.0, 1.0], device=n.device)
    return torch.cat([inputpc[:, :3], n], dim=-1).numpy()

class NormalEstimationDataset(Dataset):
    """
    Normal Estimation Dataset (No Query Points)

    Dataset for normal estimation tasks where:
    - Each PLY file contains xyz + ground truth normals
    - Returns point cloud coordinates and ground truth normals
    - No query points - each sample is a complete point cloud
    """

    def __init__(self,
                 data_root: str,
                 data_list: Optional[List[str]] = None,
                 transform=None,
                 grid_size: float = 0.02,
                 max_points: int = 50000,
                 use_preprocessed: bool = True,
                 preprocessed_dir: Optional[str] = None):
        """
        Initialize Normal Estimation Dataset

        Args:
            data_root: Root directory containing PLY files
            data_list: List of data files (if None, auto-scan directory)
            transform: Data transformation
            grid_size: Grid sampling size
            max_points: Maximum number of points per sample
            use_preprocessed: If True, use preprocessed .npz files (much faster)
            preprocessed_dir: Directory containing .npz files (if None, uses data_root + '_preprocessed')
        """
        self.data_root = data_root
        self.transform = transform
        self.grid_size = grid_size
        self.max_points = max_points
        self.use_preprocessed = use_preprocessed

        # Set preprocessed directory
        if preprocessed_dir is None:
            self.preprocessed_dir = data_root + '_preprocessed'
        else:
            self.preprocessed_dir = preprocessed_dir

        # Scan for PLY or NPZ files
        if data_list is None:
            self.files = self._scan_data_directory()
        else:
            self.files = data_list

        print(f"NormalEstimationDataset: {len(self.files)} samples")

    def _scan_data_directory(self) -> List[Dict[str, str]]:
        """Scan directory for PLY or NPZ files"""
        files_list = []

        if self.use_preprocessed and os.path.exists(self.preprocessed_dir):
            # Scan for preprocessed .npz files
            for root, _, files in os.walk(self.preprocessed_dir):
                for file in files:
                    if file.lower().endswith('.npz'):
                        full_path = os.path.join(root, file)
                        name = os.path.splitext(file)[0]
                        files_list.append({
                            'name': name,
                            'npz_path': full_path,
                            'is_preprocessed': True
                        })
            print(f"Found {len(files_list)} preprocessed NPZ files in {self.preprocessed_dir}")
        else:
            # Scan for PLY files
            for root, _, files in os.walk(self.data_root):
                for file in files:
                    if file.lower().endswith('.ply'):
                        full_path = os.path.join(root, file)
                        name = os.path.splitext(file)[0]
                        files_list.append({
                            'name': name,
                            'ply_path': full_path,
                            'is_preprocessed': False
                        })
            print(f"Found {len(files_list)} PLY files in {self.data_root}")

            if self.use_preprocessed:
                print(f"Warning: use_preprocessed=True but preprocessed directory not found at {self.preprocessed_dir}")
                print(f"Will load from PLY files.")

        return files_list

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get a single data sample

        Returns:
            point_data: Point cloud data dict
                - 'coord': (N, 3) xyz coordinates
                - 'feat': (N, 3) xyz coordinates as features
                - 'batch': (N,) batch indices
                - 'grid_size': float
                - 'filename': str
                - 'ply_path': str
            gt_normal: (N, 3) ground truth normals
        """
        file_info = self.files[idx]
        filename = file_info['name']
        is_preprocessed = file_info.get('is_preprocessed', False)

        # Load data from either .npz (fast) or .ply (slow)
        if is_preprocessed:
            # Load from preprocessed .npz file
            file_path = file_info['npz_path']
            data = np.load(file_path)
            points = data['points'].astype(np.float32)
            gt_normals = data['gt_normals'].astype(np.float32)
        else:
            # Load from PLY file
            file_path = file_info['ply_path']
            pcd = o3d.io.read_point_cloud(file_path)
            points = np.asarray(pcd.points).astype(np.float32)

            # Get ground truth normals
            if not pcd.has_normals():
                raise ValueError(f"PLY file {file_path} does not contain normals")
            gt_normals = np.asarray(pcd.normals).astype(np.float32)

        # Limit point cloud size if needed
        if len(points) > self.max_points:
            indices = np.random.choice(len(points), self.max_points, replace=False)
            points = points[indices]
            gt_normals = gt_normals[indices]

        # Build point_data dict
        # Features are just xyz coordinates (model will learn to estimate normals)
        point_data = {
            'coord': torch.from_numpy(points),  # (N, 3)
            'feat': torch.from_numpy(points.copy()),  # (N, 3) - xyz as features
            'batch': torch.zeros(len(points), dtype=torch.long),
            'grid_size': self.grid_size,
            'filename': filename,
            'ply_path': file_path,
        }

        # Ground truth normals
        gt_normal = torch.from_numpy(gt_normals)  # (N, 3)

        # Apply transforms
        if self.transform:
            point_data, gt_normal = self.transform(point_data, gt_normal)

        return point_data, gt_normal

# from torch_geometric.datasets import PCPNetDataset



# 数据格式支持说明
SUPPORTED_FORMATS = {
    'numpy': {
        'extensions': ['.npy', '.npz'],
        'structure': {
            'points': 'shape=(N,3), dtype=float32, point coordinates',
            'features': 'shape=(N,C), dtype=float32, point features (optional)',
            'labels': 'shape=(N,), dtype=int, point class labels',
            'query_position': 'shape=(3,), dtype=float32, query point coordinates',
            'query_label': 'scalar, int, query point class label'
        }
    },
    'point_cloud_library': {
        'extensions': ['.pcd', '.ply'],
        'notes': 'Requires Open3D. Will use default query point if not specified in file.'
    }
}