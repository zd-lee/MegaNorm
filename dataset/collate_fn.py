"""
Collate Functions for Point Cloud Segmentation
点云分割数据collate函数
"""

import torch
from typing import Dict, List, Tuple, Any
from torch.utils.data.dataloader import default_collate

def collate_point_cloud_batch(point_data_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    将多个点云样本组合成batch

    Args:
        point_data_list: 点云数据列表

    Returns:
        point_batch: batch化的点云数据
    """

    # 计算每个样本的点数
    point_counts = [len(data['coord']) for data in point_data_list]

    # 创建batch相关的索引
    batch_size = len(point_data_list)
    total_points = sum(point_counts)

    # 合并坐标
    coords = torch.cat([data['coord'] for data in point_data_list], dim=0)  # (N_total, 3)

    # 合并特征
    features = torch.cat([data['feat'] for data in point_data_list], dim=0)  # (N_total, C)

    # 创建batch索引
    batch_indices = []
    for i, count in enumerate(point_counts):
        batch_indices.extend([i] * count)
    batch_indices = torch.tensor(batch_indices, dtype=torch.long)

    # 创建batch offset (用于后续处理)
    batch_offsets = torch.cumsum(torch.tensor(point_counts), dim=0).long()

    # Get grid_size from first sample (assuming all samples have same grid_size)
    grid_size = point_data_list[0].get('grid_size') if point_data_list else None

    # 收集文件名信息
    filenames = [data.get('filename', f'unknown_{i}') for i, data in enumerate(point_data_list)]
    ply_paths = [data.get('ply_path', '') for data in point_data_list]

    # 收集归一化参数
    norm_params = [data.get('norm_params') for data in point_data_list if 'norm_params' in data]

    result = {
        'coord': coords,           # (N_total, 3)
        'feat': features,          # (N_total, C)
        'batch': batch_indices,    # (N_total,)
        'batch_offsets': batch_offsets,  # (B,)
        'batch_size': batch_size,
        'total_points': total_points,
        'filenames': filenames,    # List[str] - 每个样本的文件名
        'ply_paths': ply_paths,    # List[str] - 每个样本的完整路径
    }

    # 添加归一化参数（如果存在）
    if norm_params and len(norm_params) == batch_size:
        result['norm_params'] = norm_params  # List[Dict] - 每个样本的归一化参数

    # Add grid_size if it exists
    if grid_size is not None:
        result['grid_size'] = grid_size
        
    return result


def collate_normal_estimation(batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]) \
        -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Collate function for normal estimation (no query points)

    Args:
        batch: List of samples, each containing (point_data, gt_normal)

    Returns:
        point_batch: Batched point cloud data
        gt_normal_batch: Batched ground truth normals (N_total, 3)
    """
    point_data_list = []
    gt_normal_list = []

    for point_data, gt_normal in batch:
        point_data_list.append(point_data)
        gt_normal_list.append(gt_normal)

    # Collate point cloud data
    point_batch = collate_point_cloud_batch(point_data_list)

    # Concatenate ground truth normals
    gt_normal_batch = torch.cat(gt_normal_list, dim=0)  # (N_total, 3)

    return point_batch, gt_normal_batch


class NormalEstimationCollator:
    """
    Collator for normal estimation

    Simpler version without query point handling.
    """

    def __init__(self, max_points_per_sample: int = None):
        """
        Args:
            max_points_per_sample: Maximum points per sample (optional)
        """
        self.max_points_per_sample = max_points_per_sample

    def __call__(self, batch):
        return collate_normal_estimation(batch)



