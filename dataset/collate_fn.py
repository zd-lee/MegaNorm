import torch
from typing import Dict, List, Tuple


def collate_point_cloud_batch(
    point_data_list: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    point_counts = [len(data["coord"]) for data in point_data_list]
    batch_size = len(point_data_list)
    total_points = sum(point_counts)
    coords = torch.cat([data["coord"] for data in point_data_list], dim=0)
    features = torch.cat([data["feat"] for data in point_data_list], dim=0)
    batch_indices = torch.cat(
        [
            torch.full((count,), index, dtype=torch.long)
            for index, count in enumerate(point_counts)
        ]
    )
    batch_offsets = torch.cumsum(torch.tensor(point_counts), dim=0).long()
    result = {
        "coord": coords,
        "feat": features,
        "batch": batch_indices,
        "batch_offsets": batch_offsets,
        "batch_size": batch_size,
        "total_points": total_points,
        "filenames": [
            data.get("filename", f"unknown_{idx}")
            for idx, data in enumerate(point_data_list)
        ],
        "ply_paths": [data.get("ply_path", "") for data in point_data_list],
    }
    grid_size = point_data_list[0].get("grid_size") if point_data_list else None
    if grid_size is not None:
        result["grid_size"] = grid_size
    norm_params = [
        data.get("norm_params") for data in point_data_list if "norm_params" in data
    ]
    if len(norm_params) == batch_size:
        result["norm_params"] = norm_params
    return result


def collate_normal_estimation(
    batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    point_data_list = [sample[0] for sample in batch]
    gt_normal_list = [sample[1] for sample in batch]
    return collate_point_cloud_batch(point_data_list), torch.cat(gt_normal_list, dim=0)


class NormalEstimationCollator:
    def __init__(self, max_points_per_sample: int = None):
        self.max_points_per_sample = max_points_per_sample

    def __call__(self, batch):
        return collate_normal_estimation(batch)
