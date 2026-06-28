import os

import numpy as np
import torch
from torch._refs import flip
from torch.utils.data import Dataset

from typing import Dict, List, Any, Optional, Tuple

import open3d as o3d


def estimate_normals_torch(inputpc, max_nn=10):
    from torch_cluster import knn_graph

    return_numpy = not isinstance(inputpc, torch.Tensor)
    if not isinstance(inputpc, torch.Tensor):
        inputpc = torch.from_numpy(inputpc).float()
    if inputpc.shape[0] == 1:
        result = torch.cat(
            [
                inputpc[:, :3],
                torch.as_tensor([[1, 0, 0]], dtype=inputpc.dtype, device=inputpc.device),
            ],
            dim=-1,
        )
        return result.cpu().numpy() if return_numpy else result
    num_points = inputpc.shape[0]
    if max_nn > inputpc.shape[0] - 1:
        max_nn = min(max_nn, inputpc.shape[0] - 1)
        print(
            f"Warning: max_nn is larger than number of points. Set max_nn to {max_nn}"
        )
    edge_index = knn_graph(inputpc[:, :3], max_nn, loop=False)
    expected_elements = 2 * num_points * max_nn
    actual_elements = edge_index.numel()
    if actual_elements < expected_elements:
        raise RuntimeError(
            f"knn_graph returned insufficient edges: "
            f"expected {expected_elements}, got {actual_elements}"
        )
    elif actual_elements > expected_elements:
        edge_index = edge_index.flatten()[:expected_elements].view(2, -1)
    knn = edge_index.view(2, num_points, max_nn)[0]
    x = inputpc[knn][:, :, :3]
    temp = x[:, :, :3] - x.mean(dim=1)[:, None, :3]
    cov = temp.transpose(1, 2) @ temp / max_nn
    e, v = torch.linalg.eigh(cov, UPLO="U")
    n = v[:, :, 0]
    n_norm = torch.norm(n, dim=1, keepdim=True).clamp(min=1e-8)
    n = n / n_norm
    nan_mask = torch.isnan(n).any(dim=1)
    if nan_mask.any():
        n[nan_mask] = torch.tensor([0.0, 0.0, 1.0], device=n.device)
    result = torch.cat([inputpc[:, :3], n], dim=-1)
    return result.cpu().numpy() if return_numpy else result


class NormalEstimationDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        data_list: Optional[List[str]] = None,
        transform=None,
        grid_size: float = 0.02,
        max_points: int = 50000,
        use_preprocessed: bool = True,
        preprocessed_dir: Optional[str] = None,
    ):
        self.data_root = data_root
        self.transform = transform
        self.grid_size = grid_size
        self.max_points = max_points
        self.use_preprocessed = use_preprocessed
        if preprocessed_dir is None:
            self.preprocessed_dir = data_root + "_preprocessed"
        else:
            self.preprocessed_dir = preprocessed_dir
        if data_list is None:
            self.files = self._scan_data_directory()
        else:
            self.files = data_list
        print(f"NormalEstimationDataset: {len(self.files)} samples")

    def _scan_data_directory(self) -> List[Dict[str, str]]:
        files_list = []
        if self.use_preprocessed and os.path.exists(self.preprocessed_dir):
            for root, _, files in os.walk(self.preprocessed_dir):
                for file in files:
                    if file.lower().endswith(".npz"):
                        full_path = os.path.join(root, file)
                        name = os.path.splitext(file)[0]
                        files_list.append(
                            {
                                "name": name,
                                "npz_path": full_path,
                                "is_preprocessed": True,
                            }
                        )
            print(
                f"Found {len(files_list)} preprocessed NPZ files in {self.preprocessed_dir}"
            )
        else:
            for root, _, files in os.walk(self.data_root):
                for file in files:
                    if file.lower().endswith(".ply"):
                        full_path = os.path.join(root, file)
                        name = os.path.splitext(file)[0]
                        files_list.append(
                            {
                                "name": name,
                                "ply_path": full_path,
                                "is_preprocessed": False,
                            }
                        )
            print(f"Found {len(files_list)} PLY files in {self.data_root}")
            if self.use_preprocessed:
                print(
                    f"Warning: use_preprocessed=True but preprocessed directory not found at {self.preprocessed_dir}"
                )
                print(f"Will load from PLY files.")
        return files_list

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        file_info = self.files[idx]
        filename = file_info["name"]
        is_preprocessed = file_info.get("is_preprocessed", False)
        if is_preprocessed:
            file_path = file_info["npz_path"]
            data = np.load(file_path)
            points = data["points"].astype(np.float32)
            gt_normals = data["gt_normals"].astype(np.float32)
        else:
            file_path = file_info["ply_path"]
            pcd = o3d.io.read_point_cloud(file_path)
            points = np.asarray(pcd.points).astype(np.float32)
            if not pcd.has_normals():
                raise ValueError(f"PLY file {file_path} does not contain normals")
            gt_normals = np.asarray(pcd.normals).astype(np.float32)
        if len(points) > self.max_points:
            indices = np.random.choice(len(points), self.max_points, replace=False)
            points = points[indices]
            gt_normals = gt_normals[indices]
        point_data = {
            "coord": torch.from_numpy(points),
            "feat": torch.from_numpy(points.copy()),
            "batch": torch.zeros(len(points), dtype=torch.long),
            "grid_size": self.grid_size,
            "filename": filename,
            "ply_path": file_path,
        }
        gt_normal = torch.from_numpy(gt_normals)
        if self.transform:
            point_data, gt_normal = self.transform(point_data, gt_normal)
        return point_data, gt_normal


SUPPORTED_FORMATS = {
    "numpy": {
        "extensions": [".npy", ".npz"],
        "structure": {
            "points": "shape=(N,3), dtype=float32, point coordinates",
            "features": "shape=(N,C), dtype=float32, point features (optional)",
            "labels": "shape=(N,), dtype=int, point class labels",
            "query_position": "shape=(3,), dtype=float32, query point coordinates",
            "query_label": "scalar, int, query point class label",
        },
    },
    "point_cloud_library": {
        "extensions": [".pcd", ".ply"],
        "notes": "Requires Open3D. Will use default query point if not specified in file.",
    },
}
