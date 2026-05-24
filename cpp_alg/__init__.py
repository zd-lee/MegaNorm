import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.cpp_extension import load

_ext = None


def _load_ext():
    global _ext
    if _ext is not None:
        return _ext

    here = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(here, "build")
    os.makedirs(build_dir, exist_ok=True)

    _ext = load(
        name="cpp_alg_ext",
        sources=[
            os.path.join(here, "patch_bfs.cpp"),
            os.path.join(here, "connected_component.cpp"),
            os.path.join(here, "ply_io.cpp"),
            os.path.join(here, "split_patches.cpp"),
            os.path.join(here, "fps_cpu.cpp")
        ],
        extra_cflags=["-O3", "-std=c++17", "-fopenmp"],
        extra_ldflags=["-fopenmp"],
        extra_include_paths=[
            os.path.join(here, "include", "pico_tree", "src", "pico_tree")
        ],
        with_cuda=False,
        build_directory=build_dir,
        verbose=False,
    )
    return _ext


def extract_patches_bfs_cpu(
    points: torch.Tensor,
    query_indices: Union[np.ndarray, torch.Tensor],
    *,
    k: int = 10,
    num_per_patch: int = 256,
    out_device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    CPU版 extract_patches_bfs: C++ 内部完成 KDTree + KNN + BFS

    Args:
        points: (N, 3) float tensor (CPU/CUDA均可)
        query_indices: (N_q,) query点索引
        k: KNN 参数
        num_per_patch: 每个 patch 的点数
        out_device: 输出设备 (默认与 points.device 一致)

    Returns:
        patches: (N_q, num_per_patch) int64 tensor
    """
    if out_device is None:
        out_device = points.device

    # Convert points to CPU float32
    points_cpu = points.detach()
    if points_cpu.is_cuda:
        points_cpu = points_cpu.cpu()
    if points_cpu.dtype != torch.float32:
        points_cpu = points_cpu.float()
    points_cpu = points_cpu.contiguous()

    # Convert query_indices to CPU int64
    if isinstance(query_indices, np.ndarray):
        q_cpu = torch.from_numpy(query_indices.astype(np.int64, copy=False))
    else:
        q_cpu = query_indices.detach()
        if q_cpu.is_cuda:
            q_cpu = q_cpu.cpu()
        if q_cpu.dtype not in (torch.int32, torch.int64):
            q_cpu = q_cpu.to(torch.int64)
    q_cpu = q_cpu.contiguous()

    # Call C++ implementation
    ext = _load_ext()
    patches = ext.bfs_extract_patches_cpu(
        points_cpu, q_cpu, int(k), int(num_per_patch)
    )

    # Transfer to target device if needed
    if out_device.type != "cpu":
        patches = patches.to(out_device, non_blocking=True)

    return patches


def extract_largest_connected_component(
    points: np.ndarray,
    k: int = 30,
    return_timing: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
    """
    提取点云最大连通分量

    Args:
        points: (N, 3) numpy float32
        k: KNN 邻居数
        return_timing: 是否返回详细计时信息

    Returns:
        mask: (N,) numpy int64，1=保留，0=丢弃
        timing (可选): dict with keys ['kdtree_build', 'knn_query', 'bfs', 'total']
    """
    points = np.ascontiguousarray(points, dtype=np.float32)

    points_t = torch.from_numpy(points)

    ext = _load_ext()
    mask_t, timing_list = ext.extract_largest_component_cpu(points_t, k)

    mask = mask_t.numpy()

    if return_timing:
        timing = {
            'kdtree_build': timing_list[0],
            'knn_query': timing_list[1],
            'bfs': timing_list[2],
            'total': timing_list[3]
        }
        return mask, timing

    return mask


def process_ply_file(
    input_path: str,
    output_path: str,
    k: int = 30,
    return_timing: bool = False
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, float]]]:
    """
    C++ 加速版本：直接处理 PLY 文件（快 10x+）

    完整流程在 C++ 内完成：读取 PLY → 提取连通分量 → 写入 PLY

    Args:
        input_path: 输入 PLY 文件路径
        output_path: 输出 PLY 文件路径
        k: KNN 邻居数
        return_timing: 是否返回详细计时

    Returns:
        stats: {
            'input_points': int,
            'output_points': int,
            'num_components': int,
            'retention_ratio': float,
            'second_largest_points': int
        }
        timing (可选): {
            'read': float,
            'kdtree_build': float,
            'knn_query': float,
            'bfs': float,
            'filter': float,
            'write': float,
            'total': float
        }
    """
    ext = _load_ext()
    stats_tensor, timing_list = ext.process_ply_file_cpp(
        input_path, output_path, k
    )

    # 解析统计信息 tensor
    stats = {
        'input_points': int(stats_tensor[0].item()),
        'output_points': int(stats_tensor[1].item()),
        'num_components': int(stats_tensor[2].item()),
        'retention_ratio': float(stats_tensor[3].item()),
        'second_largest_points': int(stats_tensor[4].item())
    }

    if return_timing:
        timing = {
            'read': timing_list[0],
            'kdtree_build': timing_list[1],
            'knn_query': timing_list[2],
            'bfs': timing_list[3],
            'filter': timing_list[4],
            'write': timing_list[5],
            'total': timing_list[6]
        }
        return stats, timing

    return stats


def _pack_patches(patches: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(patches) == 0:
        return torch.empty((0, 0), dtype=torch.int64), torch.empty(0, dtype=torch.int64)

    P = len(patches)
    M = max(len(p) for p in patches)

    packed = torch.full((P, M), -1, dtype=torch.int64)
    sizes = torch.zeros(P, dtype=torch.int64)

    for i, patch in enumerate(patches):
        n = len(patch)
        packed[i, :n] = patch
        sizes[i] = n

    return packed, sizes


def _unpack_components(
    packed: torch.Tensor,
    sizes: torch.Tensor
) -> List[torch.Tensor]:
    components = []
    for i in range(len(sizes)):
        size = sizes[i].item()
        if size > 0:
            components.append(packed[i, :size].clone())
    return components


def split_patches_connected(
    points: torch.Tensor,
    patches: List[torch.Tensor],
    k: int = 10,
    min_component_size: int = 1,
    out_device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    """
    Split patches into connected components using KNN graph.

    Args:
        points: (N, 3) float tensor (CPU/CUDA)
        patches: List of patch index tensors (variable length)
        k: KNN parameter for connectivity
        min_component_size: Minimum points per component (default=1)
        out_device: Output device (default=points.device)

    Returns:
        List of component index tensors (more components than input patches)
    """
    if out_device is None:
        out_device = points.device

    points_cpu = points.detach().cpu().float().contiguous()
    patches_cpu = [p.detach().cpu().long().contiguous() for p in patches]

    packed_patches, patch_sizes = _pack_patches(patches_cpu)

    ext = _load_ext()
    packed_components, component_sizes = ext.split_patches_connected_cpu(
        points_cpu,
        packed_patches,
        patch_sizes,
        int(k),
        int(min_component_size)
    )

    components = _unpack_components(packed_components, component_sizes)

    if out_device.type != "cpu":
        components = [c.to(out_device, non_blocking=True) for c in components]

    return components


def extract_patches_fps_cpu(
    points: torch.Tensor,
    patch_count: int,
    overlap_count: int = 2,
    k_connectivity: int = 10,
    min_component_size: int = 5,
    out_device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    """
    Extract patches using FPS with overlap and connectivity splitting (Pure CPU).

    Args:
        points: (N, 3) float tensor (CPU/CUDA)
        patch_count: Number of FPS seed points
        overlap_count: Number of nearest centers per point (2-5 recommended)
        k_connectivity: KNN parameter for connectivity check
        min_component_size: Minimum points per component
        out_device: Output device (default=points.device)

    Returns:
        List of connected component patches (torch.Tensor)
    """
    if out_device is None:
        out_device = points.device

    # Transfer to CPU if needed
    points_cpu = points.detach().cpu().float().contiguous()

    # Call C++ implementation
    ext = _load_ext()
    components = ext.extract_patches_fps_cpu(
        points_cpu,
        int(patch_count),
        0,  # num_per_patch unused
        int(overlap_count),
        int(k_connectivity),
        int(min_component_size)
    )

    # Transfer back to target device
    if out_device.type != "cpu":
        components = [c.to(out_device, non_blocking=True) for c in components]

    return components

