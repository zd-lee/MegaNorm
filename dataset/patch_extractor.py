import torch
import numpy as np
from sklearn.neighbors import KDTree

from collections import deque
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass
import logging

from cpp_alg import extract_patches_bfs_cpu

logger = logging.getLogger(__name__)


@dataclass
class _KDTreeNode:
    indices: Optional[torch.Tensor] = None
    left: Optional["_KDTreeNode"] = None
    right: Optional["_KDTreeNode"] = None


def _build_knn_graph(points: torch.Tensor, k: int) -> List[List[int]]:
    logger.debug(f"Building KNN graph for {points.shape[0]} points with k={k}")
    points_np = points.detach().cpu().numpy()
    tree = KDTree(points_np)
    distances, indices = tree.query(points_np, k=k + 1)
    adjacency_list = indices[:, 1:].tolist()
    logger.debug(
        f"Built KNN graph with adjacency list for {len(adjacency_list)} points"
    )
    return adjacency_list


def _bfs_extract_patch(
    adjacency_list: List[List[int]],
    start_node: int,
    num_points: int,
) -> np.ndarray:
    visited = set()
    queue = deque([start_node])
    visited.add(start_node)
    patch_indices = [start_node]
    while queue and len(patch_indices) < num_points:
        current = queue.popleft()
        for neighbor in adjacency_list[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                patch_indices.append(neighbor)
                if len(patch_indices) >= num_points:
                    break
    if len(patch_indices) < num_points:
        logger.warning(
            f"Patch starting from node {start_node} only found "
            f"{len(set(patch_indices))} unique points, expected {num_points}"
        )
    return np.array(patch_indices[:num_points], dtype=np.int64)


def _build_kdtree_recursive(
    indices: torch.Tensor, points: torch.Tensor, max_points: int
) -> _KDTreeNode:
    n = len(indices)
    if n <= max_points:
        return _KDTreeNode(indices=indices.clone())
    coords = points[indices]
    variance = torch.var(coords, dim=0)
    split_axis = torch.argmax(variance).item()
    axis_values = coords[:, split_axis]
    mid_pos = n // 2
    split_value = torch.kthvalue(axis_values, mid_pos + 1).values
    mask = axis_values <= split_value
    left_indices = indices[mask]
    right_indices = indices[~mask]
    if len(left_indices) == 0 or len(right_indices) == 0:
        return _KDTreeNode(indices=indices.clone())
    left = _build_kdtree_recursive(left_indices, points, max_points)
    right = _build_kdtree_recursive(right_indices, points, max_points)
    return _KDTreeNode(left=left, right=right)


def _collect_leaf_patches(node: _KDTreeNode, patches: List[torch.Tensor]) -> None:
    if node.left is None and node.right is None:
        patches.append(node.indices)
    else:
        if node.left:
            _collect_leaf_patches(node.left, patches)
        if node.right:
            _collect_leaf_patches(node.right, patches)


def extract_patches_bfs(
    points: torch.Tensor,
    query_indices: Optional[Union[np.ndarray, torch.Tensor]],
    k: int,
    num_per_patch: int,
    device: str = "cuda",
    patch_count: Optional[int] = None,
) -> List[torch.Tensor]:
    if query_indices is None:
        from dataset.utils import farthest_point_sampling

        if patch_count is None:
            raise ValueError("patch_count is required when query_indices is None")
        query_indices = farthest_point_sampling(points, patch_count)
        query_indices = query_indices.detach().cpu().numpy()
        logger.info(f"Using FPS to sample {patch_count} query points for BFS")
    if isinstance(query_indices, torch.Tensor):
        query_indices = query_indices.detach().cpu().numpy()
    N = points.shape[0]
    N_q = len(query_indices)
    logger.info(f"Extracting {N_q} patches from {N} points using BFS")
    adjacency_list = _build_knn_graph(points, k)
    patches_indices = np.zeros((N_q, num_per_patch), dtype=np.int64)
    for i, query_idx in enumerate(query_indices):
        if i % 10 == 0:
            logger.debug(f"Extracting patch {i+1}/{N_q}")
        patch = _bfs_extract_patch(
            adjacency_list=adjacency_list,
            start_node=int(query_idx),
            num_points=num_per_patch,
        )
        patches_indices[i] = patch
    patches = [torch.from_numpy(patches_indices[i]).to(device) for i in range(N_q)]
    logger.info(f"Successfully extracted {N_q} patches using BFS")
    return patches


def extract_patches_kdtree(
    points: torch.Tensor, num_per_patch: int
) -> List[torch.Tensor]:
    device = points.device
    all_indices = torch.arange(len(points), dtype=torch.long, device=device)
    root = _build_kdtree_recursive(all_indices, points, num_per_patch)
    patches = []
    _collect_leaf_patches(root, patches)
    logger.info(f"Extracted {len(patches)} patches using KDTree subdivision")
    return patches


def extract_patches_grid(
    points: torch.Tensor, num_per_patch: int, overlap_rate: float = 0.25
) -> List[torch.Tensor]:
    n_points = len(points)
    bbox_min = points.min(dim=0).values
    bbox_max = points.max(dim=0).values
    bbox_size = bbox_max - bbox_min
    volume = (bbox_size[0] * bbox_size[1] * bbox_size[2]).item()
    density = n_points / volume if volume > 0 else 1.0
    window_volume = num_per_patch / density
    window_size = window_volume ** (1 / 3)
    window_cells = 4
    grid_size = window_size / window_cells
    stride_cells = max(1, int(window_cells * (1 - overlap_rate)))
    grid_coords = ((points - bbox_min) / grid_size).long()
    grid_max = grid_coords.max(dim=0).values
    patches = []
    for gx in range(0, grid_max[0].item() + 1, stride_cells):
        for gy in range(0, grid_max[1].item() + 1, stride_cells):
            for gz in range(0, grid_max[2].item() + 1, stride_cells):
                win_min = torch.tensor([gx, gy, gz], device=points.device)
                win_max = win_min + window_cells
                mask = ((grid_coords >= win_min) & (grid_coords < win_max)).all(dim=1)
                indices = mask.nonzero(as_tuple=True)[0]
                if len(indices) > 0:
                    patches.append(indices)
    logger.info(
        f"Extracted {len(patches)} patches using grid method with overlap={overlap_rate}"
    )
    return patches


def extract_patches_fps(
    points: torch.Tensor,
    patch_count: int,
    num_per_patch: int,
    overlap_rate: float = 0.0,
) -> List[torch.Tensor]:
    if overlap_rate <= 0.0:
        overlap_count = 1
    else:
        overlap_count = max(2, round(1.0 / (1.0 - overlap_rate)))
    if overlap_count == 1:
        from dataset.utils import farthest_point_sampling_with_assignments

        _, assignments = farthest_point_sampling_with_assignments(points, patch_count)
        patches = []
        for i in range(patch_count):
            patch_mask = assignments == i
            patch_indices = patch_mask.nonzero(as_tuple=True)[0]
            patches.append(patch_indices)
        return patches, assignments
    else:
        from dataset.utils import farthest_point_sampling_with_assignments_overlap

        _, overlap_list = farthest_point_sampling_with_assignments_overlap(
            points, patch_count, overlap_count=overlap_count
        )
        patches = []
        for center_idx in range(patch_count):
            patch_points_set = set()
            for level_idx in range(overlap_count):
                _, assignments = overlap_list[level_idx]
                mask = assignments == center_idx
                level_indices = mask.nonzero(as_tuple=True)[0]
                patch_points_set.update(level_indices.cpu().numpy())
            patch_indices = torch.tensor(
                sorted(patch_points_set), dtype=torch.long, device=points.device
            )
            patches.append(patch_indices)
        logger.info(
            f"Extracted {patch_count} overlapping patches using FPS "
            f"(overlap_rate={overlap_rate:.2f}, overlap_count={overlap_count})"
        )
        return patches, overlap_list


def extract_patches_knn(
    points: torch.Tensor, patch_count: int, num_per_patch: int, k: Optional[int] = None
) -> List[torch.Tensor]:
    from dataset.utils import farthest_point_sampling

    if k is None:
        k = num_per_patch
    device = points.device
    seed_indices = farthest_point_sampling(points, patch_count)
    points_np = points.detach().cpu().numpy()
    tree = KDTree(points_np)
    seed_points = points_np[seed_indices.cpu().numpy()]
    distances, indices = tree.query(seed_points, k=num_per_patch)
    patches = []
    for i in range(len(seed_indices)):
        patch_indices = torch.from_numpy(indices[i]).to(device)
        patches.append(patch_indices)
    logger.info(
        f"Extracted {len(patches)} patches using optimized KNN (batch query), k={num_per_patch}"
    )
    return patches


def _validate_parameters(
    method: str,
    points: torch.Tensor,
    patch_count: Optional[int],
    num_per_patch: Optional[int],
    overlap_rate: float,
    query_points: Optional[Union[torch.Tensor, np.ndarray]],
):
    if method == "bfs":
        if query_points is None and patch_count is None:
            raise ValueError(
                "BFS requires either query_points or patch_count (for FPS sampling)"
            )
        assert overlap_rate == 0.0, "BFS does not support overlap_rate != 0"
        assert num_per_patch is not None, "BFS requires num_per_patch"
    elif method == "bfs_cpu":
        if query_points is None and patch_count is None:
            raise ValueError(
                "BFS_CPU requires either query_points or patch_count (for FPS sampling)"
            )
        assert num_per_patch is not None, "BFS_CPU requires num_per_patch"
    elif method == "kdtree":
        assert overlap_rate == 0.0, "KDTree does not support overlap_rate != 0"
        assert (
            num_per_patch is not None
        ), "KDTree requires num_per_patch (as max_points)"
        if patch_count is not None:
            logger.warning("patch_count is not controllable for KDTree method")
    elif method == "grid":
        assert 0.0 <= overlap_rate < 1.0, "Grid overlap_rate must be in [0.0, 1.0)"
        assert num_per_patch is not None, "Grid requires num_per_patch"
        if patch_count is not None:
            logger.warning("patch_count is not controllable for Grid method")
    elif method == "fps" or method == "fps_bfs":
        if patch_count is None:
            assert (
                num_per_patch is not None
            ), "FPS requires num_per_patch when patch_count is None"
            patch_count = int(len(points) / num_per_patch)
        assert 0.0 <= overlap_rate < 1.0, "FPS overlap_rate must be in [0.0, 1.0)"
    elif method == "knn":
        assert num_per_patch is not None, "KNN requires num_per_patch (k value)"
        assert (
            patch_count is not None
        ), "KNN requires patch_count (number of seed points)"
    elif method == "fps_cpu":
        assert patch_count is not None, "FPS_CPU requires patch_count"
    else:
        raise ValueError(f"Unknown method: {method}")


def _calculate_metadata(
    patches: List[torch.Tensor], points: torch.Tensor, overlap_rate: float, method: str
) -> dict:
    patch_sizes = [len(p) for p in patches]
    all_points = torch.cat(patches)
    unique_points = torch.unique(all_points)
    computed_overlap_rate = 1.0 - len(unique_points) / len(all_points)
    metadata = {
        "method": method,
        "patch_count": len(patches),
        "points_per_patch": {
            "min": min(patch_sizes),
            "max": max(patch_sizes),
            "mean": sum(patch_sizes) / len(patch_sizes),
            "var": np.var(patch_sizes),
            "all": patch_sizes,
        },
        "overlap_rate": computed_overlap_rate,
        "total_points": len(points),
    }
    return metadata


def _split_oversized_patches(
    patches: List[Union[torch.Tensor, np.ndarray, List[int]]],
    points: torch.Tensor,
    method: str,
    max_points: Optional[int],
    k: int,
    overlap_rate: float,
    device: str,
    max_depth: int = 10,
    depth: int = 0,
) -> List[torch.Tensor]:
    if max_points is None:
        return [torch.as_tensor(p, dtype=torch.long).detach().cpu() for p in patches]
    output = []
    for patch in patches:
        patch_indices = torch.as_tensor(patch, dtype=torch.long).detach().cpu()
        if len(patch_indices) <= max_points:
            output.append(patch_indices)
            continue
        if depth >= max_depth:
            logger.warning(
                "Patch still exceeds max_points after recursive splitting; "
                "falling back to contiguous chunks "
                f"(size={len(patch_indices)}, max_points={max_points}, "
                f"depth={depth}, method={method}, k={k}, "
                f"overlap_rate={overlap_rate}, "
                f"chunks={int(np.ceil(len(patch_indices) / max_points))})"
            )
            for start in range(0, len(patch_indices), max_points):
                chunk = patch_indices[start : start + max_points]
                if len(chunk) > 1:
                    output.append(chunk)
            continue
        sub_patch_count = max(2, int(np.ceil(len(patch_indices) / max_points)))
        sub_points = points[patch_indices.to(points.device)]
        sub_patches, _ = extract_patches_unified(
            sub_points,
            method=method,
            patch_count=sub_patch_count,
            num_per_patch=max_points,
            overlap_rate=overlap_rate,
            k=k,
            device=device,
            cal_meta=False,
            enforce_max_points=False,
        )
        for sub_patch in sub_patches:
            sub_patch_cpu = torch.as_tensor(sub_patch, dtype=torch.long).detach().cpu()
            if len(sub_patch_cpu) > max_points:
                output.extend(
                    _split_oversized_patches(
                        patches=[patch_indices[sub_patch_cpu]],
                        points=points,
                        method=method,
                        max_points=max_points,
                        k=k,
                        overlap_rate=overlap_rate,
                        device=device,
                        max_depth=max_depth,
                        depth=depth + 1,
                    )
                )
            elif len(sub_patch_cpu) > 1 or overlap_rate <= 0.0:
                output.append(patch_indices[sub_patch_cpu])
    return output


def extract_patches_unified(
    points: torch.Tensor,
    method: str,
    patch_count: Optional[int] = None,
    num_per_patch: Optional[int] = None,
    overlap_rate: float = 0.0,
    query_points: Optional[Union[torch.Tensor, np.ndarray]] = None,
    k: int = 10,
    device: str = "cuda",
    cal_meta=False,
    enforce_max_points=True,
    max_split_depth=10,
    **kwargs,
) -> Tuple[List[torch.Tensor], dict]:
    if patch_count is None:
        patch_count = (
            int(len(points) / num_per_patch * (1 + overlap_rate))
            if num_per_patch is not None
            else None
        )
    if num_per_patch is None:
        num_per_patch = (
            int(len(points) / patch_count * (1 + overlap_rate))
            if patch_count is not None
            else None
        )
    _validate_parameters(
        method, points, patch_count, num_per_patch, overlap_rate, query_points
    )
    if method == "fps":
        patches, overlaplist = extract_patches_fps(
            points, patch_count, num_per_patch, overlap_rate
        )
    elif method == "fps_bfs":
        patches, overlaplist = extract_patches_fps(
            points, patch_count, num_per_patch, overlap_rate
        )

        from cpp_alg import split_patches_connected

        no_overlap = overlap_rate <= 0.0
        connected_patches = split_patches_connected(
            points=points,
            patches=patches,
            k=30,
            min_component_size=1 if no_overlap else 10,
        )
        sorted_patches = sorted(connected_patches, key=lambda x: len(x), reverse=False)
        if no_overlap:
            patches = sorted_patches
        else:
            visited_times = torch.full(
                (len(points),), len(overlaplist), dtype=torch.int32
            ).cuda()
            if_filter = torch.zeros(len(sorted_patches), dtype=torch.bool).cuda()
            p = len(connected_patches)
            filtered_count = 0
            for i, patch in enumerate(sorted_patches):
                patch_indices = torch.tensor(patch, dtype=torch.long).cuda()
                if (
                    visited_times[patch_indices].min() > 1
                    and len(patch) < num_per_patch / 4
                ):
                    visited_times[patch_indices] -= 1
                    if_filter[i] = True
                    p -= 1
                    filtered_count += 1
                if p <= patch_count:
                    break
            patches = [
                sorted_patches[i]
                for i in range(len(sorted_patches))
                if not if_filter[i]
            ]
        if enforce_max_points:
            patches = _split_oversized_patches(
                patches=patches,
                points=points,
                method=method,
                max_points=num_per_patch,
                k=k,
                overlap_rate=overlap_rate,
                device=device,
                max_depth=max_split_depth,
            )
        if no_overlap:
            all_indices = torch.cat(
                [torch.as_tensor(p, dtype=torch.long).detach().cpu() for p in patches]
            )
            covered = torch.unique(all_indices).numel()
            if covered != len(points) or all_indices.numel() != len(points):
                logger.warning(
                    "fps_bfs produced incomplete or overlapping no-overlap patches: "
                    f"covered={covered}/{len(points)}, "
                    f"assignments={all_indices.numel()}"
                )
    elif method == "knn":
        patches = extract_patches_knn(points, patch_count, num_per_patch, k)
    elif method == "bfs":
        patches = extract_patches_bfs(
            points, query_points, k, num_per_patch, device, patch_count
        )
    elif method == "bfs_cpu":
        if query_points is None:
            from dataset.utils import farthest_point_sampling

            if patch_count is None:
                raise ValueError("patch_count is required when query_points is None")
            query_points = farthest_point_sampling(points, patch_count)
            logger.info(f"Using FPS to sample {patch_count} query points for BFS_CPU")
        patches_tensor = extract_patches_bfs_cpu(
            points, query_points, k=k, num_per_patch=num_per_patch
        )
        patches = [patches_tensor[i] for i in range(len(patches_tensor))]
    elif method == "fps_cpu":
        from cpp_alg import extract_patches_fps_cpu

        if overlap_rate != 0:
            overlap_count = int(1.0 / overlap_rate)
        else:
            overlap_count = 0
        k_connectivity = k
        if patch_count is None:
            raise ValueError("patch_count is required for fps_cpu method")
        patches = extract_patches_fps_cpu(
            points,
            patch_count=patch_count,
            overlap_count=overlap_count,
            k_connectivity=k_connectivity,
            min_component_size=1,
            out_device=torch.device(device),
        )
    elif method == "kdtree":
        patches = extract_patches_kdtree(points, num_per_patch)
    elif method == "grid":
        patches = extract_patches_grid(points, num_per_patch, overlap_rate)
    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Valid methods: 'fps', 'knn', 'bfs', 'bfs_cpu', 'fps_cpu', 'kdtree', 'grid', 'fps_bfs"
        )
    metadata = _calculate_metadata(patches, points, overlap_rate, method)
    return patches, metadata


def visualize_patch(
    points: np.ndarray,
    patch_indices: np.ndarray,
    query_idx: int,
    output_path: str = None,
):
    try:
        import open3d as o3d

        pcd_all = o3d.geometry.PointCloud()
        pcd_all.points = o3d.utility.Vector3dVector(points)
        pcd_all.paint_uniform_color([0.7, 0.7, 0.7])
        colors = np.asarray(pcd_all.colors)
        colors[patch_indices] = [0.0, 1.0, 0.0]
        colors[query_idx] = [1.0, 0.0, 0.0]
        pcd_all.colors = o3d.utility.Vector3dVector(colors)
        if output_path:
            o3d.io.write_point_cloud(output_path, pcd_all)
            logger.info(f"Saved patch visualization to {output_path}")
        else:
            o3d.visualization.draw_geometries([pcd_all])
    except ImportError:
        logger.warning("Open3D not available, skipping visualization")


def has_edge_between_patches(
    patches: List[torch.Tensor],
    points: torch.Tensor,
) -> torch.Tensor:
    centers = torch.zeros((len(patches), 3), device=points.device)
    for i, patch in enumerate(patches):
        centers[i] = points[patch].mean(dim=0)

    from torch_cluster import knn_graph

    edge_index = knn_graph(centers, k=13, loop=False)
    P = len(patches)
    has_edge = torch.zeros((P, P), dtype=torch.bool, device=points.device)
    for src, dst in edge_index.t():
        has_edge[src, dst] = True
        has_edge[dst, src] = True
    return has_edge


def get_overlap_fast(
    patches: List[torch.Tensor], patch_normals: List[torch.Tensor], num_total: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    P = len(patches)
    device = patches[0].device
    idx1 = torch.full((num_total,), -1, dtype=torch.long, device=device)
    idx2 = torch.full((num_total,), -1, dtype=torch.long, device=device)
    normal1 = torch.zeros((num_total, 3), device=device)
    normal2 = torch.zeros((num_total, 3), device=device)
    for patch_idx, patch_points in enumerate(patches):
        is_first = idx1[patch_points] == -1
        first_points = patch_points[is_first]
        second_points = patch_points[~is_first]
        idx1[first_points] = patch_idx
        idx2[second_points] = patch_idx
        normal1[first_points] = patch_normals[patch_idx][is_first]
        normal2[second_points] = patch_normals[patch_idx][~is_first]
    valid_mask = (idx1 >= 0) & (idx2 >= 0)
    valid_points = valid_mask.nonzero(as_tuple=True)[0]
    if len(valid_points) == 0:
        return torch.zeros(P, P, device=device), torch.zeros(P, P, device=device)
    patch_i = idx1[valid_points]
    patch_j = idx2[valid_points]
    norm_i = normal1[valid_points]
    norm_j = normal2[valid_points]
    dots = (norm_i * norm_j).sum(dim=1)
    A = torch.zeros(P, P, device=device)
    B = torch.zeros(P, P, device=device)
    agree_mask = dots > 0
    disagree_mask = dots < 0
    pairs_agree = torch.stack([patch_i[agree_mask], patch_j[agree_mask]], dim=0)
    A.index_put_(
        (pairs_agree[0], pairs_agree[1]),
        torch.ones(pairs_agree.shape[1], device=device),
        accumulate=True,
    )
    A.index_put_(
        (pairs_agree[1], pairs_agree[0]),
        torch.ones(pairs_agree.shape[1], device=device),
        accumulate=True,
    )
    pairs_disagree = torch.stack(
        [patch_i[disagree_mask], patch_j[disagree_mask]], dim=0
    )
    B.index_put_(
        (pairs_disagree[0], pairs_disagree[1]),
        torch.ones(pairs_disagree.shape[1], device=device),
        accumulate=True,
    )
    B.index_put_(
        (pairs_disagree[1], pairs_disagree[0]),
        torch.ones(pairs_disagree.shape[1], device=device),
        accumulate=True,
    )
    return A, B


def compute_patch_consistency_matrix(
    patches: List[torch.Tensor],
    patch_normals: List[torch.Tensor],
    has_edge=None,
    idx2patch: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    P = len(patches)
    device = patches[0].device
    A = torch.zeros(P, P, device=device)
    B = torch.zeros(P, P, device=device)
    for i in range(P):
        for j in range(i + 1, P):
            if has_edge is not None:
                if not has_edge[i, j]:
                    continue
            mask_ij = torch.isin(patches[i], patches[j])
            mask_ji = torch.isin(patches[j], patches[i])
            overlap_pts = patches[i][mask_ij]
            if len(overlap_pts) == 0:
                continue
            idx_in_i = mask_ij.nonzero(as_tuple=True)[0].to(patch_normals[i].device)
            idx_in_j = mask_ji.nonzero(as_tuple=True)[0].to(patch_normals[j].device)
            dots = (patch_normals[i][idx_in_i] * patch_normals[j][idx_in_j]).sum(dim=1)
            A[i, j] = (dots > 0).sum()
            A[j, i] = A[i, j]
            B[i, j] = (dots < 0).sum()
            B[j, i] = B[i, j]
    return A, B
