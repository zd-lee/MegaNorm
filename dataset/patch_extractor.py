"""
Patch Extractor Module

Implements Steps 2-3 of DACPO algorithm:
- Step 2: Build KNN graph using KDTree
- Step 3: Extract patches using BFS on the KNN graph

Architecture: Pure functions without class dependency
"""

import torch
import numpy as np
from sklearn.neighbors import KDTree
from collections import deque
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass
import logging

# Import CPU BFS from cpp_alg
from cpp_alg import extract_patches_bfs_cpu

logger = logging.getLogger(__name__)


@dataclass
class _KDTreeNode:
    """Internal KDTree node for spatial subdivision."""
    indices: Optional[torch.Tensor] = None  # Only stored in leaf nodes
    left: Optional['_KDTreeNode'] = None
    right: Optional['_KDTreeNode'] = None


# ============================================================================
# Helper Functions
# ============================================================================

def _build_knn_graph(points: torch.Tensor, k: int) -> List[List[int]]:
    """Build KNN graph using KDTree."""
    logger.debug(f"Building KNN graph for {points.shape[0]} points with k={k}")

    # Convert to numpy for KDTree
    points_np = points.detach().cpu().numpy()

    # Build KDTree
    tree = KDTree(points_np)

    # Find k nearest neighbors for each point
    # query returns (distances, indices)
    distances, indices = tree.query(points_np, k=k + 1)  # +1 to include self

    # Build adjacency list (exclude self, only keep k neighbors)
    adjacency_list = indices[:, 1:].tolist()  # Skip first column (self)

    logger.debug(f"Built KNN graph with adjacency list for {len(adjacency_list)} points")
    return adjacency_list


def _bfs_extract_patch(
    adjacency_list: List[List[int]],
    start_node: int,
    num_points: int,
) -> np.ndarray:
    """Extract patch via BFS from start_node."""
    visited = set()
    queue = deque([start_node])
    visited.add(start_node)
    patch_indices = [start_node]

    while queue and len(patch_indices) < num_points:
        current = queue.popleft()

        # Add neighbors to queue
        for neighbor in adjacency_list[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                patch_indices.append(neighbor)

                if len(patch_indices) >= num_points:
                    break

    # If we didn't collect enough points, log warning
    if len(patch_indices) < num_points:
        logger.warning(
            f"Patch starting from node {start_node} only found "
            f"{len(set(patch_indices))} unique points, expected {num_points}"
        )
    return np.array(patch_indices[:num_points], dtype=np.int64)


def _build_kdtree_recursive(
    indices: torch.Tensor,
    points: torch.Tensor,
    max_points: int
) -> _KDTreeNode:
    """Recursively build KDTree with spatial subdivision."""
    n = len(indices)

    # Stop condition: leaf node
    if n <= max_points:
        return _KDTreeNode(indices=indices.clone())

    # Calculate variance and select split axis
    coords = points[indices]
    variance = torch.var(coords, dim=0)
    split_axis = torch.argmax(variance).item()

    # Fast median using kthvalue (O(N))
    axis_values = coords[:, split_axis]
    mid_pos = n // 2
    split_value = torch.kthvalue(axis_values, mid_pos + 1).values

    # Partition indices
    mask = axis_values <= split_value
    left_indices = indices[mask]
    right_indices = indices[~mask]

    # Handle edge case: unbalanced split
    if len(left_indices) == 0 or len(right_indices) == 0:
        return _KDTreeNode(indices=indices.clone())

    # Recurse
    left = _build_kdtree_recursive(left_indices, points, max_points)
    right = _build_kdtree_recursive(right_indices, points, max_points)

    return _KDTreeNode(left=left, right=right)


def _collect_leaf_patches(node: _KDTreeNode, patches: List[torch.Tensor]) -> None:
    """Collect leaf node indices as patches."""
    if node.left is None and node.right is None:  # Leaf node
        patches.append(node.indices)
    else:
        if node.left:
            _collect_leaf_patches(node.left, patches)
        if node.right:
            _collect_leaf_patches(node.right, patches)


# ============================================================================
# Extraction Functions
# ============================================================================

def extract_patches_bfs(
    points: torch.Tensor,
    query_indices: Optional[Union[np.ndarray, torch.Tensor]],
    k: int,
    num_per_patch: int,
    device: str = 'cuda',
    patch_count: Optional[int] = None
) -> List[torch.Tensor]:
    """
    Extract patches for all query points using BFS (GPU version).

    Args:
        points: Input point cloud (N, 3)
        query_indices: Query point indices (N_q,). If None, use FPS to sample query points
        k: Number of neighbors for KNN graph
        num_per_patch: Points per patch
        device: Target device for output
        patch_count: Number of patches (used when query_indices is None)

    Returns:
        List of patches (each patch is a tensor of indices)
    """
    # If query_indices not provided, use FPS to sample
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

    # Build KNN graph
    adjacency_list = _build_knn_graph(points, k)

    # Extract patches using BFS for each query point
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

    # Convert to list of tensors
    patches = [
        torch.from_numpy(patches_indices[i]).to(device)
        for i in range(N_q)
    ]

    logger.info(f"Successfully extracted {N_q} patches using BFS")
    return patches


def extract_patches_kdtree(
    points: torch.Tensor,
    num_per_patch: int
) -> List[torch.Tensor]:
    """
    Extract patches using recursive KDTree subdivision.

    Args:
        points: Input point cloud (N, 3)
        num_per_patch: Maximum points per patch

    Returns:
        List of patches (each patch is a tensor of indices)
    """
    device = points.device
    all_indices = torch.arange(len(points), dtype=torch.long, device=device)

    root = _build_kdtree_recursive(all_indices, points, num_per_patch)

    patches = []
    _collect_leaf_patches(root, patches)

    logger.info(f"Extracted {len(patches)} patches using KDTree subdivision")
    return patches


def extract_patches_grid(
    points: torch.Tensor,
    num_per_patch: int,
    overlap_rate: float = 0.25
) -> List[torch.Tensor]:
    """
    Extract overlapping patches via sliding grid window.

    Args:
        points: Input point cloud (N, 3)
        num_per_patch: Target points per patch
        overlap_rate: Overlap ratio between patches (0.0-1.0)

    Returns:
        List of patches (each patch is a tensor of indices)
    """
    n_points = len(points)

    # Calculate bounding box and point density
    bbox_min = points.min(dim=0).values
    bbox_max = points.max(dim=0).values
    bbox_size = bbox_max - bbox_min
    volume = (bbox_size[0] * bbox_size[1] * bbox_size[2]).item()
    density = n_points / volume if volume > 0 else 1.0

    # Calculate window volume and size based on target points
    window_volume = num_per_patch / density
    window_size = window_volume ** (1/3)

    # Set grid parameters: use 4 cells to cover window
    window_cells = 4
    grid_size = window_size / window_cells

    # Calculate stride based on overlap_rate
    stride_cells = max(1, int(window_cells * (1 - overlap_rate)))

    # Point → grid coordinates
    grid_coords = ((points - bbox_min) / grid_size).long()
    grid_max = grid_coords.max(dim=0).values

    # Sliding window collection
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

    logger.info(f"Extracted {len(patches)} patches using grid method with overlap={overlap_rate}")
    return patches


def extract_patches_fps(
    points: torch.Tensor,
    patch_count: int,
    num_per_patch: int,
    overlap_rate: float = 0.0
) -> List[torch.Tensor]:
    """
    Extract patches using Farthest Point Sampling (optimized).

    Algorithm:
    1. Use FPS to select patch_count seed points while tracking assignments
    2. When overlap_rate=0: Group points by their nearest seed only (no overlap)
    3. When overlap_rate>0: Track top-k nearest seeds per point, each patch includes
       all points for which this seed is among the top-k nearest (overlapping)
    4. Return list of patches with controlled overlap

    Args:
        points: Input point cloud (N, 3)
        patch_count: Number of patches to create
        num_per_patch: Points per patch (not used in FPS, kept for API compatibility)
        overlap_rate: Overlap rate (0.0-1.0), controls patch overlap
                     - overlap_rate=0.0: Non-overlapping patches (each point in 1 patch)
                     - overlap_rate=0.5: Each point in ~2 patches on average
                     - overlap_rate=0.67: Each point in ~3 patches on average
                     Formula: overlap_count = round(1 / (1 - overlap_rate))

    Returns:
        List of patches (each patch is a tensor of indices)
        When overlap_rate > 0, patches will have overlapping points

    Example:
        >>> # Standard non-overlapping patches
        >>> patches = extract_patches_fps(points, patch_count=50, num_per_patch=100)
        >>> # Each point belongs to exactly 1 patch

        >>> # Overlapping patches with overlap_rate=0.5
        >>> patches = extract_patches_fps(points, patch_count=50, num_per_patch=100, overlap_rate=0.5)
        >>> # Each point belongs to ~2 patches on average (nearest and 2nd nearest centers)
    """
    # Calculate overlap_count from overlap_rate
    # overlap_rate = (overlap_count - 1) / overlap_count
    # => overlap_count = 1 / (1 - overlap_rate)
    if overlap_rate <= 0.0:
        overlap_count = 1
    else:
        overlap_count = max(2, round(1.0 / (1.0 - overlap_rate)))

    if overlap_count == 1:
        # Standard non-overlapping FPS
        from dataset.utils import farthest_point_sampling_with_assignments
        _, assignments = farthest_point_sampling_with_assignments(points, patch_count)
        patches = []
        for i in range(patch_count):
            patch_mask = (assignments == i)
            patch_indices = patch_mask.nonzero(as_tuple=True)[0]
            patches.append(patch_indices)
        return patches, assignments

    else:
        # Overlapping FPS: each patch contains points for which this center is in top-k nearest
        from dataset.utils import farthest_point_sampling_with_assignments_overlap
        _, overlap_list = farthest_point_sampling_with_assignments_overlap(
            points, patch_count, overlap_count=overlap_count
        )

        # Build patches: for each center i, collect all points where i is in their top-k nearest
        patches = []
        for center_idx in range(patch_count):
            # Collect points from all overlap levels where this center appears
            patch_points_set = set()

            for level_idx in range(overlap_count):
                _, assignments = overlap_list[level_idx]
                # Find points assigned to this center at this overlap level
                mask = (assignments == center_idx)
                level_indices = mask.nonzero(as_tuple=True)[0]
                patch_points_set.update(level_indices.cpu().numpy())

            # Convert set to sorted tensor
            patch_indices = torch.tensor(sorted(patch_points_set), dtype=torch.long, device=points.device)
            patches.append(patch_indices)

        logger.info(
            f"Extracted {patch_count} overlapping patches using FPS "
            f"(overlap_rate={overlap_rate:.2f}, overlap_count={overlap_count})"
        )
        return patches, overlap_list


def extract_patches_knn(
    points: torch.Tensor,
    patch_count: int,
    num_per_patch: int,
    k: Optional[int] = None
) -> List[torch.Tensor]:
    """
    Extract patches using K-Nearest Neighbors (optimized with batch query).

    Algorithm:
    1. Use FPS to select seed points
    2. Build KNN index
    3. Batch query all seeds at once for k-nearest neighbors
    4. Natural overlap occurs when neighborhoods intersect

    Args:
        points: Input point cloud (N, 3)
        patch_count: Number of seed points
        num_per_patch: K value (neighbors per patch)
        k: Alias for num_per_patch (for compatibility)

    Returns:
        List of patches (each patch is a tensor of indices)
    """
    from dataset.utils import farthest_point_sampling

    if k is None:
        k = num_per_patch

    device = points.device

    # Step 1: FPS to select seed points
    seed_indices = farthest_point_sampling(points, patch_count)

    # Step 2: Build KNN index
    points_np = points.detach().cpu().numpy()
    tree = KDTree(points_np)

    # Step 3: Batch query all seeds at once - returns (P, M) index matrix!
    seed_points = points_np[seed_indices.cpu().numpy()]  # (P, 3)
    distances, indices = tree.query(seed_points, k=num_per_patch)  # indices: (P, M)

    # Step 4: Convert to patches
    patches = []
    for i in range(len(seed_indices)):
        patch_indices = torch.from_numpy(indices[i]).to(device)
        patches.append(patch_indices)

    logger.info(f"Extracted {len(patches)} patches using optimized KNN (batch query), k={num_per_patch}")
    return patches


# ============================================================================
# Parameter Validation
# ============================================================================

def _validate_parameters(
    method: str,
    points: torch.Tensor,
    patch_count: Optional[int],
    num_per_patch: Optional[int],
    overlap_rate: float,
    query_points: Optional[Union[torch.Tensor, np.ndarray]]
):
    """Validate parameters for each method."""

    if method == 'bfs':
        # If query_points not provided, must have patch_count for FPS sampling
        if query_points is None and patch_count is None:
            raise ValueError("BFS requires either query_points or patch_count (for FPS sampling)")
        assert overlap_rate == 0.0, "BFS does not support overlap_rate != 0"
        assert num_per_patch is not None, "BFS requires num_per_patch"

    elif method == 'bfs_cpu':
        # If query_points not provided, must have patch_count for FPS sampling
        if query_points is None and patch_count is None:
            raise ValueError("BFS_CPU requires either query_points or patch_count (for FPS sampling)")
        assert num_per_patch is not None, "BFS_CPU requires num_per_patch"

    elif method == 'kdtree':
        assert overlap_rate == 0.0, "KDTree does not support overlap_rate != 0"
        assert num_per_patch is not None, "KDTree requires num_per_patch (as max_points)"
        if patch_count is not None:
            logger.warning("patch_count is not controllable for KDTree method")

    elif method == 'grid':
        assert 0.0 <= overlap_rate < 1.0, "Grid overlap_rate must be in [0.0, 1.0)"
        assert num_per_patch is not None, "Grid requires num_per_patch"
        if patch_count is not None:
            logger.warning("patch_count is not controllable for Grid method")

    elif method == 'fps' or method == 'fps_bfs':
        if patch_count is None:
            assert num_per_patch is not None, "FPS requires num_per_patch when patch_count is None"
            patch_count = int(len(points) / num_per_patch)
        # FPS now supports overlap_rate
        assert 0.0 <= overlap_rate < 1.0, "FPS overlap_rate must be in [0.0, 1.0)"

    elif method == 'knn':
        assert num_per_patch is not None, "KNN requires num_per_patch (k value)"
        assert patch_count is not None, "KNN requires patch_count (number of seed points)"

    elif method == 'fps_cpu':
        assert patch_count is not None, "FPS_CPU requires patch_count"
        # overlap_count, k_connectivity, min_component_size are optional in kwargs

    else:
        raise ValueError(f"Unknown method: {method}")


def _calculate_metadata(
    patches: List[torch.Tensor],
    points: torch.Tensor,
    overlap_rate: float,
    method: str
) -> dict:
    """Calculate metadata for extracted patches."""
    patch_sizes = [len(p) for p in patches]

    # Count unique points and total points
    all_points = torch.cat(patches)
    unique_points = torch.unique(all_points)
    computed_overlap_rate = 1.0 - len(unique_points) / len(all_points)

    metadata = {
        'method': method,
        'patch_count': len(patches),
        'points_per_patch': {
            'min': min(patch_sizes),
            'max': max(patch_sizes),
            'mean': sum(patch_sizes) / len(patch_sizes),
            'var': np.var(patch_sizes),
            'all': patch_sizes
        },
        'overlap_rate': computed_overlap_rate,
        'total_points': len(points)
    }

    return metadata


# ============================================================================
# Unified Interface
# ============================================================================

def extract_patches_unified(
    points: torch.Tensor,
    method: str,
    patch_count: Optional[int] = None,
    num_per_patch: Optional[int] = None,
    overlap_rate: float = 0.0,
    query_points: Optional[Union[torch.Tensor, np.ndarray]] = None,
    k: int = 10,
    device: str = 'cuda',
    cal_meta = False,
    **kwargs
) -> Tuple[List[torch.Tensor], dict]:
    """
    Unified interface for all patch extraction methods.

    Args:
        points: (N, 3) point cloud tensor
        method: One of ['fps', 'knn', 'bfs', 'bfs_cpu', 'fps_cpu', 'kdtree', 'grid']
        patch_count: Number of patches (required for fps/knn/fps_cpu, optional for bfs/bfs_cpu)
        num_per_patch: Points per patch
        overlap_rate: Overlap rate for grid and fps methods (0.0-1.0)
                     - For 'grid': controls sliding window overlap
                     - For 'fps': controls how many nearest centers each point is assigned to
        query_points: Query indices for BFS methods (if None, use FPS to sample query points)
        k: KNN parameter for bfs/knn methods
        device: 'cuda' or 'cpu'
        **kwargs: Additional method-specific parameters
                 - For 'fps_cpu': overlap_count (default=2), k_connectivity (default=10),
                   min_component_size (default=5)

    Returns:
        patches: List of point index tensors
        metadata: Dict with statistics

    Raises:
        AssertionError: If parameter combination is invalid for the method

    Examples:
        >>> # FPS extraction (non-overlapping)
        >>> patches, meta = extract_patches_unified(
        ...     points, method='fps', patch_count=10, num_per_patch=1000
        ... )

        >>> # FPS extraction (overlapping, overlap_rate=0.5)
        >>> patches, meta = extract_patches_unified(
        ...     points, method='fps', patch_count=10, num_per_patch=1000, overlap_rate=0.5
        ... )

        >>> # FPS CPU extraction (pure CPU with overlap and connectivity splitting)
        >>> patches, meta = extract_patches_unified(
        ...     points, method='fps_cpu', patch_count=10, overlap_count=2,
        ...     k_connectivity=10, min_component_size=5
        ... )

        >>> # KNN extraction
        >>> patches, meta = extract_patches_unified(
        ...     points, method='knn', patch_count=5, num_per_patch=200, k=10
        ... )

        >>> # BFS extraction with query points
        >>> patches, meta = extract_patches_unified(
        ...     points, method='bfs', query_points=query_indices,
        ...     k=10, num_per_patch=256
        ... )

        >>> # BFS extraction with FPS sampling (no query points)
        >>> patches, meta = extract_patches_unified(
        ...     points, method='bfs', patch_count=5,
        ...     k=10, num_per_patch=256
        ... )

        >>> # CPU BFS extraction
        >>> patches, meta = extract_patches_unified(
        ...     points, method='bfs_cpu', query_points=query_indices,
        ...     k=10, num_per_patch=256
        ... )
    """

    # Validate parameters
    if patch_count is None:
        patch_count = int(len(points) / num_per_patch * (1+overlap_rate)) if num_per_patch is not None else None
    if num_per_patch is None:
        num_per_patch = int(len(points) / patch_count * (1+overlap_rate)) if patch_count is not None else None
    
    _validate_parameters(method, points, patch_count, num_per_patch,
                        overlap_rate, query_points)

    # Dispatch to extraction function
    if method == 'fps':
        patches, overlaplist = extract_patches_fps(points, patch_count, num_per_patch, overlap_rate)
    elif method == 'fps_bfs':
        patches, overlaplist = extract_patches_fps(points, patch_count, num_per_patch, overlap_rate)
        from cpp_alg import split_patches_connected
        connected_patches = split_patches_connected(
                    points=points,
                    patches=patches,
                    k=30,
                    min_component_size=10
        )
        sorted_patches = sorted(connected_patches, key=lambda x: len(x), reverse=False)
        visited_times = torch.full((len(points),), len(overlaplist), dtype=torch.int32).cuda()
        if_filter = torch.zeros(len(sorted_patches), dtype=torch.bool).cuda()
        p = len(connected_patches)
        filtered_count = 0
        for i, patch in enumerate(sorted_patches):
            patch_indices = torch.tensor(patch, dtype=torch.long).cuda()
            if visited_times[patch_indices].min() > 1 and len(patch)<num_per_patch/4:
                visited_times[patch_indices] -= 1
                if_filter[i] = True
                p -= 1
                filtered_count += 1
            if p <= patch_count:
                break
        patches = [sorted_patches[i] for i in range(len(sorted_patches)) if not if_filter[i]]


    elif method == 'knn':
        patches = extract_patches_knn(points, patch_count, num_per_patch, k)

    elif method == 'bfs':
        patches = extract_patches_bfs(points, query_points, k, num_per_patch, device, patch_count)

    elif method == 'bfs_cpu':
        # If query_points not provided, use FPS to sample
        if query_points is None:
            from dataset.utils import farthest_point_sampling
            if patch_count is None:
                raise ValueError("patch_count is required when query_points is None")
            query_points = farthest_point_sampling(points, patch_count)
            logger.info(f"Using FPS to sample {patch_count} query points for BFS_CPU")

        # Call C++ CPU implementation
        patches_tensor = extract_patches_bfs_cpu(
            points, query_points, k=k, num_per_patch=num_per_patch
        )
        # Convert from (N_q, num_per_patch) tensor to list of tensors
        patches = [patches_tensor[i] for i in range(len(patches_tensor))]

    elif method == 'fps_cpu':
        # Pure CPU FPS with overlap and connectivity splitting
        from cpp_alg import extract_patches_fps_cpu

        # Extract parameters
        if overlap_rate != 0:
            overlap_count = int(1.0/overlap_rate) 
        else:
            overlap_count = 0
        k_connectivity = k

        if patch_count is None:
            raise ValueError("patch_count is required for fps_cpu method")

        # Call C++ implementation (returns List[Tensor] directly)
        patches = extract_patches_fps_cpu(
            points,
            patch_count=patch_count,
            overlap_count=overlap_count,
            k_connectivity=k_connectivity,
            min_component_size=1,
            out_device=torch.device(device)
        )

    elif method == 'kdtree':
        patches = extract_patches_kdtree(points, num_per_patch)

    elif method == 'grid':
        patches = extract_patches_grid(points, num_per_patch, overlap_rate)

    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Valid methods: 'fps', 'knn', 'bfs', 'bfs_cpu', 'fps_cpu', 'kdtree', 'grid', 'fps_bfs" 
        )

    # Calculate metadata
    metadata = _calculate_metadata(patches, points, overlap_rate, method)

    return patches, metadata

# ============================================================================
# Visualization Utilities
# ============================================================================

def visualize_patch(
    points: np.ndarray,
    patch_indices: np.ndarray,
    query_idx: int,
    output_path: str = None
):
    """
    Visualize a single patch for debugging.

    Args:
        points: All points (N, 3)
        patch_indices: Indices of points in the patch
        query_idx: Index of the query point
        output_path: Path to save visualization (if None, display interactively)
    """
    try:
        import open3d as o3d

        # Create point cloud for all points
        pcd_all = o3d.geometry.PointCloud()
        pcd_all.points = o3d.utility.Vector3dVector(points)
        pcd_all.paint_uniform_color([0.7, 0.7, 0.7])  # Gray

        # Highlight patch points
        colors = np.asarray(pcd_all.colors)
        colors[patch_indices] = [0.0, 1.0, 0.0]  # Green for patch
        colors[query_idx] = [1.0, 0.0, 0.0]  # Red for query point
        pcd_all.colors = o3d.utility.Vector3dVector(colors)

        if output_path:
            o3d.io.write_point_cloud(output_path, pcd_all)
            logger.info(f"Saved patch visualization to {output_path}")
        else:
            o3d.visualization.draw_geometries([pcd_all])

    except ImportError:
        logger.warning("Open3D not available, skipping visualization")


# ============================================================================
# Patch Consistency Utilities
# ============================================================================

def has_edge_between_patches(
    patches: List[torch.Tensor],
    points: torch.Tensor,
) -> torch.Tensor:
    """Compute patch connectivity via KNN graph on patch centers."""
    centers = torch.zeros((len(patches), 3), device=points.device)
    for i, patch in enumerate(patches):
        centers[i] = points[patch].mean(dim=0)
    from torch_cluster import knn_graph
    edge_index = knn_graph(centers, k=13, loop=False)  # (2, E)
    P = len(patches)
    has_edge = torch.zeros((P, P), dtype=torch.bool, device=points.device)
    for src, dst in edge_index.t():
        has_edge[src, dst] = True
        has_edge[dst, src] = True
    return has_edge





def get_overlap_fast(
    patches: List[torch.Tensor],
    patch_normals: List[torch.Tensor],
    num_total: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast computation using point-to-patches mapping.
    Assumes each point belongs to exactly 2 patches.
    """
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
        accumulate=True
    )
    A.index_put_(
        (pairs_agree[1], pairs_agree[0]),
        torch.ones(pairs_agree.shape[1], device=device),
        accumulate=True
    )

    pairs_disagree = torch.stack([patch_i[disagree_mask], patch_j[disagree_mask]], dim=0)
    B.index_put_(
        (pairs_disagree[0], pairs_disagree[1]),
        torch.ones(pairs_disagree.shape[1], device=device),
        accumulate=True
    )
    B.index_put_(
        (pairs_disagree[1], pairs_disagree[0]),
        torch.ones(pairs_disagree.shape[1], device=device),
        accumulate=True
    )

    return A, B


def compute_patch_consistency_matrix(
    patches: List[torch.Tensor],
    patch_normals: List[torch.Tensor],
    has_edge=None,  # [P, P]
    idx2patch: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute patch consistency matrices A and B based on normal agreement in overlapping points."""
    P = len(patches)
    device = patches[0].device

    A = torch.zeros(P, P, device=device)
    B = torch.zeros(P, P, device=device)

    # Iterate over all patch pairs
    for i in range(P):
        for j in range(i + 1, P):
            if has_edge is not None:
                if not has_edge[i, j]:
                    continue
            # Find overlap points
            mask_ij = torch.isin(patches[i], patches[j])
            mask_ji = torch.isin(patches[j], patches[i])
            overlap_pts = patches[i][mask_ij]
            if len(overlap_pts) == 0:
                continue
            # Get normals for overlap points in both patches
            idx_in_i = mask_ij.nonzero(as_tuple=True)[0].to(patch_normals[i].device)
            idx_in_j = mask_ji.nonzero(as_tuple=True)[0].to(patch_normals[j].device)
            dots = (patch_normals[i][idx_in_i] * patch_normals[j][idx_in_j]).sum(dim=1)
            A[i, j] = (dots > 0).sum()
            A[j, i] = A[i, j]
            B[i, j] = (dots < 0).sum()
            B[j, i] = B[i, j]

    return A, B
