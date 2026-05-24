"""
Utilities Module

Provides I/O and visualization functions for DACPO.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional
import open3d as o3d

logger = logging.getLogger(__name__)



def save_point_cloud(
    file_path: str,
    points: np.ndarray,
    normals: np.ndarray,
    colors: Optional[np.ndarray] = None
):
    """
    Save a point cloud to file.

    Args:
        file_path: Output file path (.ply, .pcd, .xyz, .npy)
        points: Point coordinates (N, 3)
        normals: Point normals (N, 3)
        colors: Point colors (N, 3), optional
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving point cloud to {file_path}")

    if file_path.suffix == '.npy':
        # Save as numpy array (N x 6)
        data = np.concatenate([points, normals], axis=1)
        np.save(file_path, data)

    else:
        # Save using Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)

        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.io.write_point_cloud(str(file_path), pcd)

    logger.info(f"Saved point cloud with {len(points)} points")


def visualize_point_cloud(
    points: np.ndarray,
    normals: np.ndarray,
    title: str = "Point Cloud",
    colors: Optional[np.ndarray] = None,
    show_normals: bool = True,
    normal_length: float = 0.05
):
    """
    Visualize a point cloud interactively.

    Args:
        points: Point coordinates (N, 3)
        normals: Point normals (N, 3)
        title: Window title
        colors: Point colors (N, 3), optional
        show_normals: Whether to show normal vectors
        normal_length: Length of normal arrows
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Default color based on normals (for visualization)
        colors = (normals + 1.0) / 2.0  # Map [-1, 1] to [0, 1]
        pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries = [pcd]

    # Add normal arrows if requested
    if show_normals:
        # Sample points to avoid too many arrows
        num_samples = min(1000, len(points))
        sample_indices = np.random.choice(len(points), num_samples, replace=False)

        for idx in sample_indices:
            arrow = create_arrow(
                points[idx],
                points[idx] + normals[idx] * normal_length,
                color=[1, 0, 0]
            )
            geometries.append(arrow)

    o3d.visualization.draw_geometries(
        geometries,
        window_name=title,
        width=1024,
        height=768
    )


def visualize_before_after(
    points: np.ndarray,
    normals_before: np.ndarray,
    normals_after: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Visualize normals before and after DACPO.

    Args:
        points: Point coordinates (N, 3)
        normals_before: Normals before orientation (N, 3)
        normals_after: Normals after orientation (N, 3)
        save_path: Path to save visualization image
    """
    # Create two point clouds
    pcd_before = o3d.geometry.PointCloud()
    pcd_before.points = o3d.utility.Vector3dVector(points)
    pcd_before.normals = o3d.utility.Vector3dVector(normals_before)
    pcd_before.paint_uniform_color([0.5, 0.5, 1.0])  # Blue

    pcd_after = o3d.geometry.PointCloud()
    pcd_after.points = o3d.utility.Vector3dVector(points + np.array([1.0, 0, 0]))  # Offset
    pcd_after.normals = o3d.utility.Vector3dVector(normals_after)
    pcd_after.paint_uniform_color([1.0, 0.5, 0.5])  # Red

    # Visualize side by side
    o3d.visualization.draw_geometries(
        [pcd_before, pcd_after],
        window_name="Before (Blue) vs After (Red)",
        width=1600,
        height=768
    )


def create_arrow(start: np.ndarray, end: np.ndarray, color: list = [1, 0, 0]):
    """
    Create an arrow mesh for visualization.

    Args:
        start: Starting point (3,)
        end: End point (3,)
        color: Arrow color [r, g, b]

    Returns:
        arrow: Open3D mesh
    """
    direction = end - start
    length = np.linalg.norm(direction)

    if length < 1e-6:
        # Create a small sphere instead
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
        sphere.translate(start)
        sphere.paint_uniform_color(color)
        return sphere

    direction = direction / length

    # Create arrow (cylinder + cone)
    cylinder_height = length * 0.7
    cone_height = length * 0.3
    cylinder_radius = length * 0.02
    cone_radius = length * 0.04

    # Cylinder
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=cylinder_radius,
        height=cylinder_height
    )

    # Cone
    cone = o3d.geometry.TriangleMesh.create_cone(
        radius=cone_radius,
        height=cone_height
    )

    # Translate cone to top of cylinder
    cone.translate([0, 0, cylinder_height])

    # Combine
    arrow = cylinder + cone
    arrow.paint_uniform_color(color)

    # Rotate to align with direction
    # Default arrow points along z-axis
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, direction)
    rotation_axis_norm = np.linalg.norm(rotation_axis)

    if rotation_axis_norm > 1e-6:
        rotation_axis = rotation_axis / rotation_axis_norm
        rotation_angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))

        # Create rotation matrix
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        R = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * (K @ K)

        arrow.rotate(R, center=[0, 0, 0])

    # Translate to start position
    arrow.translate(start)

    return arrow

import torch
def sample_query_points(
    points: torch.Tensor,
    num_queries: int,
    method: str = 'fps'
) -> torch.Tensor:
    """
    Sample query points from the point cloud.

    Args:
        points: Point coordinates (N, 3)
        num_queries: Number of query points to sample
        method: Sampling method ('fps' or 'random')

    Returns:
        query_indices: Indices of query points (num_queries,)
    """
    N = len(points)

    if num_queries >= N:
        logger.warning(f"num_queries ({num_queries}) >= N ({N}), returning all points")
        return torch.arange(N)

    if method == 'random':
        query_indices = torch.randperm(N)[:num_queries]

    elif method == 'fps':
        # Farthest Point Sampling
        query_indices = farthest_point_sampling(points, num_queries)

    else:
        raise ValueError(f"Unknown sampling method: {method}")

    logger.info(f"Sampled {num_queries} query points using {method}")

    return query_indices


def farthest_point_sampling(points: torch.Tensor, num_samples: int, start_idx: Optional[int] = None) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS) to sample representative points.

    Args:
        points: Point coordinates (N, 3)
        num_samples: Number of samples to select
        start_idx: Optional starting point index. If None, use random point.

    Returns:
        indices: Indices of sampled points (num_samples,)
    """
    N = len(points)
    indices = torch.zeros(num_samples, dtype=torch.long).to(points.device)

    # Start with specified or random point
    if start_idx is None:
        current_idx = torch.randint(0, N, (1,)).item()
    else:
        current_idx = start_idx
    indices[0] = current_idx

    # Track minimum distances to sampled points
    min_distances = torch.full((N,), float('inf')).to(points.device)

    for i in range(1, num_samples):
        # Update distances
        current_point = points[current_idx]
        distances = torch.norm(points - current_point, dim=1)
        min_distances = torch.minimum(min_distances, distances)

        # Select farthest point
        current_idx = torch.argmax(min_distances).item()
        indices[i] = current_idx

    return indices

def farthest_point_sampling_with_assignments_overlap(
    points: torch.Tensor,
    num_samples: int,
    start_idx: Optional[int] = None,
    overlap_count: int = 2
) -> Tuple[torch.Tensor, list[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Farthest Point Sampling with overlap tracking.

    Returns sampled indices and for each point, tracks the k-nearest sampled centers.

    Args:
        points: Point coordinates (N, 3)
        num_samples: Number of samples to select
        start_idx: Optional starting point index. If None, use random point.
        overlap_count: Number of nearest centers to track per point (default: 2)

    Returns:
        indices: Indices of sampled points (num_samples,)
        overlap_list: List of (distances, assignments) tuples for each overlap level.
                     overlap_list[0] = (nearest distances, nearest assignments)
                     overlap_list[1] = (2nd nearest distances, 2nd nearest assignments)
                     overlap_list[k] = (k-th nearest distances, k-th nearest assignments)

    Example:
        >>> indices, overlap_list = farthest_point_sampling_with_assignments_overlap(points, 100, overlap_count=3)
        >>> nearest_dists, nearest_idx = overlap_list[0]  # Closest FPS center for each point
        >>> second_dists, second_idx = overlap_list[1]    # 2nd closest FPS center
        >>> third_dists, third_idx = overlap_list[2]      # 3rd closest FPS center
    """
    N = len(points)
    device = points.device

    # Initialize
    indices = torch.zeros(num_samples, dtype=torch.long, device=device)
    if start_idx is None:
        current_idx = torch.randint(0, N, (1,)).item()
    else:
        current_idx = start_idx
    indices[0] = current_idx

    # Track top-k distances and their corresponding center indices
    # Shape: (N, overlap_count)
    distances = torch.full((N, overlap_count), float('inf'), device=device)
    assignments = torch.zeros((N, overlap_count), dtype=torch.long, device=device)

    for i in range(num_samples):
        # Calculate distances to current sampled point
        current_point = points[current_idx]
        current_distances = torch.norm(points - current_point, dim=1)  # (N,)

        # For each point, check if current distance should be inserted into top-k nearest
        # Concatenate current distances as a new column
        all_distances = torch.cat([distances, current_distances.unsqueeze(1)], dim=1)  # (N, overlap_count+1)
        all_assignments = torch.cat([
            assignments,
            torch.full((N, 1), i, dtype=torch.long, device=device)
        ], dim=1)  # (N, overlap_count+1)

        # Sort and keep top-k smallest distances
        sorted_distances, sort_indices = torch.sort(all_distances, dim=1)
        distances = sorted_distances[:, :overlap_count]

        # Gather corresponding assignments using sorted indices
        assignments = torch.gather(all_assignments, dim=1, index=sort_indices[:, :overlap_count])

        # Select next farthest point (based on nearest distance to any sampled point)
        if i < num_samples - 1:
            min_distances = distances[:, 0]  # Nearest distance for each point
            current_idx = torch.argmax(min_distances).item()
            indices[i + 1] = current_idx

    # Prepare output: list of (distance, assignment) tuples for each overlap level
    overlap_list = []
    for k in range(overlap_count):
        overlap_list.append((distances[:, k], assignments[:, k]))

    return indices, overlap_list


def farthest_point_sampling_with_assignments(
    points: torch.Tensor,
    num_samples: int,
    start_idx: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Farthest Point Sampling that returns both sampled indices and assignments.

    This function performs FPS while tracking which seed point each point is closest to.
    This is useful for patch extraction where we need to group points by their nearest seed.

    Args:
        points: Point coordinates (N, 3)
        num_samples: Number of samples to select
        start_idx: Optional starting point index. If None, use random point.

    Returns:
        indices: Indices of sampled points (num_samples,)
        assignments: Closest seed index for each point (N,)
    """
    N = len(points)
    device = points.device

    # Initialize
    indices = torch.zeros(num_samples, dtype=torch.long, device=device)
    if start_idx is None:
        current_idx = torch.randint(0, N, (1,)).item()
    else:
        current_idx = start_idx
    indices[0] = current_idx

    # Track minimum distances and assignments
    min_distances = torch.full((N,), float('inf'), device=device)
    assignments = torch.zeros(N, dtype=torch.long, device=device)

    for i in range(num_samples):
        # Calculate distances to current point
        current_point = points[current_idx]
        distances = torch.norm(points - current_point, dim=1)

        # Update minimum distances and assignments
        mask = distances < min_distances
        min_distances[mask] = distances[mask]
        assignments[mask] = i  # Record which seed is closest

        # Select next farthest point
        if i < num_samples - 1:
            current_idx = torch.argmax(min_distances).item()
            indices[i + 1] = current_idx

    return indices, assignments

