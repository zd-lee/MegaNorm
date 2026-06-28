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
    colors: Optional[np.ndarray] = None,
):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving point cloud to {file_path}")
    if file_path.suffix == ".npy":
        data = np.concatenate([points, normals], axis=1)
        np.save(file_path, data)
    else:
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
    normal_length: float = 0.05,
):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        colors = (normals + 1.0) / 2.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    geometries = [pcd]
    if show_normals:
        num_samples = min(1000, len(points))
        sample_indices = np.random.choice(len(points), num_samples, replace=False)
        for idx in sample_indices:
            arrow = create_arrow(
                points[idx], points[idx] + normals[idx] * normal_length, color=[1, 0, 0]
            )
            geometries.append(arrow)
    o3d.visualization.draw_geometries(
        geometries, window_name=title, width=1024, height=768
    )


def visualize_before_after(
    points: np.ndarray,
    normals_before: np.ndarray,
    normals_after: np.ndarray,
    save_path: Optional[str] = None,
):
    pcd_before = o3d.geometry.PointCloud()
    pcd_before.points = o3d.utility.Vector3dVector(points)
    pcd_before.normals = o3d.utility.Vector3dVector(normals_before)
    pcd_before.paint_uniform_color([0.5, 0.5, 1.0])
    pcd_after = o3d.geometry.PointCloud()
    pcd_after.points = o3d.utility.Vector3dVector(points + np.array([1.0, 0, 0]))
    pcd_after.normals = o3d.utility.Vector3dVector(normals_after)
    pcd_after.paint_uniform_color([1.0, 0.5, 0.5])
    o3d.visualization.draw_geometries(
        [pcd_before, pcd_after],
        window_name="Before (Blue) vs After (Red)",
        width=1600,
        height=768,
    )


def create_arrow(start: np.ndarray, end: np.ndarray, color: list = [1, 0, 0]):
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-6:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
        sphere.translate(start)
        sphere.paint_uniform_color(color)
        return sphere
    direction = direction / length
    cylinder_height = length * 0.7
    cone_height = length * 0.3
    cylinder_radius = length * 0.02
    cone_radius = length * 0.04
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=cylinder_radius, height=cylinder_height
    )
    cone = o3d.geometry.TriangleMesh.create_cone(radius=cone_radius, height=cone_height)
    cone.translate([0, 0, cylinder_height])
    arrow = cylinder + cone
    arrow.paint_uniform_color(color)
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, direction)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    if rotation_axis_norm > 1e-6:
        rotation_axis = rotation_axis / rotation_axis_norm
        rotation_angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
        K = np.array(
            [
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0],
            ]
        )
        R = (
            np.eye(3)
            + np.sin(rotation_angle) * K
            + (1 - np.cos(rotation_angle)) * (K @ K)
        )
        arrow.rotate(R, center=[0, 0, 0])
    arrow.translate(start)
    return arrow

import torch


def sample_query_points(
    points: torch.Tensor, num_queries: int, method: str = "fps"
) -> torch.Tensor:
    N = len(points)
    if num_queries >= N:
        logger.warning(f"num_queries ({num_queries}) >= N ({N}), returning all points")
        return torch.arange(N)
    if method == "random":
        query_indices = torch.randperm(N)[:num_queries]
    elif method == "fps":
        query_indices = farthest_point_sampling(points, num_queries)
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    logger.info(f"Sampled {num_queries} query points using {method}")
    return query_indices


def farthest_point_sampling(
    points: torch.Tensor, num_samples: int, start_idx: Optional[int] = None
) -> torch.Tensor:
    N = len(points)
    indices = torch.zeros(num_samples, dtype=torch.long).to(points.device)
    if start_idx is None:
        current_idx = torch.randint(0, N, (1,)).item()
    else:
        current_idx = start_idx
    indices[0] = current_idx
    min_distances = torch.full((N,), float("inf")).to(points.device)
    for i in range(1, num_samples):
        current_point = points[current_idx]
        distances = torch.norm(points - current_point, dim=1)
        min_distances = torch.minimum(min_distances, distances)
        current_idx = torch.argmax(min_distances).item()
        indices[i] = current_idx
    return indices


def farthest_point_sampling_with_assignments_overlap(
    points: torch.Tensor,
    num_samples: int,
    start_idx: Optional[int] = None,
    overlap_count: int = 2,
) -> Tuple[torch.Tensor, list[Tuple[torch.Tensor, torch.Tensor]]]:
    N = len(points)
    device = points.device
    indices = torch.zeros(num_samples, dtype=torch.long, device=device)
    if start_idx is None:
        current_idx = torch.randint(0, N, (1,)).item()
    else:
        current_idx = start_idx
    indices[0] = current_idx
    distances = torch.full((N, overlap_count), float("inf"), device=device)
    assignments = torch.zeros((N, overlap_count), dtype=torch.long, device=device)
    for i in range(num_samples):
        current_point = points[current_idx]
        current_distances = torch.norm(points - current_point, dim=1)
        all_distances = torch.cat([distances, current_distances.unsqueeze(1)], dim=1)
        all_assignments = torch.cat(
            [assignments, torch.full((N, 1), i, dtype=torch.long, device=device)], dim=1
        )
        sorted_distances, sort_indices = torch.sort(all_distances, dim=1)
        distances = sorted_distances[:, :overlap_count]
        assignments = torch.gather(
            all_assignments, dim=1, index=sort_indices[:, :overlap_count]
        )
        if i < num_samples - 1:
            min_distances = distances[:, 0]
            current_idx = torch.argmax(min_distances).item()
            indices[i + 1] = current_idx
    overlap_list = []
    for k in range(overlap_count):
        overlap_list.append((distances[:, k], assignments[:, k]))
    return indices, overlap_list


def farthest_point_sampling_with_assignments(
    points: torch.Tensor, num_samples: int, start_idx: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    N = len(points)
    device = points.device
    indices = torch.zeros(num_samples, dtype=torch.long, device=device)
    if start_idx is None:
        current_idx = torch.randint(0, N, (1,)).item()
    else:
        current_idx = start_idx
    indices[0] = current_idx
    min_distances = torch.full((N,), float("inf"), device=device)
    assignments = torch.zeros(N, dtype=torch.long, device=device)
    for i in range(num_samples):
        current_point = points[current_idx]
        distances = torch.norm(points - current_point, dim=1)
        mask = distances < min_distances
        min_distances[mask] = distances[mask]
        assignments[mask] = i
        if i < num_samples - 1:
            current_idx = torch.argmax(min_distances).item()
            indices[i + 1] = current_idx
    return indices, assignments
