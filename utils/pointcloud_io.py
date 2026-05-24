"""
Point Cloud I/O Utilities for Segmentation Results
Provides functions to save point clouds with segmentation masks as colored PLY files.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any
import struct

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    pass  # Use custom PLY writer silently


def get_segmentation_colors(num_classes: int = 20) -> np.ndarray:
    """
    Generate distinct colors for segmentation classes.

    Args:
        num_classes: Number of segmentation classes

    Returns:
        colors: (num_classes, 3) RGB colors in [0, 255] range
    """
    colors = np.array([
        [128, 128, 128],  # 0: Background (gray)
        [255, 0, 0],      # 1: Class 1 (red)
        [0, 255, 0],      # 2: Class 2 (green)
        [0, 0, 255],      # 3: Class 3 (blue)
        [255, 255, 0],    # 4: Class 4 (yellow)
        [255, 0, 255],    # 5: Class 5 (magenta)
        [0, 255, 255],    # 6: Class 6 (cyan)
        [255, 128, 0],    # 7: Class 7 (orange)
        [128, 255, 0],    # 8: Class 8 (lime)
        [255, 0, 128],    # 9: Class 9 (pink)
        [128, 0, 255],    # 10: Class 10 (purple)
        [0, 128, 255],    # 11: Class 11 (sky blue)
        [255, 255, 128],  # 12: Class 12 (light yellow)
        [128, 255, 255],  # 13: Class 13 (light cyan)
        [255, 128, 255],  # 14: Class 14 (light magenta)
        [128, 128, 255],  # 15: Class 15 (light blue)
        [255, 128, 128],  # 16: Class 16 (light red)
        [128, 255, 128],  # 17: Class 17 (light green)
        [64, 64, 64],     # 18: Class 18 (dark gray)
        [192, 192, 192],  # 19: Class 19 (light gray)
    ])

    # Extend colors if more classes needed
    if num_classes > len(colors):
        # Generate additional random colors
        additional_colors = np.random.randint(0, 256, (num_classes - len(colors), 3))
        colors = np.vstack([colors, additional_colors])

    return colors[:num_classes]


def apply_segmentation_colors(points: np.ndarray, labels: np.ndarray,
                            num_classes: int = 2) -> np.ndarray:
    """
    Apply segmentation colors to points based on labels.

    Args:
        points: (N, 3) point coordinates
        labels: (N,) segmentation labels
        num_classes: Number of classes

    Returns:
        colors: (N, 3) RGB colors in [0, 255] range
    """
    colors_palette = get_segmentation_colors(num_classes)

    # Ensure labels are within valid range
    labels = np.clip(labels, 0, num_classes - 1)

    # Apply colors
    colors = colors_palette[labels.astype(int)]

    return colors


def highlight_query_point(points: np.ndarray, colors: np.ndarray,
                         query_point: np.ndarray, radius: float = 0.05,
                         highlight_color: Tuple[int, int, int] = (255, 215, 0)) -> np.ndarray:
    """
    Highlight query point and nearby points with a special color.

    Args:
        points: (N, 3) point coordinates
        colors: (N, 3) current point colors
        query_point: (3,) query point location
        radius: Radius around query point to highlight
        highlight_color: RGB color for highlighting (default: gold)

    Returns:
        colors: (N, 3) updated colors with query point highlighted
    """
    if len(query_point.shape) == 0 or query_point.size == 0:
        return colors

    # Ensure query_point is the right shape
    if query_point.shape[-1] != 3:
        return colors

    query_point = query_point.reshape(-1, 3)

    updated_colors = colors.copy()

    for qp in query_point:
        # Calculate distances to query point
        distances = np.linalg.norm(points - qp, axis=1)

        # Highlight points within radius
        nearby_mask = distances <= radius
        updated_colors[nearby_mask] = highlight_color

    return updated_colors


def write_ply_ascii(filename: str, points: np.ndarray, colors: Optional[np.ndarray] = None,
                   normals: Optional[np.ndarray] = None):
    """
    Write PLY file in ASCII format.

    Args:
        filename: Output PLY file path
        points: (N, 3) point coordinates
        colors: (N, 3) RGB colors in [0, 255] range (optional)
        normals: (N, 3) normal vectors (optional)
    """
    N = points.shape[0]

    # Write header
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        if normals is not None:
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")

        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")

        f.write("end_header\n")

        # Write vertices
        for i in range(N):
            line = f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}"

            if normals is not None:
                line += f" {normals[i, 0]:.6f} {normals[i, 1]:.6f} {normals[i, 2]:.6f}"

            if colors is not None:
                line += f" {int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}"

            line += "\n"
            f.write(line)


def save_colored_pointcloud(points: np.ndarray, colors: np.ndarray,
                           filename: str, normals: Optional[np.ndarray] = None):
    """
    Save colored point cloud to PLY file.

    Args:
        points: (N, 3) point coordinates
        colors: (N, 3) RGB colors in [0, 255] range
        filename: Output PLY file path
        normals: (N, 3) normal vectors (optional)
    """
    filename = str(filename)

    if OPEN3D_AVAILABLE:
        # Use Open3D if available
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))

        o3d.io.write_point_cloud(filename, pcd)
    else:
        # Use custom PLY writer
        write_ply_ascii(filename, points, colors, normals)


def save_segmentation_results(points: np.ndarray, predictions: np.ndarray,
                            ground_truth: np.ndarray, query_point: np.ndarray,
                            query_label: Union[int, np.ndarray], save_dir: Path,
                            num_classes: int = 2, prefix: str = ""):
    """
    Save segmentation results as colored PLY files.

    Args:
        points: (N, 3) point coordinates
        predictions: (N,) predicted labels
        ground_truth: (N,) ground truth labels
        query_point: (3,) or (B, 3) query point location(s)
        query_label: scalar or (B,) query label(s)
        save_dir: Directory to save PLY files
        num_classes: Number of segmentation classes
        prefix: Optional prefix for filenames
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Ensure inputs are numpy arrays
    points = np.array(points)
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    query_point = np.array(query_point)

    # Handle different input shapes
    if len(points.shape) != 2 or points.shape[1] != 3:
        # Try to reshape if it's a flattened array
        if points.size % 3 == 0 and points.size > 3:
            points = points.reshape(-1, 3)
        else:
            return  # Skip invalid data silently

    if len(predictions.shape) != 1:
        if len(predictions.shape) == 2 and predictions.shape[1] > 1:
            # Convert logits to predictions
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions.flatten()

    if len(ground_truth.shape) != 1:
        ground_truth = ground_truth.flatten()

    # Ensure same length
    min_len = min(len(points), len(predictions), len(ground_truth))
    points = points[:min_len]
    predictions = predictions[:min_len]
    ground_truth = ground_truth[:min_len]

    # Generate colors for predictions
    pred_colors = apply_segmentation_colors(points, predictions, num_classes)

    # Generate colors for ground truth
    gt_colors = apply_segmentation_colors(points, ground_truth, num_classes)

    # Highlight query points
    if query_point.size > 0:
        pred_colors = highlight_query_point(points, pred_colors, query_point)
        gt_colors = highlight_query_point(points, gt_colors, query_point)

    # Save prediction results
    pred_filename = save_dir / f"{prefix}predictions.ply"
    save_colored_pointcloud(points, pred_colors, pred_filename)

    # Save ground truth
    gt_filename = save_dir / f"{prefix}ground_truth.ply"
    save_colored_pointcloud(points, gt_colors, gt_filename)

    # Save input points (gray)
    input_colors = np.full_like(points, 128)  # Gray color
    if query_point.size > 0:
        input_colors = highlight_query_point(points, input_colors, query_point)
    input_filename = save_dir / f"{prefix}input_points.ply"
    save_colored_pointcloud(points, input_colors, input_filename)

    # Save comparison (side by side visualization data)
    comparison_data = {
        'points': points.tolist(),
        'predictions': predictions.tolist(),
        'ground_truth': ground_truth.tolist(),
        'query_point': query_point.tolist(),
        'query_label': int(query_label) if np.isscalar(query_label) else query_label.tolist(),
        'num_classes': num_classes
    }

    import json
    comparison_filename = save_dir / f"{prefix}comparison_data.json"
    with open(comparison_filename, 'w') as f:
        json.dump(comparison_data, f, indent=2)


def create_segmentation_legend(num_classes: int, save_path: str):
    """
    Create a legend image showing segmentation colors.

    Args:
        num_classes: Number of segmentation classes
        save_path: Path to save legend image
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        colors = get_segmentation_colors(num_classes)

        fig, ax = plt.subplots(figsize=(8, max(4, num_classes * 0.3)))

        patches = []
        labels = []

        for i in range(num_classes):
            color = colors[i] / 255.0  # Normalize to [0, 1]
            patch = mpatches.Rectangle((0, 0), 1, 1, facecolor=color)
            patches.append(patch)
            labels.append(f"Class {i}")

        ax.legend(patches, labels, loc='center', bbox_to_anchor=(0.5, 0.5))
        ax.axis('off')
        ax.set_title('Segmentation Color Legend', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    except ImportError:
        pass  # Skip legend creation silently


def batch_save_predictions(batch_points: np.ndarray, batch_predictions: np.ndarray,
                          batch_ground_truth: np.ndarray, batch_query_points: np.ndarray,
                          batch_query_labels: np.ndarray, save_dir: Path,
                          epoch: int, batch_idx: int, max_samples: int = 4):
    """
    Save predictions for a batch of samples.

    Args:
        batch_points: (B, N, 3) batch of point clouds
        batch_predictions: (B, N) batch of predictions
        batch_ground_truth: (B, N) batch of ground truth
        batch_query_points: (B, 3) batch of query points
        batch_query_labels: (B,) batch of query labels
        save_dir: Directory to save results
        epoch: Current epoch number
        batch_idx: Current batch index
        max_samples: Maximum number of samples to save from batch
    """
    batch_size = min(batch_points.shape[0], max_samples)

    for i in range(batch_size):
        sample_dir = save_dir / f"epoch_{epoch:03d}" / f"batch_{batch_idx:03d}" / f"sample_{i:02d}"

        save_segmentation_results(
            points=batch_points[i],
            predictions=batch_predictions[i],
            ground_truth=batch_ground_truth[i],
            query_point=batch_query_points[i],
            query_label=batch_query_labels[i],
            save_dir=sample_dir,
            num_classes=2  # Binary segmentation
        )