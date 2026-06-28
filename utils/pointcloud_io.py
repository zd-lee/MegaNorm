import numpy as np

from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any
import struct

try:
    import open3d as o3d

    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    pass


def get_segmentation_colors(num_classes: int = 20) -> np.ndarray:
    colors = np.array(
        [
            [128, 128, 128],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 128, 0],
            [128, 255, 0],
            [255, 0, 128],
            [128, 0, 255],
            [0, 128, 255],
            [255, 255, 128],
            [128, 255, 255],
            [255, 128, 255],
            [128, 128, 255],
            [255, 128, 128],
            [128, 255, 128],
            [64, 64, 64],
            [192, 192, 192],
        ]
    )
    if num_classes > len(colors):
        additional_colors = np.random.randint(0, 256, (num_classes - len(colors), 3))
        colors = np.vstack([colors, additional_colors])
    return colors[:num_classes]


def apply_segmentation_colors(
    points: np.ndarray, labels: np.ndarray, num_classes: int = 2
) -> np.ndarray:
    colors_palette = get_segmentation_colors(num_classes)
    labels = np.clip(labels, 0, num_classes - 1)
    colors = colors_palette[labels.astype(int)]
    return colors


def highlight_query_point(
    points: np.ndarray,
    colors: np.ndarray,
    query_point: np.ndarray,
    radius: float = 0.05,
    highlight_color: Tuple[int, int, int] = (255, 215, 0),
) -> np.ndarray:
    if len(query_point.shape) == 0 or query_point.size == 0:
        return colors
    if query_point.shape[-1] != 3:
        return colors
    query_point = query_point.reshape(-1, 3)
    updated_colors = colors.copy()
    for qp in query_point:
        distances = np.linalg.norm(points - qp, axis=1)
        nearby_mask = distances <= radius
        updated_colors[nearby_mask] = highlight_color
    return updated_colors


def write_ply_ascii(
    filename: str,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
):
    N = points.shape[0]
    with open(filename, "w") as f:
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
        for i in range(N):
            line = f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}"
            if normals is not None:
                line += f" {normals[i, 0]:.6f} {normals[i, 1]:.6f} {normals[i, 2]:.6f}"
            if colors is not None:
                line += f" {int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}"
            line += "\n"
            f.write(line)


def save_colored_pointcloud(
    points: np.ndarray,
    colors: np.ndarray,
    filename: str,
    normals: Optional[np.ndarray] = None,
):
    filename = str(filename)
    if OPEN3D_AVAILABLE:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
        o3d.io.write_point_cloud(filename, pcd)
    else:
        write_ply_ascii(filename, points, colors, normals)


def save_segmentation_results(
    points: np.ndarray,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    query_point: np.ndarray,
    query_label: Union[int, np.ndarray],
    save_dir: Path,
    num_classes: int = 2,
    prefix: str = "",
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    points = np.array(points)
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    query_point = np.array(query_point)
    if len(points.shape) != 2 or points.shape[1] != 3:
        if points.size % 3 == 0 and points.size > 3:
            points = points.reshape(-1, 3)
        else:
            return
    if len(predictions.shape) != 1:
        if len(predictions.shape) == 2 and predictions.shape[1] > 1:
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions.flatten()
    if len(ground_truth.shape) != 1:
        ground_truth = ground_truth.flatten()
    min_len = min(len(points), len(predictions), len(ground_truth))
    points = points[:min_len]
    predictions = predictions[:min_len]
    ground_truth = ground_truth[:min_len]
    pred_colors = apply_segmentation_colors(points, predictions, num_classes)
    gt_colors = apply_segmentation_colors(points, ground_truth, num_classes)
    if query_point.size > 0:
        pred_colors = highlight_query_point(points, pred_colors, query_point)
        gt_colors = highlight_query_point(points, gt_colors, query_point)
    pred_filename = save_dir / f"{prefix}predictions.ply"
    save_colored_pointcloud(points, pred_colors, pred_filename)
    gt_filename = save_dir / f"{prefix}ground_truth.ply"
    save_colored_pointcloud(points, gt_colors, gt_filename)
    input_colors = np.full_like(points, 128)
    if query_point.size > 0:
        input_colors = highlight_query_point(points, input_colors, query_point)
    input_filename = save_dir / f"{prefix}input_points.ply"
    save_colored_pointcloud(points, input_colors, input_filename)
    comparison_data = {
        "points": points.tolist(),
        "predictions": predictions.tolist(),
        "ground_truth": ground_truth.tolist(),
        "query_point": query_point.tolist(),
        "query_label": (
            int(query_label) if np.isscalar(query_label) else query_label.tolist()
        ),
        "num_classes": num_classes,
    }

    import json

    comparison_filename = save_dir / f"{prefix}comparison_data.json"
    with open(comparison_filename, "w") as f:
        json.dump(comparison_data, f, indent=2)


def create_segmentation_legend(num_classes: int, save_path: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        colors = get_segmentation_colors(num_classes)
        fig, ax = plt.subplots(figsize=(8, max(4, num_classes * 0.3)))
        patches = []
        labels = []
        for i in range(num_classes):
            color = colors[i] / 255.0
            patch = mpatches.Rectangle((0, 0), 1, 1, facecolor=color)
            patches.append(patch)
            labels.append(f"Class {i}")
        ax.legend(patches, labels, loc="center", bbox_to_anchor=(0.5, 0.5))
        ax.axis("off")
        ax.set_title("Segmentation Color Legend", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    except ImportError:
        pass


def batch_save_predictions(
    batch_points: np.ndarray,
    batch_predictions: np.ndarray,
    batch_ground_truth: np.ndarray,
    batch_query_points: np.ndarray,
    batch_query_labels: np.ndarray,
    save_dir: Path,
    epoch: int,
    batch_idx: int,
    max_samples: int = 4,
):
    batch_size = min(batch_points.shape[0], max_samples)
    for i in range(batch_size):
        sample_dir = (
            save_dir
            / f"epoch_{epoch:03d}"
            / f"batch_{batch_idx:03d}"
            / f"sample_{i:02d}"
        )
        save_segmentation_results(
            points=batch_points[i],
            predictions=batch_predictions[i],
            ground_truth=batch_ground_truth[i],
            query_point=batch_query_points[i],
            query_label=batch_query_labels[i],
            save_dir=sample_dir,
            num_classes=2,
        )
