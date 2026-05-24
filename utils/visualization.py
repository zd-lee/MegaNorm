"""
Visualization utilities for Point Cloud Segmentation
点云分割可视化工具
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from pathlib import Path
from typing import Optional, Union
import json


def visualize_segmentation(coords: np.ndarray,
                         predictions: np.ndarray,
                         ground_truth: Optional[np.ndarray] = None,
                         query_point: Optional[np.ndarray] = None,
                         save_path: Optional[Union[str, Path]] = None,
                         title: str = "Point Cloud Segmentation",
                         figsize: tuple = (15, 5),
                         point_size: float = 1.0,
                         query_size: float = 50.0):
    """
    可视化点云分割结果
    
    Args:
        coords: 点云坐标 (N, 3)
        predictions: 预测结果 (N,)
        ground_truth: 真实标签 (N,) - 可选
        query_point: 查询点坐标 (3,) - 可选
        save_path: 保存路径
        title: 图像标题
        figsize: 图像大小
        point_size: 点的大小
        query_size: 查询点的大小
    """
    
    # 确定子图数量
    num_plots = 1 + (1 if ground_truth is not None else 0)
    
    fig = plt.figure(figsize=figsize)
    
    # 生成颜色映射
    unique_labels = np.unique(predictions)
    if ground_truth is not None:
        unique_labels = np.unique(np.concatenate([predictions, ground_truth]))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # 预测结果可视化
    ax1 = fig.add_subplot(1, num_plots, 1, projection='3d')
    
    for label in unique_labels:
        mask = predictions == label
        if mask.sum() > 0:
            ax1.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2],
                       c=[color_map[label]], s=point_size, alpha=0.7, label=f'Class {label}')
    
    # 添加查询点
    if query_point is not None:
        ax1.scatter(query_point[0], query_point[1], query_point[2],
                   c='red', s=query_size, marker='*', label='Query Point')
    
    ax1.set_title('Predictions')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    
    # 真实标签可视化（如果提供）
    if ground_truth is not None:
        ax2 = fig.add_subplot(1, num_plots, 2, projection='3d')
        
        for label in unique_labels:
            mask = ground_truth == label
            if mask.sum() > 0:
                ax2.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2],
                           c=[color_map[label]], s=point_size, alpha=0.7, label=f'Class {label}')
        
        # 添加查询点
        if query_point is not None:
            ax2.scatter(query_point[0], query_point[1], query_point[2],
                       c='red', s=query_size, marker='*', label='Query Point')
        
        ax2.set_title('Ground Truth')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_point_cloud_open3d(coords: np.ndarray,
                                labels: np.ndarray,
                                query_point: Optional[np.ndarray] = None,
                                save_path: Optional[Union[str, Path]] = None,
                                window_name: str = "Point Cloud Segmentation"):
    """
    使用Open3D可视化点云分割结果（交互式）
    
    Args:
        coords: 点云坐标 (N, 3)
        labels: 点云标签 (N,)
        query_point: 查询点坐标 (3,)
        save_path: 保存路径
        window_name: 窗口名称
    """
    
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    
    # 生成颜色
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    # 为每个点分配颜色
    point_colors = np.zeros((len(coords), 3))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        point_colors[mask] = colors[i][:3]  # RGB only
    
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    
    # 创建几何对象列表
    geometries = [pcd]
    
    # 添加查询点
    if query_point is not None:
        query_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        query_sphere.translate(query_point)
        query_sphere.paint_uniform_color([1, 0, 0])  # 红色
        geometries.append(query_sphere)
    
    # 可视化
    if save_path:
        # 保存为图像
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, visible=False)
        for geom in geometries:
            vis.add_geometry(geom)
        vis.capture_screen_image(str(save_path))
        vis.destroy_window()
    else:
        # 交互式显示
        o3d.visualization.draw_geometries(geometries, window_name=window_name)


def create_confusion_matrix_plot(confusion_matrix: np.ndarray,
                                class_names: Optional[list] = None,
                                save_path: Optional[Union[str, Path]] = None,
                                title: str = "Confusion Matrix",
                                figsize: tuple = (10, 8)):
    """
    创建混淆矩阵可视化
    
    Args:
        confusion_matrix: 混淆矩阵 (num_classes, num_classes)
        class_names: 类别名称列表
        save_path: 保存路径
        title: 图像标题
        figsize: 图像大小
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 归一化混淆矩阵
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    # 创建热力图
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # 设置标签
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(confusion_matrix))]
    
    ax.set(xticks=np.arange(confusion_matrix.shape[1]),
           yticks=np.arange(confusion_matrix.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 添加数值标注
    fmt = '.2f'
    thresh = cm_normalized.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, format(cm_normalized[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm_normalized[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_metrics_over_time(metrics_history: dict,
                          save_path: Optional[Union[str, Path]] = None,
                          title: str = "Training Metrics",
                          figsize: tuple = (15, 10)):
    """
    绘制训练过程中的指标变化
    
    Args:
        metrics_history: 指标历史字典
        save_path: 保存路径
        title: 图像标题
        figsize: 图像大小
    """
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Loss曲线
    if 'train_loss' in metrics_history:
        axes[0, 0].plot(metrics_history['train_loss'], label='Train Loss')
    if 'val_loss' in metrics_history:
        axes[0, 0].plot(metrics_history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy曲线
    if 'train_acc' in metrics_history:
        axes[0, 1].plot(metrics_history['train_acc'], label='Train Accuracy')
    if 'val_acc' in metrics_history:
        axes[0, 1].plot(metrics_history['val_acc'], label='Val Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # IoU曲线
    if 'train_iou' in metrics_history:
        axes[1, 0].plot(metrics_history['train_iou'], label='Train IoU')
    if 'val_iou' in metrics_history:
        axes[1, 0].plot(metrics_history['val_iou'], label='Val IoU')
    axes[1, 0].set_title('Mean IoU')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # F1 Score曲线
    if 'train_f1' in metrics_history:
        axes[1, 1].plot(metrics_history['train_f1'], label='Train F1')
    if 'val_f1' in metrics_history:
        axes[1, 1].plot(metrics_history['val_f1'], label='Val F1')
    axes[1, 1].set_title('Mean F1 Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_segmentation_results(coords: np.ndarray,
                            predictions: np.ndarray,
                            ground_truth: Optional[np.ndarray] = None,
                            query_point: Optional[np.ndarray] = None,
                            save_dir: Union[str, Path] = "results",
                            sample_name: str = "sample"):
    """
    保存分割结果的完整可视化
    
    Args:
        coords: 点云坐标
        predictions: 预测结果
        ground_truth: 真实标签
        query_point: 查询点
        save_dir: 保存目录
        sample_name: 样本名称
    """
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存matplotlib可视化
    visualize_segmentation(
        coords, predictions, ground_truth, query_point,
        save_path=save_dir / f"{sample_name}_segmentation.png",
        title=f"{sample_name} Segmentation Results"
    )
    
    # 保存点云文件
    save_point_cloud_with_labels(
        coords, predictions,
        save_path=save_dir / f"{sample_name}_predictions.ply"
    )
    
    if ground_truth is not None:
        save_point_cloud_with_labels(
            coords, ground_truth,
            save_path=save_dir / f"{sample_name}_ground_truth.ply"
        )
    
    print(f"Results saved to: {save_dir}")


def save_training_visualization(coords: np.ndarray,
                               normals: np.ndarray,
                               predictions: np.ndarray,
                               ground_truth: np.ndarray,
                               confidence: np.ndarray,
                               query_point: np.ndarray,
                               save_dir: Union[str, Path],
                               epoch: int,
                               batch_idx: int,
                               model_names: list = [],
                               offset: np.ndarray = None):
    """
    保存训练过程中的点云数据为PLY文件
    输出两个点云：红/绿区分正确与错误，渐变色展示置信度，并高亮查询点附近样本

    Args:
        coords: 点云坐标 (N, 3)
        normals: 点云法向量 (N, 3)
        predictions: 预测结果 (N,)
        ground_truth: 真实标签 (N,)
        confidence: 置信度 (N,)
        query_point: 查询点 (3,)
        save_dir: 保存目录
        epoch: 当前epoch
        batch_idx: 当前batch索引
        model_names: 点云样本名称列表，对应生成的PLY文件前缀
        offset: 批处理偏移量 (N,)
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    
    # 准备正确/错误与置信度的配色
    correct_mask = (predictions == ground_truth)
    incorrect_mask = ~correct_mask

    correctness_colors = np.zeros((len(coords), 3))
    correctness_colors[correct_mask] = (0.0, 1.0, 0.0)  # 绿色表示正确
    correctness_colors[incorrect_mask] = (1.0, 0.0, 0.0)  # 红色表示错误
    
    gt_flip_colors = np.zeros((len(coords), 3))
    gt_flip_colors[np.bool(ground_truth)] = (0.5, 0.5, 0.5)  
    gt_flip_colors[~np.bool(ground_truth)] = (0.0, 0.0, 1.0) 

    # 使用渐变色映射置信度，防止除零
    confidence = confidence.astype(float, copy=False)
    conf_min = confidence.min()
    conf_range = confidence.max() - conf_min
    if conf_range == 0:
        normalized_conf = np.zeros_like(confidence)
    else:
        normalized_conf = (confidence - conf_min) / conf_range
    cmap = plt.get_cmap('viridis')
    confidence_colors = cmap(normalized_conf)[:, :3]

    # 高亮查询点附近的若干点，便于定位
    query_points = np.asarray(query_point) if query_point is not None else None
    if query_points is not None and query_points.size > 0:
        if query_points.ndim == 1:
            query_points = query_points.reshape(1, -1)
        highlight_color = (1.0, 0.0, 1.0)  # 品红色，易于分辨
        highlight_count = 4
        start_idx = 0
        for i, end_idx in enumerate(offset):
            sample_size = end_idx - start_idx
            if sample_size <= 0:
                start_idx = end_idx
                continue
            sample_coords = coords[start_idx:end_idx]
            if query_points.shape[0] == len(offset):
                sample_query = query_points[i]
            else:
                sample_query = query_points[0]
            if sample_query.shape[-1] != 3:
                start_idx = end_idx
                continue
            distances = np.linalg.norm(sample_coords - sample_query[:3], axis=1)
            k = min(highlight_count, len(distances))
            if k == 0:
                start_idx = end_idx
                continue
            nearest_idx = np.argpartition(distances, k - 1)[:k]
            global_idx = start_idx + nearest_idx
            correctness_colors[global_idx] = highlight_color
            confidence_colors[global_idx] = highlight_color
            start_idx = end_idx



    start_idx = 0
    for i in range(len(offset)):
        end_idx = offset[i]
        slice_coords = coords[start_idx:end_idx]
        slice_normals = normals[start_idx:end_idx]

        correctness_pcd = o3d.geometry.PointCloud()
        correctness_pcd.points = o3d.utility.Vector3dVector(slice_coords)
        correctness_pcd.colors = o3d.utility.Vector3dVector(
            correctness_colors[start_idx:end_idx]
        )
        correctness_pcd.normals = o3d.utility.Vector3dVector(slice_normals)
        correctness_path = save_dir / f"{model_names[i]}_correctness.ply"
        o3d.io.write_point_cloud(str(correctness_path), correctness_pcd)

        confidence_pcd = o3d.geometry.PointCloud()
        confidence_pcd.points = o3d.utility.Vector3dVector(slice_coords)
        confidence_pcd.colors = o3d.utility.Vector3dVector(
            confidence_colors[start_idx:end_idx]
        )
        confidence_pcd.normals = o3d.utility.Vector3dVector(slice_normals)
        confidence_path = save_dir / f"{model_names[i]}_confidence.ply"
        o3d.io.write_point_cloud(str(confidence_path), confidence_pcd)

        gt_flip_pcd = o3d.geometry.PointCloud()
        gt_flip_pcd.points = o3d.utility.Vector3dVector(slice_coords)
        gt_flip_pcd.colors = o3d.utility.Vector3dVector(
            gt_flip_colors[start_idx:end_idx]
        )
        gt_flip_pcd.normals = o3d.utility.Vector3dVector(slice_normals)
        gt_flip_path = save_dir / f"{model_names[i]}_gt_flip.ply"
        o3d.io.write_point_cloud(str(gt_flip_path), gt_flip_pcd)

        slice_input_normals = normals[start_idx:end_idx]  # (N_slice, 3)
        # slice_gt_mask = ground_truth[start_idx:end_idx]  # (N_slice,)
        slice_pred_mask = predictions[start_idx:end_idx]  # (N_slice,)

        # gt_flipped_normals = slice_input_normals * (1 - 2 * slice_gt_mask.reshape(-1, 1))
        # gt_op_colors = np.zeros((len(slice_coords), 3))
        # gt_op_colors[:] = (0.5, 0.5, 0.5)  # Default gray
        # gt_op_colors[slice_gt_mask > 0] = (0.0, 1.0, 0.0)  # Green for flipped points

        # gt_op_pcd = o3d.geometry.PointCloud()
        # gt_op_pcd.points = o3d.utility.Vector3dVector(slice_coords)
        # gt_op_pcd.colors = o3d.utility.Vector3dVector(gt_op_colors)
        # gt_op_pcd.normals = o3d.utility.Vector3dVector(gt_flipped_normals)
        # gt_op_path = save_dir / f"{model_names[i]}_gt_mask_op.ply"
        # o3d.io.write_point_cloud(str(gt_op_path), gt_op_pcd)

        pred_flipped_normals = slice_input_normals * (1 - 2 * slice_pred_mask.reshape(-1, 1))
        pred_op_colors = np.zeros((len(slice_coords), 3))
        pred_op_colors[:] = (0.5, 0.5, 0.5)  # Default gray
        pred_op_colors[slice_pred_mask > 0] = (0.0, 0.0, 1.0)  # Blue for flipped points

        pred_op_pcd = o3d.geometry.PointCloud()
        pred_op_pcd.points = o3d.utility.Vector3dVector(slice_coords)
        pred_op_pcd.colors = o3d.utility.Vector3dVector(pred_op_colors)
        pred_op_pcd.normals = o3d.utility.Vector3dVector(pred_flipped_normals)
        pred_op_path = save_dir / f"{model_names[i]}_pred_op.ply"
        o3d.io.write_point_cloud(str(pred_op_path), pred_op_pcd)

        start_idx = end_idx



    # 保存额外的数据信息为JSON文件
    metadata = {
        "epoch": int(epoch),
        "batch_idx": int(batch_idx),
        "sample_name": model_names,
        "query_point": query_point.tolist(),
        "num_points": len(coords),
        "mean_confidence": float(confidence.mean()),
        "ground_truth_positive": int(ground_truth.sum()),
        "predictions_positive": int(predictions.sum())
    }

    json_path = save_dir / f"metadata.json"
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Training data saved to PLY: {save_dir}")
    print(f"Metadata saved to JSON: {json_path}")


def save_point_cloud_with_labels(coords: np.ndarray,
                                labels: np.ndarray,
                                save_path: Union[str, Path]):
    """
    保存带标签的点云文件

    Args:
        coords: 点云坐标 (N, 3)
        labels: 点云标签 (N,)
        save_path: 保存路径
    """

    # 创建颜色映射
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    # 为每个点分配颜色
    point_colors = np.zeros((len(coords), 3))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        point_colors[mask] = colors[i][:3]  # RGB only

    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)

    # 保存
    o3d.io.write_point_cloud(str(save_path), pcd)
    print(f"Point cloud saved to: {save_path}")


import torch
from torch import nn
def calculate_normal_metrics(pred_normals, gt_normals):
    """
    Calculate angular error metrics for normal estimation

    Args:
        pred_normals: (N, 3) predicted normals (unit vectors)
        gt_normals: (N, 3) ground truth normals

    Returns:
        dict with metrics: mean_error, median_error, pct_5deg, pct_10deg, pct_30deg, rmse_unoriented, rmse_oriented
    """
    # Normalize both vectors
    pred_normalized = nn.functional.normalize(pred_normals, p=2, dim=1)
    gt_normalized = nn.functional.normalize(gt_normals, p=2, dim=1)

    # Compute dot product
    dot_product = (pred_normalized * gt_normalized).sum(dim=1)

    # Clamp to [-1, 1] to avoid numerical issues with arccos
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    # ========== Unoriented metrics ==========
    # Compute angular error in radians, then convert to degrees
    angular_error_rad = torch.acos(torch.abs(dot_product))  # Use abs for unoriented error
    angular_error_deg = angular_error_rad * 180.0 / np.pi

    # Calculate statistics
    mean_error = angular_error_deg.mean().item()
    median_error = angular_error_deg.median().item()

    # Percentage within thresholds
    pct_5deg = (angular_error_deg < 5.0).float().mean().item() * 100
    pct_10deg = (angular_error_deg < 10.0).float().mean().item() * 100
    pct_30deg = (angular_error_deg < 30.0).float().mean().item() * 100

    # Unoriented RMSE
    rmse_unoriented = torch.sqrt((angular_error_deg ** 2).mean()).item()

    # ========== Oriented metrics ==========
    # Oriented angular error (without abs)
    angular_error_oriented_rad = torch.acos(dot_product)  # No abs - considers direction
    angular_error_oriented_deg = angular_error_oriented_rad * 180.0 / np.pi

    # Check if we need to flip normals (if more than half points have angle > 90)
    needs_flip = (angular_error_oriented_deg > 90.0).float().mean().item() > 0.5

    if needs_flip:
        # Flip all normals and recompute
        dot_product_flipped = (-pred_normalized * gt_normalized).sum(dim=1)
        dot_product_flipped = torch.clamp(dot_product_flipped, -1.0, 1.0)
        angular_error_oriented_rad = torch.acos(dot_product_flipped)
        angular_error_oriented_deg = angular_error_oriented_rad * 180.0 / np.pi

    # Oriented RMSE
    rmse_oriented = torch.sqrt((angular_error_oriented_deg ** 2).mean()).item()

    return {
        'mean_error': mean_error,
        'median_error': median_error,
        'pct_5deg': pct_5deg,
        'pct_10deg': pct_10deg,
        'pct_30deg': pct_30deg,
        'rmse_unoriented': rmse_unoriented,
        'rmse_oriented': rmse_oriented
    }

def save_normal_estimation_visualization(coords: np.ndarray,
                                         pred_normals: np.ndarray,
                                         pca_normals: np.ndarray,
                                         gt_normals: np.ndarray,
                                         save_dir: Union[str, Path],
                                         epoch: int,
                                         batch_idx: int,
                                         model_names: list = [],
                                         offset: np.ndarray = None):
    """
    保存 normal estimation 的可视化结果（PLY文件）

    生成两个PLY文件:
    1. {model_name}_pred.ply: 预测的法向量，颜色基于 normal consistency
    2. {model_name}_pca.ply: PCA估计的法向量，颜色基于 normal consistency

    Normal consistency = dot(normalize(normal), normalize(gt_normal))
    颜色映射: 绿色(consistency=1) -> 黄色(0) -> 红色(-1)

    Args:
        coords: 点云坐标 (N_total, 3)
        pred_normals: 预测的法向量 (N_total, 3)
        pca_normals: PCA估计的法向量 (N_total, 3)
        gt_normals: 地面真实法向量 (N_total, 3)
        save_dir: 保存目录
        epoch: 当前epoch
        batch_idx: 当前batch索引
        model_names: 点云样本名称列表
        offset: 批处理偏移量 (B,) - numpy array
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Normalize all normals
    pred_normals_norm = pred_normals / (np.linalg.norm(pred_normals, axis=1, keepdims=True) + 1e-8)
    pca_normals_norm = pca_normals / (np.linalg.norm(pca_normals, axis=1, keepdims=True) + 1e-8)
    gt_normals_norm = gt_normals / (np.linalg.norm(gt_normals, axis=1, keepdims=True) + 1e-8)

    # Compute normal consistency (dot product with GT)
    pred_consistency = np.sum(pred_normals_norm * gt_normals_norm, axis=1)  # (N_total,) in [-1, 1]
    pca_consistency = np.sum(pca_normals_norm * gt_normals_norm, axis=1)    # (N_total,) in [-1, 1]
    
    pred_consistency = np.clip(pred_consistency, -1.0, 1.0)
    pca_consistency = np.clip(pca_consistency, -1.0, 1.0)
    
    pred_unoriented_consistency = np.abs(pred_consistency)
    pca_unoriented_consistency = np.abs(pca_consistency)
    
    pred_metric = calculate_normal_metrics(
        torch.tensor(pred_normals), torch.tensor(gt_normals)
    )
    pca_metric = calculate_normal_metrics(
        torch.tensor(pca_normals), torch.tensor(gt_normals)
    )
    pred_rmse_u = pred_metric['rmse_unoriented']
    pca_rmse_u = pca_metric['rmse_unoriented']
    

    # Map consistency to colors: [-1, 1] -> [red, yellow, green]
    # Use RdYlGn colormap: Red (bad) -> Yellow (medium) -> Green (good)
    def consistency_to_color(consistency):
        """Map consistency in [-1, 1] to RGB color"""
        # Normalize to [0, 1]
        normalized = (consistency + 1.0) / 2.0  # [-1, 1] -> [0, 1]

        # Use RdYlGn colormap
        cmap = plt.get_cmap('viridis')
        colors = cmap(normalized)[:, :3]  # Get RGB, discard alpha

        return colors

    pred_colors = consistency_to_color(pred_unoriented_consistency)
    pca_colors = consistency_to_color(pca_unoriented_consistency)

    # Split by offset and save each sample
    if offset is None:
        # Single sample
        offset = np.array([len(coords)])
        if not model_names:
            model_names = ['unknown']

    start_idx = 0
    for i in range(len(offset)):
        end_idx = offset[i]

        # Get model name
        model_name = model_names[i] if i < len(model_names) else f'sample_{i}'

        # Slice data for current sample
        slice_coords = coords[start_idx:end_idx]
        slice_pred_normals = pred_normals_norm[start_idx:end_idx]
        slice_pca_normals = pca_normals_norm[start_idx:end_idx]
        slice_pred_colors = pred_colors[start_idx:end_idx]
        slice_pca_colors = pca_colors[start_idx:end_idx]

        # Save pred normals PLY
        pred_pcd = o3d.geometry.PointCloud()
        pred_pcd.points = o3d.utility.Vector3dVector(slice_coords)
        pred_pcd.normals = o3d.utility.Vector3dVector(slice_pred_normals)
        pred_pcd.colors = o3d.utility.Vector3dVector(slice_pred_colors)
        pred_path = save_dir / f"{model_name}_pred.ply"
        o3d.io.write_point_cloud(str(pred_path), pred_pcd)

        # Save PCA normals PLY
        pca_pcd = o3d.geometry.PointCloud()
        pca_pcd.points = o3d.utility.Vector3dVector(slice_coords)
        pca_pcd.normals = o3d.utility.Vector3dVector(slice_pca_normals)
        pca_pcd.colors = o3d.utility.Vector3dVector(slice_pca_colors)
        pca_path = save_dir / f"{model_name}_pca.ply"
        o3d.io.write_point_cloud(str(pca_path), pca_pcd)

        start_idx = end_idx

    # Save metadata
    metadata = {
        "epoch": int(epoch),
        "batch_idx": int(batch_idx),
        "model_names": model_names if model_names else ["unknown"],
        "num_points": len(coords),
        "mean_pred_consistency": float(pred_consistency.mean()),
        "mean_pca_consistency": float(pca_consistency.mean()),
        "mean_pred_unoriented_consistency": float(pred_unoriented_consistency.mean()),
        "mean_pca_unoriented_consistency": float(pca_unoriented_consistency.mean()),
        "pred_rmseu":float(pred_rmse_u),
        "pca_rmseu":float(pca_rmse_u),
        "pred_consistency_stats": {
            "min": float(pred_consistency.min()),
            "max": float(pred_consistency.max()),
            "median": float(np.median(pred_consistency))
        },
        "pca_consistency_stats": {
            "min": float(pca_consistency.min()),
            "max": float(pca_consistency.max()),
            "median": float(np.median(pca_consistency))
        }
    }

    json_path = save_dir / "metadata.json"
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Normal estimation visualization saved to: {save_dir}")
    print(f"  - {len(offset)} samples saved")
    print(f"  - Mean pred consistency: {metadata['mean_pred_consistency']:.4f}")
    print(f"  - Mean PCA consistency: {metadata['mean_pca_consistency']:.4f}")
    print(f"  - Mean pred unoriented consistency: {metadata['mean_pred_unoriented_consistency']:.4f}")
    print(f"  - Mean PCA unoriented consistency: {metadata['mean_pca_unoriented_consistency']:.4f}")
    print(f"  - Pred RMSE (unoriented): {pred_rmse_u:.4f} degrees")
    print(f"  - PCA RMSE (unoriented): {pca_rmse_u:.4f} degrees")
    

