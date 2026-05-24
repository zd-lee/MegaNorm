"""
Metrics for Point Cloud Segmentation
点云分割评估指标
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def calculate_metrics(masks, targets, offset=None):
    """
    Calculate accuracy, IoU, precision, recall, and mean GT

    Args:
        masks: (N_total,) mask logits
        targets: (N_total,) binary ground truth

    Returns:
        accuracy, iou, precision, recall, mean_gt
    """
    # Get binary predictions
    masks_sigmoid = torch.sigmoid(masks)
    preds = (masks_sigmoid > 0.5).float()

    assert(preds.shape == targets.shape)

    correct = (preds == targets).sum().item()
    total = targets.numel()
    accuracy = correct / total

    intersection = (preds * targets).sum().item()
    union = (preds + targets).clamp(0, 1).sum().item()
    iou = intersection / union if union > 0 else 0

    # Precision/Recall for single sample
    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    mean_gt = targets.float().mean().item()

    return accuracy, iou, precision, recall, mean_gt

def calculate_metrics_inv(masks, targets, offset=None):
    '''
    确保每个样本的acc>0.5
    '''
    pred = torch.sigmoid(masks) > 0.5
    inv_target = targets.clone()
    if offset==None:
        ori_acc = (pred==inv_target).float().mean()
        if ori_acc < 0.5:
            inv_target = 1 - inv_target
        return calculate_metrics(masks,inv_target,offset)
    start_idx = 0
    for edi in offset:
        ori_acc = (pred[start_idx:edi]==inv_target[start_idx:edi]).float().mean()
        if ori_acc < 0.5:
            inv_target[start_idx:edi] = 1 - inv_target[start_idx:edi]
        start_idx = edi

    return calculate_metrics(masks, inv_target, offset)


def calculate_unoriented_normal_metric(normals, gt_normals):
    """
    Calculate unoriented normal metric (average angular error in degrees, 0-90)

    Args:
        normals: (N, 3) predicted normals
        gt_normals: (N, 3) ground truth normals

    Returns:
        avg_angle_error: Average angular error in degrees (0-90)
    """
    # Normalize vectors
    normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-8)
    gt_normals = gt_normals / (torch.norm(gt_normals, dim=1, keepdim=True) + 1e-8)

    # Calculate dot product
    dot_product = (normals * gt_normals).sum(dim=1).clamp(-1.0, 1.0)

    # Calculate angle in radians, then convert to degrees
    angles_rad = torch.acos(torch.abs(dot_product))  # abs handles unoriented normals
    angles_deg = angles_rad * 180.0 / np.pi

    # Return average angle error
    avg_angle_error = angles_deg.mean().item()

    rmse = torch.sqrt((angles_deg**2).mean()).item()
    return avg_angle_error, rmse


def calculate_normal_metrics(pred_normals, gt_normals):
    """
    计算法向量估计的详细指标
    Calculate comprehensive angular error metrics for normal estimation

    Args:
        pred_normals: (N, 3) predicted normals
        gt_normals: (N, 3) ground truth normals

    Returns:
        dict: 包含以下指标的字典
            - mean_error: 平均角度误差 (degrees)
            - median_error: 中位数角度误差 (degrees)
            - pct_5deg: 误差<5°的点百分比 (PGP@5°)
            - pct_10deg: 误差<10°的点百分比 (PGP@10°)
            - pct_30deg: 误差<30°的点百分比 (PGP@30°)
            - rmse_unoriented: Unoriented RMSE
            - rmse_oriented: Oriented RMSE
    """
    import torch.nn.functional as F

    # 归一化法向量
    pred_normalized = F.normalize(pred_normals, p=2, dim=1)
    gt_normalized = F.normalize(gt_normals, p=2, dim=1)
    dot_product = torch.clamp((pred_normalized * gt_normalized).sum(dim=1), -1.0, 1.0)

    # Unoriented metrics (不考虑方向)
    angular_error_rad = torch.acos(torch.abs(dot_product))
    angular_error_deg = angular_error_rad * 180.0 / np.pi

    mean_error = angular_error_deg.mean().item()
    median_error = angular_error_deg.median().item()
    pct_5deg = (angular_error_deg < 5.0).float().mean().item() * 100
    pct_10deg = (angular_error_deg < 10.0).float().mean().item() * 100
    pct_30deg = (angular_error_deg < 30.0).float().mean().item() * 100
    rmse_unoriented = torch.sqrt((angular_error_deg ** 2).mean()).item()

    # Oriented metrics (考虑方向)
    angular_error_oriented_rad = torch.acos(dot_product)
    angular_error_oriented_deg = angular_error_oriented_rad * 180.0 / np.pi
    needs_flip = (angular_error_oriented_deg > 90.0).float().mean().item() > 0.5

    if needs_flip:
        dot_product_flipped = torch.clamp((-pred_normalized * gt_normalized).sum(dim=1), -1.0, 1.0)
        angular_error_oriented_rad = torch.acos(dot_product_flipped)
        angular_error_oriented_deg = angular_error_oriented_rad * 180.0 / np.pi

    rmse_oriented = torch.sqrt((angular_error_oriented_deg ** 2).mean()).item()

    return {
        'Mean_Error': mean_error,
        'Median_Error': median_error,
        'PGP5': pct_5deg,
        'PGP10': pct_10deg,
        'PGP30': pct_30deg,
        'RMSE_Unoriented': rmse_unoriented,
        'RMSE_Oriented': rmse_oriented
    }


def calculate_edge_accuracy(A, B, gt_flip):
    """
    Calculate edge accuracy metric.
    计算边一致性准确率指标

    An edge (i,j) is correct when:
    (gt_flip[i] == gt_flip[j]) == (A[i][j] > B[i][j])

    Edge is consistent if both patches have same flip status (same orientation)
    边是一致的，当且仅当两个patch具有相同的flip状态(相同方向)

    Args:
        A: (P, P) consistency matrix - counts agreements between patches
        B: (P, P) inconsistency matrix - counts disagreements
        gt_flip: (P,) ground truth flip status per patch (0 or 1)

    Returns:
        edge_accuracy: float, percentage of edges correctly predicted
        total_edges: int, total number of edges evaluated
    """
    # Convert to numpy if needed
    if not isinstance(A, np.ndarray):
        A = A.cpu().numpy() if hasattr(A, 'cpu') else np.array(A)
    if not isinstance(B, np.ndarray):
        B = B.cpu().numpy() if hasattr(B, 'cpu') else np.array(B)
    if not isinstance(gt_flip, np.ndarray):
        gt_flip = gt_flip.cpu().numpy() if hasattr(gt_flip, 'cpu') else np.array(gt_flip)

    # Get edges (i, j) where there is overlap (A + B > 0)
    sum_AB = A + B
    edge_mask = sum_AB > 0

    # For upper triangle only (avoid double counting)
    edge_mask = np.triu(edge_mask, k=1)
    edge_indices = np.where(edge_mask)

    if len(edge_indices[0]) == 0:
        return 0.0, 0

    # For each edge (i, j):
    # GT consistency: gt_flip[i] == gt_flip[j] (True if both patches have same orientation)
    # Predicted consistency: A[i,j] > B[i,j] (True if agreement > disagreement)
    i_indices = edge_indices[0]
    j_indices = edge_indices[1]

    gt_consistent = (gt_flip[i_indices] == gt_flip[j_indices])
    pred_consistent = (A[i_indices, j_indices] > B[i_indices, j_indices])

    # Edge is correct if GT consistency matches predicted consistency
    correct_edges = (gt_consistent == pred_consistent).sum()
    total_edges = len(i_indices)

    edge_accuracy = (correct_edges / total_edges) * 100.0

    return edge_accuracy, total_edges