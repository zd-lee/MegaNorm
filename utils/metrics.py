import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def calculate_metrics(masks, targets, offset=None):
    masks_sigmoid = torch.sigmoid(masks)
    preds = (masks_sigmoid > 0.5).float()
    assert preds.shape == targets.shape
    correct = (preds == targets).sum().item()
    total = targets.numel()
    accuracy = correct / total
    intersection = (preds * targets).sum().item()
    union = (preds + targets).clamp(0, 1).sum().item()
    iou = intersection / union if union > 0 else 0
    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    mean_gt = targets.float().mean().item()
    return accuracy, iou, precision, recall, mean_gt


def calculate_metrics_inv(masks, targets, offset=None):
    pred = torch.sigmoid(masks) > 0.5
    inv_target = targets.clone()
    if offset == None:
        ori_acc = (pred == inv_target).float().mean()
        if ori_acc < 0.5:
            inv_target = 1 - inv_target
        return calculate_metrics(masks, inv_target, offset)
    start_idx = 0
    for edi in offset:
        ori_acc = (pred[start_idx:edi] == inv_target[start_idx:edi]).float().mean()
        if ori_acc < 0.5:
            inv_target[start_idx:edi] = 1 - inv_target[start_idx:edi]
        start_idx = edi
    return calculate_metrics(masks, inv_target, offset)


def calculate_unoriented_normal_metric(normals, gt_normals):
    normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-8)
    gt_normals = gt_normals / (torch.norm(gt_normals, dim=1, keepdim=True) + 1e-8)
    dot_product = (normals * gt_normals).sum(dim=1).clamp(-1.0, 1.0)
    angles_rad = torch.acos(torch.abs(dot_product))
    angles_deg = angles_rad * 180.0 / np.pi
    avg_angle_error = angles_deg.mean().item()
    rmse = torch.sqrt((angles_deg**2).mean()).item()
    return avg_angle_error, rmse


def calculate_normal_metrics(pred_normals, gt_normals):
    import torch.nn.functional as F

    pred_normalized = F.normalize(pred_normals, p=2, dim=1)
    gt_normalized = F.normalize(gt_normals, p=2, dim=1)
    dot_product = torch.clamp((pred_normalized * gt_normalized).sum(dim=1), -1.0, 1.0)
    angular_error_rad = torch.acos(torch.abs(dot_product))
    angular_error_deg = angular_error_rad * 180.0 / np.pi
    mean_error = angular_error_deg.mean().item()
    median_error = angular_error_deg.median().item()
    pct_5deg = (angular_error_deg < 5.0).float().mean().item() * 100
    pct_10deg = (angular_error_deg < 10.0).float().mean().item() * 100
    pct_30deg = (angular_error_deg < 30.0).float().mean().item() * 100
    rmse_unoriented = torch.sqrt((angular_error_deg**2).mean()).item()
    angular_error_oriented_rad = torch.acos(dot_product)
    angular_error_oriented_deg = angular_error_oriented_rad * 180.0 / np.pi
    needs_flip = (angular_error_oriented_deg > 90.0).float().mean().item() > 0.5
    if needs_flip:
        dot_product_flipped = torch.clamp(
            (-pred_normalized * gt_normalized).sum(dim=1), -1.0, 1.0
        )
        angular_error_oriented_rad = torch.acos(dot_product_flipped)
        angular_error_oriented_deg = angular_error_oriented_rad * 180.0 / np.pi
    rmse_oriented = torch.sqrt((angular_error_oriented_deg**2).mean()).item()
    return {
        "Mean_Error": mean_error,
        "Median_Error": median_error,
        "PGP5": pct_5deg,
        "PGP10": pct_10deg,
        "PGP30": pct_30deg,
        "RMSE_Unoriented": rmse_unoriented,
        "RMSE_Oriented": rmse_oriented,
    }


def calculate_edge_accuracy(A, B, gt_flip):
    if not isinstance(A, np.ndarray):
        A = A.cpu().numpy() if hasattr(A, "cpu") else np.array(A)
    if not isinstance(B, np.ndarray):
        B = B.cpu().numpy() if hasattr(B, "cpu") else np.array(B)
    if not isinstance(gt_flip, np.ndarray):
        gt_flip = (
            gt_flip.cpu().numpy() if hasattr(gt_flip, "cpu") else np.array(gt_flip)
        )
    sum_AB = A + B
    edge_mask = sum_AB > 0
    edge_mask = np.triu(edge_mask, k=1)
    edge_indices = np.where(edge_mask)
    if len(edge_indices[0]) == 0:
        return 0.0, 0
    i_indices = edge_indices[0]
    j_indices = edge_indices[1]
    gt_consistent = gt_flip[i_indices] == gt_flip[j_indices]
    pred_consistent = A[i_indices, j_indices] > B[i_indices, j_indices]
    correct_edges = (gt_consistent == pred_consistent).sum()
    total_edges = len(i_indices)
    edge_accuracy = (correct_edges / total_edges) * 100.0
    return edge_accuracy, total_edges
