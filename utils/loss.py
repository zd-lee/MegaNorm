#!/usr/bin/env python
"""
Loss functions for point cloud segmentation
"""

from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss


# def modify_target_by_input(inputs, targets, offset):

class BiFocalLoss(nn.Module):
    """
    BiFocal Loss for addressing class imbalance
    """
    def __init__(self, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    def _focal_loss(self, inputs, targets, alpha=None, gamma=None):
        """
        Focal loss for addressing class imbalance

        Args:
            inputs: (N,) mask logits
            targets: (N,) binary targets (0 or 1)
            alpha: Focal loss balancing factor (default: self.focal_alpha)
            gamma: Focal loss focusing parameter (default: self.focal_gamma)

        Returns:
            loss: Scalar focal loss
        """
        alpha = alpha if alpha is not None else self.focal_alpha
        gamma = gamma if gamma is not None else self.focal_gamma

        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()
    
    def _bifocalloss(self, inputs, targets):
        """
        Forward pass for BiFocal Loss
        """
        t1 = targets.detach().clone()
        t2 = targets.detach().clone()
        t2 = 1 - t2

        bce_loss1 = self._focal_loss(inputs,t1,self.focal_alpha,self.focal_gamma)
        bce_loss2 = self._focal_loss(inputs,t2,self.focal_alpha,self.focal_gamma)
        return min(bce_loss1,bce_loss2)

    def forward(self,masks,iou_pred,targets, offset=None):
        losses = {}
        total_loss = 0
        
        if masks.dim() == 2:
            mask = masks[0]
        else:
            mask = masks

        if offset is not None:
            start_idx = 0
            total_num = offset[-1]
            for i, end_idx in enumerate(offset):
                weight = (float(end_idx-start_idx) / total_num)
                pred = mask[start_idx:end_idx]
                target = targets[start_idx:end_idx]
                loss = self._bifocalloss(pred, target)
                total_loss += weight * loss
                start_idx = end_idx
        else:
            pred = mask
            target = targets[0]
            loss = self._bifocalloss(pred, target)
            total_loss += loss

        losses['total_loss'] = total_loss
        return total_loss, losses

class OrientedNormalLoss(nn.Module):
    """
    Oriented Normal Loss
    Loss = 1 - dot(normalize(pred), gt)

    Considers normal direction - penalizes both orientation and angle errors.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_normals, gt_normals, offset=None):
        """
        Forward pass

        Args:
            pred_normals: (N_total, 3) predicted normals (already normalized)
            gt_normals: (N_total, 3) ground truth normals
            offset: Optional (B,) offset tensor for batch processing

        Returns:
            total_loss: Scalar loss value
            losses: Dictionary of loss components
        """
        losses = {}

        # Normalize ground truth normals
        gt_normals_normalized = nn.functional.normalize(gt_normals, p=2, dim=1)

        # Compute dot product (normal consistency)
        dot_product = (pred_normals * gt_normals_normalized).sum(dim=1)  # (N_total,)

        # Loss = 1 - dot_product
        pointwise_loss = 1.0 - dot_product  # (N_total,)

        if offset is not None:
            # Weighted average by sample size
            start_idx = 0
            total_loss = 0.0
            total_num = offset[-1].item()

            for i, end_idx in enumerate(offset):
                sample_loss = pointwise_loss[start_idx:end_idx].mean()
                weight = float(end_idx - start_idx) / total_num
                total_loss += weight * sample_loss
                start_idx = end_idx
        else:
            total_loss = pointwise_loss.mean()

        losses['oriented_loss'] = total_loss
        losses['total_loss'] = total_loss

        return total_loss, losses
class MSELoss(nn.Module):
    """
    Mean Squared Error Loss
    """
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target, offset=None):
        """
        Forward pass

        Args:
            pred: Predicted values
            target: Ground truth values

        Returns:
            loss: Scalar MSE loss
        """
        loss = self.mse_loss(pred, target)
        return loss

class SinSimilarityLoss(nn.Module):
    """
    Sine Similarity Loss for normal vectors
    Loss = sin(angle(pred, gt)) = ||pred x gt|| / (||pred|| * ||gt||)

    Penalizes angular difference between predicted and ground truth normals.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_normals, gt_normals, offset=None):
        """
        Forward pass

        Args:
            pred_normals: (N_total, 3) predicted normals (already normalized)
            gt_normals: (N_total, 3) ground truth normals
            offset: Optional (B,) offset tensor for batch processing

        Returns:
            total_loss: Scalar loss value
            losses: Dictionary of loss components
        """
        losses = {}

        # Normalize ground truth normals
        gt_normals_normalized = nn.functional.normalize(gt_normals, p=2, dim=1)

        # Compute cross product
        cross_product = torch.cross(pred_normals, gt_normals_normalized, dim=1)  # (N_total, 3)
        sin_similarity = torch.norm(cross_product, dim=1)  # (N_total,)
        total_loss = sin_similarity.mean()

        losses['sin_similarity_loss'] = total_loss
        losses['total_loss'] = total_loss

        return total_loss, losses

class InLineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred_normals, gt_normals, offset=None):
        """
        Forward pass

        Args:
            pred_normals: (N_total, 3) predicted normals (already normalized)
            gt_normals: (N_total, 3) ground truth normals

        Returns:
            loss: Scalar L1 loss
        """
        sign = torch.sign((pred_normals * gt_normals).sum(dim=1, keepdim=True))
        loss = self.l1_loss(pred_normals, gt_normals * sign)
        return loss

class EikonalLoss(nn.Module):
    """
    Eikonal Loss for normal vectors
    Loss = mean(||pred_normals|| - 1)^2
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_normals, gt_normals, offset=None):
        """
        Forward pass

        Args:
            pred_normals: (N_total, 3) predicted normals

        Returns:
            loss: Scalar Eikonal loss
        """
        norms = torch.norm(pred_normals, dim=1)  # (N_total,)
        loss = torch.mean((norms - 1.0) ** 2)
        return loss

class UnorientedNormalLoss(nn.Module):
    """
    Unoriented Normal Loss
    Loss = 1 - |dot(normalize(pred), gt)|

    Ignores normal direction - only penalizes angle errors.
    """
    def __init__(self):
        super().__init__()
        inline_loss_fn = InLineLoss()
        eikonal_loss_fn = EikonalLoss()

    def forward(self, pred_normals, gt_normals, offset=None):
        """
        Forward pass

        Args:
            pred_normals: (N_total, 3) predicted normals (already normalized)
            gt_normals: (N_total, 3) ground truth normals
            offset: Optional (B,) offset tensor for batch processing

        Returns:
            total_loss: Scalar loss value
            losses: Dictionary of loss components
        """
        losses = {}

        # Normalize ground truth normals
        gt_normals_normalized = nn.functional.normalize(gt_normals, p=2, dim=1)

        # Compute dot product (normal consistency)
        dot_product = (pred_normals * gt_normals_normalized).sum(dim=1)  # (N_total,)

        # Loss = 1 - |dot_product|
        cos_loss = 1.0 - torch.abs(dot_product)  # (N_total,)
        inline_loss = InLineLoss()(pred_normals, gt_normals_normalized, offset)
        eikonal_loss = EikonalLoss()(pred_normals, gt_normals_normalized, offset)
        
        pointwise_loss = cos_loss + inline_loss + eikonal_loss  # (N_total,)
        total_loss = pointwise_loss.mean()
   
        losses['unoriented_loss'] = total_loss
        losses['total_loss'] = total_loss

        return total_loss, losses


class CombinedNormalLoss(nn.Module):
    """
    Combined Normal Loss
    Loss = oriented_weight * OrientedLoss + unoriented_weight * UnorientedLoss

    Weighted combination of oriented and unoriented losses.
    """
    def __init__(self, oriented_weight=0.5, unoriented_weight=0.5):
        """
        Args:
            oriented_weight: Weight for oriented loss
            unoriented_weight: Weight for unoriented loss
        """
        super().__init__()
        self.oriented_weight = oriented_weight
        self.unoriented_weight = unoriented_weight

        self.oriented_loss = OrientedNormalLoss()
        self.unoriented_loss = UnorientedNormalLoss()

    def forward(self, pred_normals, gt_normals, offset=None):
        """
        Forward pass

        Args:
            pred_normals: (N_total, 3) predicted normals (already normalized)
            gt_normals: (N_total, 3) ground truth normals
            offset: Optional (B,) offset tensor for batch processing

        Returns:
            total_loss: Scalar loss value
            losses: Dictionary of loss components
        """
        # Compute both losses
        oriented_loss, oriented_dict = self.oriented_loss(pred_normals, gt_normals, offset)
        unoriented_loss, unoriented_dict = self.unoriented_loss(pred_normals, gt_normals, offset)

        # Combine losses
        total_loss = (self.oriented_weight * oriented_loss +
                     self.unoriented_weight * unoriented_loss)

        losses = {
            'oriented_loss': oriented_loss,
            'unoriented_loss': unoriented_loss,
            'total_loss': total_loss
        }

        return total_loss, losses


class MSESinLoss(nn.Module):
    """
    MSE + Sine Similarity Loss for normal estimation
    Loss = mse_weight * MSE(pred, gt) + sin_weight * SinSimilarity(pred, gt)
    """
    def __init__(self, mse_weight=1.0, sin_weight=1.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.sin_weight = sin_weight
        self.sin_loss = SinSimilarityLoss()

    def forward(self, pred_normals, gt_normals, offset=None):
        """
        Forward pass

        Args:
            pred_normals: (N_total, 3) predicted normals (already normalized)
            gt_normals: (N_total, 3) ground truth normals
            offset: Optional (B,) offset tensor for batch processing

        Returns:
            total_loss: Scalar loss value
            losses: Dictionary of loss components
        """
        losses = {}

        # Normalize ground truth normals
        gt_normals_normalized = nn.functional.normalize(gt_normals, p=2, dim=1)

        # MSE loss
        mse_loss = F.mse_loss(pred_normals, gt_normals_normalized)

        # Sine similarity loss
        sin_loss, _ = self.sin_loss(pred_normals, gt_normals_normalized, offset)

        # Combined loss
        total_loss = self.mse_weight * mse_loss + self.sin_weight * sin_loss

        losses['mse_loss'] = mse_loss
        losses['sin_loss'] = sin_loss
        losses['total_loss'] = total_loss

        return total_loss, losses


# ============= Decoupled Loss Architecture =============

class BaseWeightedLoss(nn.Module):
    """Base class for losses with offset-based weighted averaging"""
    def forward(self, masks, iou_pred, targets, offset=None):
        mask = masks[0] if masks.dim() == 2 else masks

        if offset is None:
            loss = self.compute_sample_loss(mask, targets)
            return loss, {'total_loss': loss}

        total_loss = 0.0
        start_idx = 0
        total_num = offset[-1].item()

        for end_idx in offset:
            weight = float(end_idx - start_idx) / total_num
            loss = self.compute_sample_loss(mask[start_idx:end_idx], targets[start_idx:end_idx])
            total_loss += weight * loss
            start_idx = end_idx

        return total_loss, {'total_loss': total_loss}

    def compute_sample_loss(self, logits, targets):
        raise NotImplementedError


class FocalLoss(BaseWeightedLoss):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def compute_sample_loss(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * bce_loss).mean()


class BCELoss(BaseWeightedLoss):
    def compute_sample_loss(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets)


class DiceLoss(BaseWeightedLoss):
    def compute_sample_loss(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        return 1 - (2 * intersection + 1) / (union + 1)


class FlipAwareLoss(BaseWeightedLoss):
    """Wrapper that tries both orientations and picks minimum loss"""
    def __init__(self, base_loss):
        super().__init__()
        self.base_loss = base_loss

    def compute_sample_loss(self, logits, targets):
        loss1 = self.base_loss.compute_sample_loss(logits, targets)
        loss2 = self.base_loss.compute_sample_loss(logits, 1 - targets)
        return min(loss1, loss2)


def create_loss_function(config):
    """
    Factory function to create loss function based on config

    Args:
        config: Configuration dictionary

    Returns:
        Loss function instance (nn.Module)
    """
    loss_type = config['training']['loss'].get('type')

    # Decoupled losses
    if loss_type == 'focal':
        return FocalLoss(
            alpha=config['training']['loss'].get('alpha', 0.25),
            gamma=config['training']['loss'].get('gamma', 2.0)
        )
    elif loss_type == 'flip_focal':
        base = FocalLoss(
            alpha=config['training']['loss'].get('alpha', 0.25),
            gamma=config['training']['loss'].get('gamma', 2.0)
        )
        return FlipAwareLoss(base)
    elif loss_type == 'bce':
        return BCELoss()
    elif loss_type == 'flip_bce':
        return FlipAwareLoss(BCELoss())
    elif loss_type == 'dice':
        return DiceLoss()
    elif loss_type == 'flip_dice':
        return FlipAwareLoss(DiceLoss())
    elif loss_type == "bifocalloss":
        return BiFocalLoss(focal_alpha=config['training']['loss'].get('alpha', 0.25),
            focal_gamma=config['training']['loss'].get('gamma', 2.0))
    elif loss_type == 'oriented_normal':
        return OrientedNormalLoss()
    elif loss_type == 'unoriented_normal':
        return UnorientedNormalLoss()
    elif loss_type == 'combined_normal':
        return CombinedNormalLoss(
            oriented_weight=config['training']['loss'].get('oriented_weight', 0.5),
            unoriented_weight=config['training']['loss'].get('unoriented_weight', 0.5)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
