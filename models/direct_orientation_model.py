"""
Direct Orientation Prediction Model
直接预测点云方向的模型 - 使用PTv3 + MLP进行二分类
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

from .ptv3_backbone import PTv3Backbone


class DirectOrientationModel(nn.Module):
    """
    Direct Orientation Prediction Model

    A simplified model for direct per-point binary orientation prediction.
    Architecture: PTv3 Backbone → MLP Head → Binary Classification

    No query points, no transformers, no mask decoder - just direct prediction.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Direct Orientation Model

        Args:
            config: Configuration dictionary containing:
                - backbone: PTv3 backbone config
                - mlp_head: MLP head config (hidden_dim, num_layers, dropout)
        """
        super().__init__()

        self.config = config

        # 1. PTv3 Backbone for feature extraction
        backbone_config = config['backbone']
        self.backbone = PTv3Backbone(**backbone_config)

        # Get backbone output dimension
        self.backbone_dim = self.backbone.get_feature_dim()

        # 2. MLP Segmentation Head for binary classification
        mlp_config = config.get('mlp_head', {})
        hidden_dim = mlp_config.get('hidden_dim', 256)
        num_layers = mlp_config.get('num_layers', 3)
        dropout = mlp_config.get('dropout', 0.1)
        num_classes = 1  # Binary classification (flip or not flip)

        # Build MLP layers
        layers = []
        in_dim = self.backbone_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        # Final classification layer
        layers.append(nn.Linear(in_dim, num_classes))

        self.mlp_head = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize MLP head weights"""
        for m in self.mlp_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, point_cloud_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass

        Args:
            point_cloud_data: Point cloud data dictionary containing:
                - 'feat': Point features (N_total, C)
                - 'coord': Point coordinates (N_total, 3)
                - 'batch': Batch indices (N_total,)
                - 'grid_size': Grid size for PTv3 serialization

        Returns:
            logits: Per-point classification logits (N_total, 2)
        """
        # 1. Extract features from PTv3 backbone
        backbone_output = self.backbone(point_cloud_data)
        point_features = backbone_output['feat']  # (N_total, backbone_dim)

        # 2. Apply MLP head for binary classification
        logits = self.mlp_head(point_features)  # (N_total, 2)

        return logits

    def predict(self, point_cloud_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Predict binary flip status

        Args:
            point_cloud_data: Point cloud data

        Returns:
            predictions: Binary predictions (N_total,) where 0=no flip, 1=flip
        """
        logits = self.forward(point_cloud_data)
        predictions = torch.argmax(logits, dim=1)  # (N_total,)
        return predictions

    def predict_proba(self, point_cloud_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Predict flip probabilities

        Args:
            point_cloud_data: Point cloud data

        Returns:
            probs: Probabilities for each class (N_total, 2)
        """
        logits = self.forward(point_cloud_data)
        probs = torch.softmax(logits, dim=1)  # (N_total, 2)
        return probs

    def freeze_backbone(self, freeze: bool = True):
        """Freeze/unfreeze backbone parameters"""
        self.backbone.freeze_backbone(freeze)

    def load_backbone_weights(self, checkpoint_path: str):
        """Load pretrained backbone weights"""
        self.backbone.load_pretrained_weights(checkpoint_path)

    def get_num_parameters(self) -> str:
        """Get number of model parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        def format_number(num):
            if num >= 1e6:
                return f"{num/1e6:.1f}M"
            elif num >= 1e3:
                return f"{num/1e3:.1f}K"
            else:
                return str(num)

        return f"{format_number(total_params)} ({format_number(trainable_params)} trainable)"

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'backbone_dim': self.backbone_dim,
            'mlp_config': self.config.get('mlp_head', {}),
            'backbone_config': self.backbone.get_config(),
            'total_params': self.get_num_parameters(),
            'device': next(self.parameters()).device
        }


def create_direct_orientation_model(config: Dict[str, Any]) -> DirectOrientationModel:
    """
    Model factory function

    Args:
        config: Model configuration dictionary

    Returns:
        Initialized model instance
    """
    return DirectOrientationModel(config)
