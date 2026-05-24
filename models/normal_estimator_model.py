"""
Normal Estimation Model
法向量估计模型 - 使用PTv3 + MLP进行法向量回归
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from .ptv3_backbone import PTv3Backbone


class NormalEstimatorModel(nn.Module):
    """
    Normal Estimation Model

    A model for per-point normal vector estimation.
    Architecture: PTv3 Backbone → MLP Head → 3D Normal Vectors

    No query points - direct per-point normal prediction.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Normal Estimator Model

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

        # 2. MLP Head for normal regression
        mlp_config = config.get('mlp_head', {})
        hidden_dim = mlp_config.get('hidden_dim', 256)
        num_layers = mlp_config.get('num_layers', 3)
        dropout = mlp_config.get('dropout', 0.1)

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

        # Final regression layer to 3D normals
        layers.append(nn.Linear(in_dim, 3))

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
            normals: Per-point normalized normal vectors (N_total, 3)
        """
        # 1. Extract features from PTv3 backbone
        backbone_output = self.backbone(point_cloud_data)
        point_features = backbone_output['feat']  # (N_total, backbone_dim)

        # 2. Apply MLP head for normal regression
        normals_raw = self.mlp_head(point_features)  # (N_total, 3)

        # 3. L2 normalize to unit vectors
        normals = nn.functional.normalize(normals_raw, p=2, dim=1)  # (N_total, 3)

        return normals

    def predict(self, point_cloud_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Predict normal vectors

        Args:
            point_cloud_data: Point cloud data

        Returns:
            normals: Normalized normal vectors (N_total, 3)
        """
        return self.forward(point_cloud_data)

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


def create_normal_estimator_model(config: Dict[str, Any]) -> NormalEstimatorModel:
    """
    Model factory function

    Args:
        config: Model configuration dictionary

    Returns:
        Initialized model instance
    """
    return NormalEstimatorModel(config)
