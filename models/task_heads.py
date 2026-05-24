#!/usr/bin/env python
"""
Task-Specific Heads for Multi-Task Learning
Multi-task learning任务头模块 - 包含法向量估计头和其他任务头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalEstimationHead(nn.Module):
    """
    MLP head for normal vector estimation
    法向量估计头 - 从点云特征预测单位法向量

    Architecture:
        input_dim → hidden_dim → hidden_dim → 3 (normals)
        With LayerNorm, ReLU, and Dropout between layers

    Args:
        in_dim: Input feature dimension (e.g., 512 from PTv3 backbone)
        hidden_dim: Hidden layer dimension (default: 256)
        num_layers: Number of MLP layers (default: 3)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, in_dim=512, hidden_dim=256, num_layers=3, dropout=0.1):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Build MLP layers
        layers = []

        # First layer: in_dim → hidden_dim
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))

        # Middle layers: hidden_dim → hidden_dim
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))

        # Final layer: hidden_dim → 3 (normal vector)
        layers.append(nn.Linear(hidden_dim, 3))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features):
        """
        Forward pass

        Args:
            features: (N, in_dim) point features from backbone

        Returns:
            normals: (N, 3) normalized normal vectors (unit length)
        """
        # Pass through MLP
        normals = self.mlp(features)  # (N, 3)

        # L2 normalization to ensure unit length
        normals = F.normalize(normals, p=2, dim=-1)  # (N, 3)

        return normals

    def get_num_parameters(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FeatureFusionModule(nn.Module):
    """
    Feature fusion module for combining backbone features with predicted normals
    特征融合模块 - 将骨干网络特征与预测法向量融合

    This module concatenates backbone features with predicted normals and
    projects back to the original feature dimension.

    Args:
        backbone_dim: Backbone feature dimension (e.g., 512)
        normal_dim: Normal vector dimension (always 3)
        output_dim: Output feature dimension (usually same as backbone_dim)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, backbone_dim=512, normal_dim=3, output_dim=512, dropout=0.1):
        super().__init__()

        self.backbone_dim = backbone_dim
        self.normal_dim = normal_dim
        self.output_dim = output_dim

        # Fusion layer: [backbone_feat, normals] → output_dim
        self.fusion = nn.Sequential(
            nn.Linear(backbone_dim + normal_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, backbone_features, normals, detach_normals=True):
        """
        Fuse backbone features with predicted normals

        Args:
            backbone_features: (N, backbone_dim) features from PTv3 backbone
            normals: (N, 3) predicted normal vectors
            detach_normals: If True, detach normals to prevent gradient backprop
                          (default: True for training stability)

        Returns:
            fused_features: (N, output_dim) fused features
        """
        # Optionally detach normals to prevent gradient flow
        if detach_normals:
            normals = normals.detach()

        # Concatenate features
        concat_features = torch.cat([backbone_features, normals], dim=-1)  # (N, backbone_dim + 3)

        # Project to output dimension
        fused_features = self.fusion(concat_features)  # (N, output_dim)

        return fused_features

    def get_num_parameters(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

