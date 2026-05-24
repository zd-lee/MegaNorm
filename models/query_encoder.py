"""
Query Encoder for Interactive Point Cloud Segmentation
查询点编码器 - 将查询点的位置和类别信息编码为特征向量
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union

try:
    from utils.model_debug import get_global_nan_detector
except ImportError:
    # Fallback if model_debug is not available (during inference)
    def get_global_nan_detector():
        return None

# https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/prompt_encoder.py
class OrientedPointEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, position_dim=6,num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((position_dim, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [-1,1]."""
        # assuming coords are in [-1, 1] and have d_1 x ... x d_n x D shape
        coords = coords @ self.positional_encoding_gaussian_matrix
        # TODO: Why using 2 * np.pi?
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: shape (..., coord_dim), normalized coordinates in [-1, 1].

        Returns:
            torch.Tensor: shape (..., num_pos_feats), positional encoding.
        """
        if (coords < -1 - 1e-6).any() or (coords > 1 + 1e-6).any():
            print("Bounds: ", (coords.min(), coords.max()))
            raise ValueError(f"Input coordinates must be normalized to [-1, 1].")
        # TODO: whether to convert to float?
        return self._pe_encoding(coords)
    
# https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/prompt_encoder.py
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((3, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [-1,1]."""
        # assuming coords are in [-1, 1] and have d_1 x ... x d_n x D shape
        coords = coords @ self.positional_encoding_gaussian_matrix
        # TODO: Why using 2 * np.pi?
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: shape (..., coord_dim), normalized coordinates in [-1, 1].

        Returns:
            torch.Tensor: shape (..., num_pos_feats), positional encoding.
        """
        if (coords < -1 - 1e-6).any() or (coords > 1 + 1e-6).any():
            print("Bounds: ", (coords.min(), coords.max()))
            raise ValueError(f"Input coordinates must be normalized to [-1, 1].")
        # TODO: whether to convert to float?
        return self._pe_encoding(coords)


class QueryEncoder(nn.Module):
    """
    Query Point Encoder

    将查询点的位置 (x,y,z) 和类别标签编码为特征向量，
    用于指导点云分割任务的网络决策。
    """

    def __init__(self,
                 embed_dim: int = 256,
                 feature_dim: int = 6):
        """
        初始化查询编码器

        Args:
            embed_dim: 嵌入维度
            num_classes: 类别数量
            feature_dim: 位置维度 (通常为3: x,y,z)
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        # 位置编码器：将3D坐标映射到特征空间
        self.encoder = OrientedPointEmbeddingRandom(feature_dim,embed_dim//2)


    def forward(self, query_data: dict) -> torch.Tensor:
        # Extract features (could be 3D position only or 6D position + normal)
        feature = query_data['feat']

        # Get NaN detector if available
        nan_detector = get_global_nan_detector()

        # Check input features for NaN
        if nan_detector is not None:
            nan_detector.check_tensor_nan(
                feature, "query_encoder.input_features",
                {'component': 'query_encoder', 'stage': 'input'}
            )

        # Handle both cases: if feature_dim is 3 (position only) or 6 (position + normal)
        if feature.shape[-1] == 6 and self.feature_dim == 3:
            # Query data has 6D features (pos + normal), but encoder expects 3D (position only)
            if feature.dim() == 1:
                # Single query point: (6,) -> extract first 3 elements
                pos_only = feature[:3]
            else:
                # Batch of query points: (B, 6) -> extract first 3 columns
                pos_only = feature[:, :3]
            pos_encoded = self.encoder(pos_only)  # (B, embed_dim)
        else:
            # Feature dimension matches encoder configuration
            pos_encoded = self.encoder(feature)  # (B, embed_dim)

        # Check positional encoding output for NaN
        if nan_detector is not None:
            nan_detector.check_tensor_nan(
                pos_encoded, "query_encoder.positional_encoding",
                {'component': 'query_encoder', 'stage': 'output'}
            )

        return pos_encoded
