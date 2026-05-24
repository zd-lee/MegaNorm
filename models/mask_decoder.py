"""
Mask Decoder for Interactive Point Cloud Segmentation
基于Two-Way Transformer的掩码解码器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Type
import math
from .transformer import TwoWayTransformer

try:
    from utils.model_debug import get_global_nan_detector
except ImportError:
    # Fallback if model_debug is not available (during inference)
    def get_global_nan_detector():
        return None


class QueryPointFusion(nn.Module):
    """
    Query-Point Feature Fusion

    使用交叉注意力机制将查询点的信息融合到每个点云点的特征中，
    从而使网络能够根据查询点的信息进行精确的分割决策。
    """

    def __init__(self,
                 embed_dim: int = 512,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        初始化特征融合模块

        Args:
            embed_dim: 特征维度
            num_heads: 注意力头数
            dropout: dropout概率
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 交叉注意力层：查询点 attend to 点云点
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.LayerNorm(embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        # Layer规范
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # 可选：查询特征投影
        self.query_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self,
                point_features: torch.Tensor,
                query_feature: torch.Tensor,
                offset: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            point_features: 点云特征 (N_total, embed_dim)
            query_feature: 查询点特征 (B, embed_dim) 或 (embed_dim,)
            offset: batch offset information (B,) - 用于区分不同点云

        Returns:
            enhanced_features: 融合后的点云特征 (N_total, embed_dim)
        """

        # 处理查询特征的维度
        if query_feature.dim() == 1:  # (embed_dim,) -> (1, embed_dim)
            query_feature = query_feature.unsqueeze(0)

        B = query_feature.shape[0]  # batch size

        if offset is not None:
            # 批量处理：每个batch中的点云都有对应的查询
            return self._batch_fusion(point_features, query_feature, offset)
        else:
            # 单个点云的情况
            return self._single_fusion(point_features, query_feature[0])

    def _single_fusion(self,
                      point_features: torch.Tensor,
                      query_feature: torch.Tensor) -> torch.Tensor:
        """
        单点云融合

        Args:
            point_features: (N, embed_dim)
            query_feature: (embed_dim,)

        Returns:
            enhanced_features: (N, embed_dim)
        """
        # Get NaN detector if available
        nan_detector = get_global_nan_detector()

        # Check inputs for NaN
        if nan_detector is not None:
            nan_detector.check_tensor_nan(
                point_features, "query_point_fusion.point_features",
                {'component': 'query_point_fusion', 'stage': 'input'}
            )
            nan_detector.check_tensor_nan(
                query_feature, "query_point_fusion.query_feature",
                {'component': 'query_point_fusion', 'stage': 'input'}
            )

        # 准备查询特征 (1, embed_dim)
        query = query_feature.unsqueeze(0)

        # 交叉注意力：query attend to points
        attn_output, _ = self.cross_attention(
            query=query,                    # (1, embed_dim)
            key=point_features, # (1, N, embed_dim)
            value=point_features # (1, N, embed_dim)
        )

        # Check attention output for NaN
        if nan_detector is not None:
            nan_detector.check_tensor_nan(
                attn_output, "query_point_fusion.attention_output",
                {'component': 'query_point_fusion', 'stage': 'after_attention'}
            )

        # 添加残差连接和层规范
        enhanced = self.norm1(point_features + attn_output.squeeze(0))  # (N, embed_dim)

        # Check after residual connection and normalization
        if nan_detector is not None:
            nan_detector.check_tensor_nan(
                enhanced, "query_point_fusion.residual_norm1",
                {'component': 'query_point_fusion', 'stage': 'after_norm1'}
            )

        # 前馈网络
        ff_output = self.feed_forward(enhanced)
        if nan_detector is not None:
            nan_detector.check_tensor_nan(
                ff_output, "query_point_fusion.feed_forward_output",
                {'component': 'query_point_fusion', 'stage': 'after_feed_forward'}
            )

        enhanced = self.norm2(enhanced + ff_output)

        # Final output check
        if nan_detector is not None:
            nan_detector.check_tensor_nan(
                enhanced, "query_point_fusion.final_output",
                {'component': 'query_point_fusion', 'stage': 'output'}
            )

        return enhanced

    def _batch_fusion(self,
                     point_features: torch.Tensor,
                     query_features: torch.Tensor,
                     offset: torch.Tensor) -> torch.Tensor:
        """
        批量融合 - 处理多个点云的融合

        Args:
            point_features: (N_total, embed_dim) 所有点云的特征
            query_features: (B, embed_dim) 对应每个batch的查询特征
            offset: (B,) 每个batch在总特征中的结束位置

        Returns:
            enhanced_features: (N_total, embed_dim)
        """
        enhanced_list = []
        start_idx = 0

        for i, end_idx in enumerate(offset):
            # 当前点云的特征
            curr_features = point_features[start_idx:end_idx]  # (Ni, embed_dim)
            curr_query = query_features[i]                     # (embed_dim,)

            # 单点云融合
            enhanced = self._single_fusion(curr_features, curr_query)
            enhanced_list.append(enhanced)

            start_idx = end_idx

        return torch.cat(enhanced_list, dim=0)

    def get_feature_dim(self) -> int:
        """获取输出特征维度"""
        return self.embed_dim


class MLP(nn.Module):
    """
    Multi-Layer Perceptron
    Used for hypernetworks and IoU prediction head
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x), inplace=True) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class MaskDecoder(nn.Module):
    """
    Mask Decoder using Two-Way Transformer

    This decoder uses a two-way transformer to process point cloud features
    and query embeddings to generate segmentation masks. Adapted for SegmentOpNet:
    1. Supports batch processing with offset tensors
    2. No dense prompt embeddings (mask inputs) - only query point embeddings
    3. No feature upscaling - PTv3 already provides per-point features

    Architecture:
        Point Features (N, D) + Query Features (B, D)
            ↓
        Learnable Tokens [IoU, Mask_1, ..., Mask_k] + Query
            ↓
        TwoWayTransformer (bidirectional attention)
            ↓
        Updated Point Features (N, D)
            ↓
        Hypernetworks generate prediction weights from mask tokens
            ↓
        Mask Logits: weights @ point_features^T
    """

    def __init__(
        self,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Initialize Mask Decoder

        Args:
            transformer_dim: Channel dimension for transformer
            transformer: TwoWayTransformer instance
            num_multimask_outputs: Number of mask predictions (for multi-mask output)
            iou_head_depth: Depth of IoU prediction MLP
            iou_head_hidden_dim: Hidden dimension of IoU prediction MLP
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        # self.num_multimask_outputs = num_multimask_outputs

        # Learnable tokens
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # Hypernetwork MLPs - transform mask tokens into prediction weights
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(transformer_dim, transformer_dim, transformer_dim, 3)
            for _ in range(self.num_mask_tokens)
        ])

        # IoU prediction head
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        point_features: torch.Tensor,
        point_pe: torch.Tensor,
        query_features: torch.Tensor,
        offset: Optional[torch.Tensor] = None,
        multimask_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            point_features: Point cloud features (N_total, transformer_dim)
            point_pe: Positional encoding for points (N_total, transformer_dim)
            query_features: Query point features (B, transformer_dim) or (transformer_dim,)
            offset: Batch offset information (B,) - marks end position of each batch
            multimask_output: Whether to return multiple masks or single mask

        Returns:
            masks: Segmentation logits (B, num_masks, N) or (num_masks, N)
            iou_pred: IoU prediction for each mask (B, num_masks) or (num_masks,)
        """
        # Get NaN detector if available
        nan_detector = get_global_nan_detector()

        # Check inputs for NaN
        if nan_detector is not None:
            nan_detector.check_tensor_nan(
                point_features, "mask_decoder.point_features",
                {'component': 'mask_decoder', 'stage': 'input'}
            )
            nan_detector.check_tensor_nan(
                point_pe, "mask_decoder.point_pe",
                {'component': 'mask_decoder', 'stage': 'input'}
            )
            nan_detector.check_tensor_nan(
                query_features, "mask_decoder.query_features",
                {'component': 'mask_decoder', 'stage': 'input'}
            )

        # Handle query feature dimensions
        if query_features.dim() == 1:  # (D,) -> (1, D)
            query_features = query_features.unsqueeze(0)

        B = query_features.shape[0]  # batch size

        if offset is not None:
            # Batch processing
            masks, iou_pred = self._batch_fusion(
                point_features, point_pe, query_features, offset, multimask_output
            )
        else:
            # Single point cloud
            masks, iou_pred = self._single_fusion(
                point_features, point_pe, query_features[0], multimask_output
            )
            # Add batch dimension for consistency
            masks = masks.unsqueeze(0)  # (1, num_masks, N)
            iou_pred = iou_pred.unsqueeze(0)  # (1, num_masks)

        # Check final outputs for NaN
        if nan_detector is not None:
            nan_detector.check_tensor_nan(
                masks, "mask_decoder.output_masks",
                {'component': 'mask_decoder', 'stage': 'output'}
            )
            nan_detector.check_tensor_nan(
                iou_pred, "mask_decoder.output_iou_pred",
                {'component': 'mask_decoder', 'stage': 'output'}
            )

        return masks, iou_pred

    def _single_fusion(
        self,
        point_features: torch.Tensor,
        point_pe: torch.Tensor,
        query_feature: torch.Tensor,
        multimask_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single point cloud fusion

        Args:
            point_features: (N, D)
            point_pe: (N, D)
            query_feature: (D,)
            multimask_output: Whether to output multiple masks

        Returns:
            masks: (num_masks, N)
            iou_pred: (num_masks,)
        """
        # Prepare learnable output tokens: [iou_token, mask_token_1, ..., mask_token_k]
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )  # (1 + num_mask_tokens, D)

        # Concatenate with query features
        tokens = torch.cat(
            [output_tokens, query_feature.unsqueeze(0)], dim=0
        )  # (1 + num_mask_tokens + 1, D)

        # Add batch dimension for transformer
        tokens = tokens.unsqueeze(0)  # (1, num_tokens, D)
        point_features_batch = point_features.unsqueeze(0)  # (1, N, D)
        point_pe_batch = point_pe.unsqueeze(0)  # (1, N, D)

        # Run two-way transformer
        # Returns: updated tokens and updated point features
        hs, src = self.transformer(
            point_features_batch,  # keys: point cloud features
            point_pe_batch,        # positional encoding for keys
            tokens                 # queries: learnable tokens + query embedding
        )
        # hs: (1, num_tokens, D) - updated tokens
        # src: (1, N, D) - updated point features

        # Extract token outputs
        iou_token_out = hs[:, 0, :]  # (1, D)
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]  # (1, num_mask_tokens, D)

        # Use updated point features directly (no upscaling needed)
        src = src.squeeze(0)  # (N, D)

        # Generate mask predictions using hypernetworks
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )  # Each: (1, D)
        hyper_in = torch.stack(hyper_in_list, dim=1)  # (1, num_mask_tokens, D)
        hyper_in = hyper_in.squeeze(0)  # (num_mask_tokens, D)

        # Compute mask logits: hypernetwork_output @ point_features^T
        masks = hyper_in @ src.transpose(-1, -2)  # (num_mask_tokens, N)

        # Generate IoU predictions
        iou_pred = self.iou_prediction_head(iou_token_out)  # (1, num_mask_tokens)
        iou_pred = iou_pred.squeeze(0)  # (num_mask_tokens,)

        # Select output masks based on multimask_output flag
        if multimask_output:
            mask_slice = slice(1, None)  # Use masks 1 to end
        else:
            mask_slice = slice(0, 1)  # Use only first mask

        masks = masks[mask_slice, :]  # (num_selected_masks, N)
        iou_pred = iou_pred[mask_slice]  # (num_selected_masks,)

        return masks, iou_pred

    def _batch_fusion(
        self,
        point_features: torch.Tensor,
        point_pe: torch.Tensor,
        query_features: torch.Tensor,
        offset: torch.Tensor,
        multimask_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch fusion - process multiple point clouds

        Args:
            point_features: (N_total, D) - all point features
            point_pe: (N_total, D) - all positional encodings
            query_features: (B, D) - query features for each batch
            offset: (B,) - end position of each batch in total features
            multimask_output: Whether to output multiple masks

        Returns:
            masks: List of masks for each batch, each (num_masks, N_i)
            iou_pred: (B, num_masks) - IoU predictions for each batch
        """
        masks_list = []
        iou_list = []
        start_idx = 0

        for i, end_idx in enumerate(offset):
            # Extract current point cloud
            curr_features = point_features[start_idx:end_idx]  # (N_i, D)
            curr_pe = point_pe[start_idx:end_idx]  # (N_i, D)
            curr_query = query_features[i]  # (D,)

            # Process single point cloud
            curr_masks, curr_iou = self._single_fusion(
                curr_features, curr_pe, curr_query, multimask_output
            )
            # curr_masks: (num_masks, N_i)
            # curr_iou: (num_masks,)

            masks_list.append(curr_masks)
            iou_list.append(curr_iou)

            start_idx = end_idx

        # Stack IoU predictions
        iou_pred = torch.stack(iou_list, dim=0)  # (B, num_masks)

        # Return masks as list (variable length) or concatenate along point dimension
        # For compatibility with loss computation, concatenate along point dimension
        masks = torch.cat(masks_list, dim=1)  # (num_masks, N_total)

        return masks, iou_pred

    def get_feature_dim(self) -> int:
        """Get output feature dimension"""
        return self.transformer_dim


class SimplifiedMaskDecoder(nn.Module):
    """
    Simplified Mask Decoder using Two-Way Transformer

    This decoder uses a two-way transformer to process point cloud features
    and query embeddings to generate a single segmentation mask. Adapted for SegmentOpNet:
    1. Supports batch processing with offset tensors
    2. No dense prompt embeddings (mask inputs) - only query point embeddings
    3. No feature upscaling - PTv3 already provides per-point features
    4. Single mask output - no multi-mask or IoU prediction

    Architecture:
        Point Features (N, D) + Query Features (B, D)
            ↓
        Learnable Tokens [Mask] + Query
            ↓
        TwoWayTransformer (bidirectional attention)
            ↓
        Updated Point Features (N, D)
            ↓
        Hypernetwork generates prediction weights from mask token
            ↓
        Mask Logits: weights @ point_features^T
    """

    def __init__(
        self,
        transformer_dim: int,
        transformer: nn.Module,
    ) -> None:
        """
        Initialize Simplified Mask Decoder

        Args:
            transformer_dim: Channel dimension for transformer
            transformer: TwoWayTransformer instance
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        # Single learnable mask token
        self.mask_token = nn.Embedding(1, transformer_dim)

        # Single hypernetwork MLP - transform mask token into prediction weights
        self.output_hypernetwork_mlp = MLP(transformer_dim, transformer_dim, transformer_dim, 3)

    def forward(
        self,
        point_features: torch.Tensor,
        point_pe: torch.Tensor,
        query_features: torch.Tensor,
        offset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            point_features: Point cloud features (N_total, transformer_dim)
            point_pe: Positional encoding for points (N_total, transformer_dim)
            query_features: Query point features (B, transformer_dim) or (transformer_dim,)
            offset: Batch offset information (B,) - marks end position of each batch

        Returns:
            masks: Segmentation logits (B, N) for single point cloud or (N_total,) for batch processing
        """
        # Get NaN detector if available
        nan_detector = get_global_nan_detector()

        # Check inputs for NaN
        if nan_detector is not None:
            nan_detector.check_tensor_nan(
                point_features, "simplified_mask_decoder.point_features",
                {'component': 'simplified_mask_decoder', 'stage': 'input'}
            )
            nan_detector.check_tensor_nan(
                point_pe, "simplified_mask_decoder.point_pe",
                {'component': 'simplified_mask_decoder', 'stage': 'input'}
            )
            nan_detector.check_tensor_nan(
                query_features, "simplified_mask_decoder.query_features",
                {'component': 'simplified_mask_decoder', 'stage': 'input'}
            )

        # Handle query feature dimensions
        if query_features.dim() == 1:  # (D,) -> (1, D)
            query_features = query_features.unsqueeze(0)

        B = query_features.shape[0]  # batch size

        if offset is not None:
            # Batch processing - returns concatenated masks (N_total,)
            masks = self._batch_fusion(
                point_features, point_pe, query_features, offset
            )
        else:
            # Single point cloud
            masks = self._single_fusion(
                point_features, point_pe, query_features[0]
            )
            # Add batch dimension for consistency
            masks = masks.unsqueeze(0)  # (1, N)

        # Check final outputs for NaN
        if nan_detector is not None:
            nan_detector.check_tensor_nan(
                masks, "simplified_mask_decoder.output_masks",
                {'component': 'simplified_mask_decoder', 'stage': 'output'}
            )

        return masks

    def _single_fusion(
        self,
        point_features: torch.Tensor,
        point_pe: torch.Tensor,
        query_feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single point cloud fusion

        Args:
            point_features: (N, D)
            point_pe: (N, D)
            query_feature: (D,)

        Returns:
            masks: (N,)
        """
        # Prepare learnable output tokens: [mask_token]
        mask_token = self.mask_token.weight  # (1, D)

        # Concatenate with query features
        tokens = torch.cat(
            [mask_token, query_feature.unsqueeze(0)], dim=0
        )  # (2, D)

        # Add batch dimension for transformer
        tokens = tokens.unsqueeze(0)  # (1, 2, D)
        point_features_batch = point_features.unsqueeze(0)  # (1, N, D)
        point_pe_batch = point_pe.unsqueeze(0)  # (1, N, D)

        # Run two-way transformer
        # Returns: updated tokens and updated point features
        hs, src = self.transformer(
            point_features_batch,  # keys: point cloud features
            point_pe_batch,        # positional encoding for keys
            tokens                 # queries: learnable tokens + query embedding
        )
        # hs: (1, 2, D) - updated tokens
        # src: (1, N, D) - updated point features

        # Extract mask token output
        mask_token_out = hs[:, 0, :]  # (1, D)

        # Use updated point features directly (no upscaling needed)
        src = src.squeeze(0)  # (N, D)

        # Generate mask prediction using hypernetwork
        hyper_in = self.output_hypernetwork_mlp(mask_token_out)  # (1, D)
        hyper_in = hyper_in.squeeze(0)  # (D,)

        # Compute mask logits: hypernetwork_output @ point_features^T
        masks = hyper_in @ src.transpose(-1, -2)  # (N,)

        return masks

    def _batch_fusion(
        self,
        point_features: torch.Tensor,
        point_pe: torch.Tensor,
        query_features: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batch fusion - process multiple point clouds

        Args:
            point_features: (N_total, D) - all point features
            point_pe: (N_total, D) - all positional encodings
            query_features: (B, D) - query features for each batch
            offset: (B,) - end position of each batch in total features

        Returns:
            masks: (num_masks, N_total) - mask predictions concatenated along point dimension
        """
        masks_list = []
        start_idx = 0

        for i, end_idx in enumerate(offset):
            # Extract current point cloud
            curr_features = point_features[start_idx:end_idx]  # (N_i, D)
            curr_pe = point_pe[start_idx:end_idx]  # (N_i, D)
            curr_query = query_features[i]  # (D,)

            # Process single point cloud
            curr_masks = self._single_fusion(
                curr_features, curr_pe, curr_query
            )
            # curr_masks: (N_i,)

            masks_list.append(curr_masks)
            start_idx = end_idx

        # Return masks as list (variable length) or concatenate along point dimension
        # For compatibility with loss computation, concatenate along point dimension
        masks = torch.cat(masks_list, dim=0)  # (N_total,)

        return masks

    def get_feature_dim(self) -> int:
        """Get output feature dimension"""
        return self.transformer_dim

