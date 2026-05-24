"""
Global Flip Transformer Model
使用Transformer预测每个patch的flip决策
"""

import torch
import torch.nn as nn
from typing import Dict


class GlobalFlipTransformer(nn.Module):
    def __init__(self,
                 feature_dim: int = 512,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        super().__init__()

        self.feature_dim = feature_dim
        self.d_model = d_model

        # 特征投影
        self.feature_proj = nn.Linear(feature_dim, d_model)

        # 位置编码（MLP编码patch center）
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, features: torch.Tensor, patch_centers: torch.Tensor,
                batch_offsets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (N_total, 512) patch特征
            patch_centers: (N_total, 3) patch中心坐标
            batch_offsets: (B,) 累积偏移量 [end1, end2, ..., endB]

        Returns:
            logits: (N_total, 1) 每个patch的flip预测logits
        """
        # 投影特征
        x = self.feature_proj(features)  # (N_total, d_model)

        # 添加位置编码
        pos_embed = self.pos_encoder(patch_centers)  # (N_total, d_model)
        x = x + pos_embed

        # 处理batch（分别处理每个样本）
        outputs = []
        start_idx = 0
        for end_idx in batch_offsets:
            # 单个样本的patches
            sample_x = x[start_idx:end_idx].unsqueeze(0)  # (1, P, d_model)

            # Transformer
            out = self.transformer(sample_x)  # (1, P, d_model)

            # 分类
            logits = self.classifier(out.squeeze(0))  # (P, 1)
            outputs.append(logits)

            start_idx = end_idx

        # 拼接所有样本
        all_logits = torch.cat(outputs, dim=0)  # (N_total, 1)
        return all_logits


def create_global_flip_transformer(config: Dict) -> GlobalFlipTransformer:
    """从配置创建模型"""
    model_config = config['model']
    return GlobalFlipTransformer(
        feature_dim=model_config.get('feature_dim', 512),
        d_model=model_config.get('d_model', 256),
        nhead=model_config.get('nhead', 8),
        num_layers=model_config.get('num_layers', 4),
        dim_feedforward=model_config.get('dim_feedforward', 1024),
        dropout=model_config.get('dropout', 0.1)
    )
