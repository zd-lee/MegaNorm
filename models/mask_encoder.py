"""
Simple Mask Encoder for Two-Stage Point Cloud Segmentation
使用纯卷积(无注意力)的轻量级mask编码器
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from .ptv3_origin import Point, PointModule, PointSequential
import spconv.pytorch as spconv


class ConvBlock(PointModule):
    """卷积块 (无注意力机制)"""
    
    def __init__(self, in_channels, out_channels, indice_key=None):
        super().__init__()
        
        self.conv = PointSequential(
            spconv.SubMConv3d(
                in_channels, out_channels,
                kernel_size=3, padding=1, bias=False,
                indice_key=indice_key
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, point: Point):
        return self.conv(point)


class SimpleMaskEncoder(PointModule):
    """
    Simple Mask Encoder using only Convolutions (No Attention)
    
    使用纯卷积的U-Net结构编码mask，不使用注意力机制。
    更加轻量快速。
    """
    
    def __init__(self,
                 in_channels: int = 32,
                 base_channels: int = 32,
                 num_layers: int = 3,
                 embed_dim: int = 512):
        """
        Args:
            in_channels: 输入特征维度
            base_channels: 基础通道数  
            num_layers: 下采样层数
            embed_dim: 输出嵌入维度 (需与pc_embedding一致)
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Input embedding
        self.input_embed = PointSequential(
            spconv.SubMConv3d(
                in_channels, base_channels,
                kernel_size=5, padding=2, bias=False,
                indice_key='stem'
            ),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Encoder
        self.encoder = nn.ModuleList()
        for i in range(num_layers):
            in_ch = base_channels * (2 ** i)
            out_ch = base_channels * (2 ** (i + 1))
            
            self.encoder.append(nn.ModuleList([
                ConvBlock(in_ch, in_ch, indice_key=f'enc{i}'),
                PointSequential(
                    spconv.SparseConv3d(
                        in_ch, out_ch,
                        kernel_size=3, stride=2, padding=1, bias=False,
                        indice_key=f'down{i}'
                    ),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True)
                )
            ]))
        
        # Bottleneck
        bottleneck_ch = base_channels * (2 ** num_layers)
        self.bottleneck = ConvBlock(bottleneck_ch, bottleneck_ch, indice_key='bottleneck')
        
        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(num_layers - 1, -1, -1):
            in_ch = base_channels * (2 ** (i + 1))
            out_ch = base_channels * (2 ** i)
            
            self.decoder.append(nn.ModuleList([
                PointSequential(
                    spconv.SparseInverseConv3d(
                        in_ch, out_ch,
                        kernel_size=3, bias=False,
                        indice_key=f'down{i}'
                    ),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True)
                ),
                ConvBlock(out_ch, out_ch, indice_key=f'dec{i}')
            ]))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(base_channels, embed_dim // 2),
            nn.BatchNorm1d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )
    
    def forward(self, point_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """前向传播"""
        point = Point(point_dict)
        point.sparsify()
        
        # Input
        point = self.input_embed(point)
        
        # Encoder
        for conv, down in self.encoder:
            point = conv(point)
            point = down(point)
        
        # Bottleneck
        point = self.bottleneck(point)
        
        # Decoder
        for up, conv in self.decoder:
            point = up(point)
            point = conv(point)
        
        # Project to embed_dim
        mask_embedding = self.output_proj(point.feat)
        
        return {
            'mask_embedding': mask_embedding,
            'offset': point.offset,
            'batch_size': point.batch.max().item() + 1
        }
    
    def get_feature_dim(self) -> int:
        return self.embed_dim


class ConfidenceDecoder(nn.Module):
    """
    Confidence Decoder - 融合三种embedding预测confidence
    
    优化版本: 使用单次attention
    conf = attention(pc_emb + mask_emb, query_emb, query_emb)
    复杂度: O(N) 而不是 O(N²)
    """
    
    def __init__(self, embed_dim: int = 512, hidden_dim: int = 256,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # 单个cross attention: (pc+mask) attend to query
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, pc_embedding: torch.Tensor, mask_embedding: torch.Tensor,
                query_embedding: torch.Tensor, offset: torch.Tensor = None) -> torch.Tensor:
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        
        if offset is not None:
            return self._batch_forward(pc_embedding, mask_embedding, query_embedding, offset)
        else:
            return self._single_forward(pc_embedding, mask_embedding, query_embedding)
    
    def _single_forward(self, pc_embedding, mask_embedding, query_embedding):
        """
        单样本前向传播
        
        新设计: conf = attention(pc_emb + mask_emb, query_emb, query_emb)
        - 先融合pc和mask (element-wise addition)
        - 然后用单次attention与query交互
        - 复杂度: O(N) 而不是 O(N²)
        """
        # 融合pc和mask embedding
        fused_emb = pc_embedding + mask_embedding  # (N, 512)
        fused_emb = fused_emb.unsqueeze(0)  # (1, N, 512)
        
        # 单次cross attention: fused attend to query
        # Q: fused_emb (1, N, 512)
        # K,V: query (1, 1, 512)
        # Attention矩阵: (1, N, 1) ← 轻量级!
        attn_output, _ = self.cross_attention(
            query=fused_emb,
            key=query_embedding.unsqueeze(0),
            value=query_embedding.unsqueeze(0)
        )
        
        # Residual + norm
        enhanced = self.norm1(fused_emb + attn_output)  # (1, N, 512)
        
        # Feed-forward
        enhanced = enhanced.squeeze(0)  # (N, 512)
        ffn_output = self.ffn(enhanced)  # (N, 256)
        ffn_output = self.norm2(ffn_output)
        
        # Confidence prediction
        confidence = self.confidence_head(ffn_output)  # (N, 1)
        
        return confidence
    
    def _batch_forward(self, pc_embedding, mask_embedding, query_embedding, offset):
        confidence_list = []
        start_idx = 0
        
        for i, end_idx in enumerate(offset):
            curr_pc = pc_embedding[start_idx:end_idx]
            curr_mask = mask_embedding[start_idx:end_idx]
            curr_query = query_embedding[i:i+1]
            
            curr_confidence = self._single_forward(curr_pc, curr_mask, curr_query)
            confidence_list.append(curr_confidence)
            
            start_idx = end_idx
        
        return torch.cat(confidence_list, dim=0)