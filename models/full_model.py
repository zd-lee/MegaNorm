"""
Complete Point Cloud Segmentation Model with PTv3 Backbone
完整点云分割模型 - 集成 PTv3 backbone + 查询驱动机制
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Union

from .ptv3_backbone import PTv3Backbone
from .query_encoder import QueryEncoder, PositionEmbeddingRandom
from .mask_decoder import MaskDecoder, SimplifiedMaskDecoder
from .transformer import TwoWayTransformer


class PTv3PointCloudSegmentation(nn.Module):
    """
    Point Transformer V3 Point Cloud Segmentation Model

    一个完整的查询驱动点云分割模型，使用 PTv3 作为骨干网络，
    支持交互式分割：给定点云和查询点，预测与查询点类别相同的mask。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化完整分割模型

        Args:
            config: 配置字典，包含所有组件的参数
        """
        super().__init__()

        self.config = config

        # 1. PTv3 Backbone
        backbone_config = config['backbone']
        self.backbone = PTv3Backbone(**backbone_config)

        # 2. Positional Embedding Layer
        embed_dim = config['embed_dim']
        self.pos_embd_layer = PositionEmbeddingRandom(
            num_pos_feats=embed_dim // 2,
            scale=config['pos_embd_scale']
        )

        # query_encoder
        if config.get('use_query_encoder', False):
            query_encoder_config = config['query_encoder']
            self.query_encoder = QueryEncoder(**query_encoder_config)
        else:
            self.query_encoder = None
            
        # 3. Two-Way Transformer
        transformer_config = config['transformer']
        dec_transformer = TwoWayTransformer(**transformer_config)

        # 4. Mask Decoder
        mask_decoder_config = config['mask_decoder']
        self.mask_decoder = MaskDecoder(
            transformer_dim=embed_dim,
            transformer=dec_transformer,
            **mask_decoder_config
        )

        # Store key information
        self.embed_dim = embed_dim
        self.multimask_output = config['multimask_output']


    def forward(self,
                point_cloud_data: Dict[str, torch.Tensor],
                query_point_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            point_cloud_data: 点云数据字典
                - 'feat': 点云特征 (N, C)
                - 'coord': 坐标 (N, 3)
                - 'batch': batch信息
            query_point_data: 查询点数据字典
                - 'position': 查询点位置 (3,) 或 (B, 3)

        Returns:
            masks: 分割mask logits (B, num_masks, N) or (num_masks, N_total)
            iou_pred: IoU predictions (B, num_masks)
        """
        # 1. Extract point cloud features from backbone
        backbone_output = self.backbone(point_cloud_data)
        point_features = backbone_output['feat']  # (N_total, embed_dim)

        # Get coordinates for positional encoding
        point_coords = point_cloud_data['coord']  # (N_total, 3)

        # 2. Generate positional embeddings for point cloud
        point_pe = self.pos_embd_layer(point_coords)  # (N_total, embed_dim)

        # 4. Compute offset from batch information
        if 'batch' in point_cloud_data:
            batch_indices = point_cloud_data['batch']
            batch_size = batch_indices.max().item() + 1
            offset = []
            cumsum = 0
            for i in range(batch_size):
                count = (batch_indices == i).sum().item()
                cumsum += count
                offset.append(cumsum)
            offset = torch.tensor(offset, device=point_features.device)
        elif 'offset' in point_cloud_data:
            offset = point_cloud_data['offset']
        else:
            offset = None

        # 3. Generate query point embeddings
      
        if self.query_encoder:
            query_positions = query_point_data['position']  # (B, 3) or (3,)
            query_features = self.query_encoder(query_point_data)  # (B, embed_dim)
        else:
            query_id = query_point_data['idx']  # (B,)
            start_idx = torch.zeros_like(query_id, device=query_id.device)
            start_idx[1:] = offset[:-1].clone()
            query_features = point_features[query_id+start_idx]  # (B, embed_dim)
            # query_features = self.pos_embd_layer(query_point_data['position'])  # (B, embed_dim)


        # 5. Run mask decoder
        masks, iou_pred = self.mask_decoder(
            point_features=point_features,  # (N_total, embed_dim)
            point_pe=point_pe,              # (N_total, embed_dim)
            query_features=query_features,  # (B, embed_dim)
            offset=offset,                  # (B,)
            multimask_output=self.multimask_output
        )

        return masks, iou_pred

    def predict(self,
                point_cloud_data: Dict[str, torch.Tensor],
                query_point_data: Dict[str, torch.Tensor],
                threshold: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测分割结果

        Args:
            point_cloud_data: 点云数据
            query_point_data: 查询点数据
            threshold: 二值化阈值

        Returns:
            masks: 二值化mask (B, num_masks, N) or (num_masks, N_total)
            iou_pred: IoU predictions (B, num_masks)
        """
        masks, iou_pred = self.forward(point_cloud_data, query_point_data)

        # Apply sigmoid and threshold
        masks = (torch.sigmoid(masks) > threshold).float()

        return masks, iou_pred

    def get_feature_maps(self,
                        point_cloud_data: Dict[str, torch.Tensor],
                        query_point_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        获取中间特征图 (用于分析或可视化)

        Returns:
            dict包含各种中间特征
        """
        # Backbone features
        backbone_output = self.backbone(point_cloud_data)
        point_features = backbone_output['feat']

        # Positional embeddings
        point_coords = point_cloud_data['coord']
        point_pe = self.pos_embd_layer(point_coords)

        # Query features
        query_positions = query_point_data['position']
        query_features = self.pos_embd_layer(query_positions)

        # Compute offset
        if 'batch' in point_cloud_data:
            batch_indices = point_cloud_data['batch']
            batch_size = batch_indices.max().item() + 1
            offset = []
            cumsum = 0
            for i in range(batch_size):
                count = (batch_indices == i).sum().item()
                cumsum += count
                offset.append(cumsum)
            offset = torch.tensor(offset, device=point_features.device)
        else:
            offset = None

        # Mask predictions
        masks, iou_pred = self.mask_decoder(
            point_features=point_features,
            point_pe=point_pe,
            query_features=query_features,
            offset=offset,
            multimask_output=self.multimask_output
        )

        return {
            'backbone_features': point_features,
            'point_pe': point_pe,
            'query_features': query_features,
            'masks': masks,
            'iou_pred': iou_pred,
            'offset': offset
        }

    def freeze_backbone(self, freeze: bool = True):
        """冻结/解冻 backbone"""
        self.backbone.freeze_backbone(freeze)

    def load_backbone_weights(self, checkpoint_path: str):
        """加载预训练 backbone 权重"""
        self.backbone.load_pretrained_weights(checkpoint_path)

    def get_num_parameters(self) -> str:
        """获取模型参数数量"""
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
        """获取模型信息"""
        return {
            'embed_dim': self.embed_dim,
            'multimask_output': self.multimask_output,
            'backbone_config': self.backbone.get_config(),
            'total_params': self.get_num_parameters(),
            'device': next(self.parameters()).device
        }


class SimplifiedPTv3PointCloudSegmentation(nn.Module):
    """
    Simplified Point Transformer V3 Point Cloud Segmentation Model with SimplifiedMaskDecoder

    使用简化的MaskDecoder的查询驱动点云分割模型，生成单个mask输出，不包含IoU预测。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化简化分割模型

        Args:
            config: 配置字典，包含所有组件的参数
        """
        super().__init__()

        self.config = config

        # 1. PTv3 Backbone
        backbone_config = config['backbone']
        self.backbone = PTv3Backbone(**backbone_config)

        # 2. Positional Embedding Layer
        embed_dim = config['embed_dim']
        self.pos_embd_layer = PositionEmbeddingRandom(
            num_pos_feats=embed_dim // 2,
            scale=config['pos_embd_scale']
        )

        # 3. Query Encoder
        query_encoder_config = config['query_encoder']
        self.query_encoder = QueryEncoder(**query_encoder_config)

        # 4. Two-Way Transformer
        transformer_config = config['transformer']
        dec_transformer = TwoWayTransformer(**transformer_config)

        # 5. Simplified Mask Decoder
        self.mask_decoder = SimplifiedMaskDecoder(
            transformer_dim=embed_dim,
            transformer=dec_transformer
        )

        # Store key information
        self.embed_dim = embed_dim

    def forward(self,
                point_cloud_data: Dict[str, torch.Tensor],
                query_point_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播

        Args:
            point_cloud_data: 点云数据字典
                - 'feat': 点云特征 (N, C)
                - 'coord': 坐标 (N, 3)
                - 'batch': batch信息
            query_point_data: 查询点数据字典
                - 'position': 查询点位置 (3,) 或 (B, 3)

        Returns:
            masks: 分割mask logits (B, N) or (N_total,)
        """
        # 1. Extract point cloud features from backbone
        backbone_output = self.backbone(point_cloud_data)
        point_features = backbone_output['feat']  # (N_total, embed_dim)

        # Get coordinates for positional encoding
        point_coords = point_cloud_data['coord']  # (N_total, 3)

        # 2. Generate positional embeddings for point cloud
        point_pe = self.pos_embd_layer(point_coords)  # (N_total, embed_dim)

        # 3. Generate query point embeddings
        query_features = self.query_encoder(query_point_data)  # (B, embed_dim)

        # 4. Compute offset from batch information
        if 'batch' in point_cloud_data:
            batch_indices = point_cloud_data['batch']
            batch_size = batch_indices.max().item() + 1
            offset = []
            cumsum = 0
            for i in range(batch_size):
                count = (batch_indices == i).sum().item()
                cumsum += count
                offset.append(cumsum)
            offset = torch.tensor(offset, device=point_features.device)
        else:
            offset = None

        # 5. Run simplified mask decoder
        masks = self.mask_decoder(
            point_features=point_features,  # (N_total, embed_dim)
            point_pe=point_pe,              # (N_total, embed_dim)
            query_features=query_features,  # (B, embed_dim)
            offset=offset                  # (B,)
        )

        return masks

    def predict(self,
                point_cloud_data: Dict[str, torch.Tensor],
                query_point_data: Dict[str, torch.Tensor],
                threshold: float = 0.0) -> torch.Tensor:
        """
        预测分割结果

        Args:
            point_cloud_data: 点云数据
            query_point_data: 查询点数据
            threshold: 二值化阈值

        Returns:
            masks: 二值化mask (B, N) or (N_total,)
        """
        masks = self.forward(point_cloud_data, query_point_data)

        # Apply sigmoid and threshold
        masks = (torch.sigmoid(masks) > threshold).float()

        return masks

    def get_feature_maps(self,
                        point_cloud_data: Dict[str, torch.Tensor],
                        query_point_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        获取中间特征图 (用于分析或可视化)

        Returns:
            dict包含各种中间特征
        """
        # Backbone features
        backbone_output = self.backbone(point_cloud_data)
        point_features = backbone_output['feat']

        # Positional embeddings
        point_coords = point_cloud_data['coord']
        point_pe = self.pos_embd_layer(point_coords)

        # Query features
        query_features = self.query_encoder(query_point_data)

        # Compute offset
        if 'batch' in point_cloud_data:
            batch_indices = point_cloud_data['batch']
            batch_size = batch_indices.max().item() + 1
            offset = []
            cumsum = 0
            for i in range(batch_size):
                count = (batch_indices == i).sum().item()
                cumsum += count
                offset.append(cumsum)
            offset = torch.tensor(offset, device=point_features.device)
        else:
            offset = None

        # Mask predictions
        masks = self.mask_decoder(
            point_features=point_features,
            point_pe=point_pe,
            query_features=query_features,
            offset=offset
        )

        return {
            'backbone_features': point_features,
            'point_pe': point_pe,
            'query_features': query_features,
            'masks': masks,
            'offset': offset
        }

    def freeze_backbone(self, freeze: bool = True):
        """冻结/解冻 backbone"""
        self.backbone.freeze_backbone(freeze)

    def load_backbone_weights(self, checkpoint_path: str):
        """加载预训练 backbone 权重"""
        self.backbone.load_pretrained_weights(checkpoint_path)

    def get_num_parameters(self) -> str:
        """获取模型参数数量"""
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
        """获取模型信息"""
        return {
            'embed_dim': self.embed_dim,
            'backbone_config': self.backbone.get_config(),
            'total_params': self.get_num_parameters(),
            'device': next(self.parameters()).device,
            'decoder_type': 'SimplifiedMaskDecoder'
        }

def create_model(config: Dict[str, Any], use_simplified_decoder: bool = False) -> Union[PTv3PointCloudSegmentation, 'SimplifiedPTv3PointCloudSegmentation']:
    """
    模型工厂函数

    Args:
        config: 模型配置字典
        use_simplified_decoder: 是否使用简化的MaskDecoder

    Returns:
        初始化的模型实例
    """
    if use_simplified_decoder:
        return SimplifiedPTv3PointCloudSegmentation(config)
    else:
        return PTv3PointCloudSegmentation(config)