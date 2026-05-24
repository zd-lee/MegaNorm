"""
PointTransformer V3 Backbone Adapter
适配 PTv3 用作点云分割任务的骨干网络
"""

import sys
import os
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from utils.model_logger import get_model_logger

# 添加 PTv3 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
ptv3_dir = os.path.join(os.path.dirname(current_dir))
sys.path.insert(0, ptv3_dir)

from .ptv3_origin import PointTransformerV3


class PTv3Backbone(nn.Module):
    """
    Point Transformer V3 Backbone for Point Cloud Segmentation

    这是一个适配器，将原始 PTv3 包装为适合点云分割任务的骨干网络。
    支持输出点级特征用于下游分割任务。
    """

    def __init__(self,
                 # Input parameters
                 in_channels: int = 6,  # 输入特征维度
                 
                 # Encoder参数
                 enc_depths: list = [2, 2, 2, 6, 2],
                 enc_channels: list = [32, 64, 128, 256, 512],
                 enc_num_head: list = [2, 4, 8, 16, 32],
                 enc_patch_size: list = [1024, 1024, 1024, 1024, 1024],

                 # Decoder参数
                 dec_depths: list = [2, 2, 2, 2],
                 dec_channels: list = [64, 64, 128, 256],
                 dec_num_head: list = [4, 4, 8, 16],
                 dec_patch_size: list = [1024, 1024, 1024, 1024],

                 # 其他参数
                 stride: list = [2, 2, 2, 2],
                 order: str = 'z',
                 enable_flash: bool = True,
                 enable_rpe: bool = True,
                 upcast_attention: bool = False,
                 upcast_softmax: bool = False,
                 cls_mode: bool = False,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0,
                 drop_path: float = 0.3,
                 pre_norm: bool = True,
                 shuffle_orders: bool = True,

                 # 为了向后兼容保留的旧参数
                 depths: list = None,
                 channels: list = None,
                 num_heads: int = None,
                 patch_size: int = None,
                 num_classes: int = 20):
        """
        初始化 PTv3 Backbone

        Args:
            in_channels: 输入特征的通道数 (默认6: xyz + rgb 或 xyz + normals)
            enc_depths: encoder各层深度
            enc_channels: encoder各层通道数
            dec_depths: decoder各层深度
            dec_channels: decoder各层通道数
            order: 空间序列化顺序 ('z', 'hilbert')
            enable_flash: 是否启用 Flash Attention
            enable_rpe: 是否启用相对位置编码
        """
        super().__init__()

        # 向后兼容：如果使用了旧参数，转换为新参数
        if depths is not None:
            enc_depths = depths if len(depths) == 5 else [2, 2, 2, 6, 2]
        if channels is not None:
            enc_channels = channels if len(channels) == 5 else [32, 64, 128, 256, 512]
        if num_heads is not None:
            enc_num_head = [num_heads] * 5 if isinstance(num_heads, int) else enc_num_head
        if patch_size is not None:
            enc_patch_size = [patch_size] * 5 if isinstance(patch_size, int) else enc_patch_size

        # 创建 PTv3 模型实例（现在包含encoder和decoder）
        self.ptv3 = PointTransformerV3(
            in_channels=in_channels,  # 传入输入通道数
            enc_depths=tuple(enc_depths),
            enc_channels=tuple(enc_channels),
            enc_num_head=tuple(enc_num_head),
            enc_patch_size=tuple(enc_patch_size),
            dec_depths=tuple(dec_depths),
            dec_channels=tuple(dec_channels),
            dec_num_head=tuple(dec_num_head),
            dec_patch_size=tuple(dec_patch_size),
            stride=tuple(stride),
            order=order,
            enable_flash=enable_flash,
            enable_rpe=enable_rpe,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            cls_mode=cls_mode,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path,
            pre_norm=pre_norm,
            shuffle_orders=shuffle_orders
        )

        # 存储配置
        self.config = {
            'in_channels': in_channels,
            'enc_depths': enc_depths,
            'enc_channels': enc_channels,
            'enc_num_head': enc_num_head,
            'enc_patch_size': enc_patch_size,
            'dec_depths': dec_depths,
            'dec_channels': dec_channels,
            'dec_num_head': dec_num_head,
            'dec_patch_size': dec_patch_size,
            'stride': stride,
            'order': order,
            'enable_flash': enable_flash,
            'enable_rpe': enable_rpe,
            'cls_mode': cls_mode
        }

    def forward(self, point_dict: Dict[str, torch.Tensor], use_decoder: bool = None) -> Dict[str, Any]:
        """
        前向传播
        返回点级特征用于下游分割任务

        Args:
            point_dict: 点云数据字典，必须包含：
                - 'feat': 点云特征 (N, C)
                - 'grid_coord' 或 'coord'+'grid_size': 坐标信息
                - 'offset' 或 'batch': batch信息
            use_decoder: Optional[bool] - 控制是否使用decoder
                - None: 使用模型默认的cls_mode设置
                - True: 强制使用decoder（encoder+decoder）
                - False: 强制不使用decoder（仅encoder）

        Returns:
            dict with keys:
                - 'feat': 提取的点云特征 (N, embed_dim)
                - 'offset': batch offset信息
                - 'batch_size': batch大小
        """

        # [NaN Check 0] Input to PTv3
        logger = get_model_logger()
        if torch.isnan(point_dict['feat']).any():
            logger.warning(f"[NaN DETECTED] in PTv3 input features (point_dict['feat'])")
            logger.warning(f"  Shape: {point_dict['feat'].shape}")
            logger.warning(f"  NaN count: {torch.isnan(point_dict['feat']).sum().item()}")
            logger.warning(f"  Min/Max: {point_dict['feat'][~torch.isnan(point_dict['feat'])].min().item() if (~torch.isnan(point_dict['feat'])).any() else 'all NaN'}")

        if 'coord' in point_dict and torch.isnan(point_dict['coord']).any():
            logger.warning(f"[NaN DETECTED] in PTv3 input coords (point_dict['coord'])")
            logger.warning(f"  Shape: {point_dict['coord'].shape}")
            logger.warning(f"  NaN count: {torch.isnan(point_dict['coord']).sum().item()}")

        # 将数据字典传递给 PTv3，并传递 use_decoder 参数
        with torch.no_grad() if not self.training else torch.enable_grad():
            point = self.ptv3(point_dict, use_decoder=use_decoder)

        # [NaN Check 0.1] PTv3 output
        if torch.isnan(point.feat).any():
            logger.warning(f"[NaN DETECTED] in PTv3 output features (point.feat)")
            logger.warning(f"  Shape: {point.feat.shape}")
            logger.warning(f"  NaN count: {torch.isnan(point.feat).sum().item()}")
            logger.warning(f"  Min/Max (non-NaN): {point.feat[~torch.isnan(point.feat)].min().item() if (~torch.isnan(point.feat)).any() else 'all NaN'}/{point.feat[~torch.isnan(point.feat)].max().item() if (~torch.isnan(point.feat)).any() else 'all NaN'}")

        # 返回特征给下游任务
        return {
            'feat': point.feat,            # (N, embed_dim)
            'offset': getattr(point, 'offset', None),  # batch offset
            'batch_size': getattr(point, 'batch', None) if hasattr(point, 'batch') else None,
            'spatial_shape': getattr(point, 'spatial_shape', None), # 如果有的话
            'coord': getattr(point, 'coord', None)  # 如果有的话
        }

    def get_feature_dim(self) -> int:
        """
        获取输出特征维度
        
        返回backbone输出的特征向量维度，用于后续网络层的输入维度配置。
        该维度等于decoder第一层的输出通道数。
        
        Returns:
            int: 输出特征的维度大小
            
        Example:
            >>> backbone = PTv3Backbone(dec_channels=[64, 64, 128, 256])
            >>> feat_dim = backbone.get_feature_dim()
            >>> print(feat_dim)  # 输出: 64
        """
        return self.config['dec_channels'][0]  # decoder第一层的输出通道数

    def get_config(self) -> Dict[str, Any]:
        """
        获取当前模型配置
        
        返回包含所有模型超参数的配置字典副本。这对于模型保存、
        日志记录和配置比较非常有用。
        
        Returns:
            Dict[str, Any]: 包含以下键值的配置字典：
                - in_channels: 输入特征通道数
                - enc_depths: encoder各层深度列表
                - enc_channels: encoder各层通道数列表
                - enc_num_head: encoder各层注意力头数列表
                - enc_patch_size: encoder各层patch大小列表
                - dec_depths: decoder各层深度列表
                - dec_channels: decoder各层通道数列表
                - dec_num_head: decoder各层注意力头数列表
                - dec_patch_size: decoder各层patch大小列表
                - stride: 下采样步长列表
                - order: 空间序列化顺序
                - enable_flash: 是否启用Flash Attention
                - enable_rpe: 是否启用相对位置编码
                - cls_mode: 是否为分类模式
                
        Example:
            >>> backbone = PTv3Backbone()
            >>> config = backbone.get_config()
            >>> print(config['enc_channels'])  # [32, 64, 128, 256, 512]
        """
        return self.config.copy()

    def freeze_backbone(self, freeze: bool = True):
        """
        冻结或解冻backbone参数
        
        在迁移学习或微调场景中，可能需要冻结预训练的backbone参数，
        只训练新添加的分割头。此方法允许灵活控制backbone的可训练状态。
        
        Args:
            freeze (bool): 如果为True，冻结所有backbone参数（不更新梯度）；
                          如果为False，解冻参数使其可训练。默认为True。
                          
        Example:
            >>> backbone = PTv3Backbone()
            >>> # 冻结backbone用于迁移学习
            >>> backbone.freeze_backbone(freeze=True)
            >>> # 稍后解冻进行端到端微调
            >>> backbone.freeze_backbone(freeze=False)
            
        Note:
            冻结参数后，这些参数在反向传播时不会更新，可以节省显存和计算资源。
        """
        for param in self.ptv3.parameters():
            param.requires_grad = not freeze

    def load_pretrained_weights(self, checkpoint_path: str):
        """
        加载预训练权重
        
        从checkpoint文件加载预训练的模型权重。支持加载完整checkpoint或
        仅包含state_dict的文件。会自动过滤掉不兼容的decoder层权重。
        
        Args:
            checkpoint_path (str): 预训练权重文件的路径。支持.pth或.pt格式。
            
        Raises:
            FileNotFoundError: 如果checkpoint文件不存在（隐式处理）
            
        Note:
            - 使用strict=False允许部分加载权重，跳过不匹配的层
            - 自动过滤以'dec.'开头的decoder层权重，因为下游任务可能有不同的decoder
            - checkpoint会被加载到CPU上，避免GPU内存问题
            
        Example:
            >>> backbone = PTv3Backbone()
            >>> backbone.load_pretrained_weights('checkpoints/ptv3_pretrained.pth')
            >>> print("预训练权重加载完成")
            
        Warning:
            如果checkpoint文件不存在，此方法会静默失败。
            建议在调用前检查文件是否存在。
        """
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # 过滤掉分类层权重（如果有的话）
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if not k.startswith('dec.'):  # 移除解码器权重
                    filtered_state_dict[k] = v

            self.ptv3.load_state_dict(filtered_state_dict, strict=False)
        else:
            pass