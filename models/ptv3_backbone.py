import sys
import os
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from utils.model_logger import get_model_logger

current_dir = os.path.dirname(os.path.abspath(__file__))
ptv3_dir = os.path.join(os.path.dirname(current_dir))
sys.path.insert(0, ptv3_dir)

from .ptv3_origin import PointTransformerV3


class PTv3Backbone(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        enc_depths: list = [2, 2, 2, 6, 2],
        enc_channels: list = [32, 64, 128, 256, 512],
        enc_num_head: list = [2, 4, 8, 16, 32],
        enc_patch_size: list = [1024, 1024, 1024, 1024, 1024],
        dec_depths: list = [2, 2, 2, 2],
        dec_channels: list = [64, 64, 128, 256],
        dec_num_head: list = [4, 4, 8, 16],
        dec_patch_size: list = [1024, 1024, 1024, 1024],
        stride: list = [2, 2, 2, 2],
        order: str = "z",
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
        depths: list = None,
        channels: list = None,
        num_heads: int = None,
        patch_size: int = None,
        num_classes: int = 20,
    ):
        super().__init__()
        if depths is not None:
            enc_depths = depths if len(depths) == 5 else [2, 2, 2, 6, 2]
        if channels is not None:
            enc_channels = channels if len(channels) == 5 else [32, 64, 128, 256, 512]
        if num_heads is not None:
            enc_num_head = (
                [num_heads] * 5 if isinstance(num_heads, int) else enc_num_head
            )
        if patch_size is not None:
            enc_patch_size = (
                [patch_size] * 5 if isinstance(patch_size, int) else enc_patch_size
            )
        self.ptv3 = PointTransformerV3(
            in_channels=in_channels,
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
            shuffle_orders=shuffle_orders,
        )
        self.config = {
            "in_channels": in_channels,
            "enc_depths": enc_depths,
            "enc_channels": enc_channels,
            "enc_num_head": enc_num_head,
            "enc_patch_size": enc_patch_size,
            "dec_depths": dec_depths,
            "dec_channels": dec_channels,
            "dec_num_head": dec_num_head,
            "dec_patch_size": dec_patch_size,
            "stride": stride,
            "order": order,
            "enable_flash": enable_flash,
            "enable_rpe": enable_rpe,
            "cls_mode": cls_mode,
        }

    def forward(
        self, point_dict: Dict[str, torch.Tensor], use_decoder: bool = None
    ) -> Dict[str, Any]:
        logger = get_model_logger()
        if torch.isnan(point_dict["feat"]).any():
            logger.warning(
                f"[NaN DETECTED] in PTv3 input features (point_dict['feat'])"
            )
            logger.warning(f"  Shape: {point_dict['feat'].shape}")
            logger.warning(
                f"  NaN count: {torch.isnan(point_dict['feat']).sum().item()}"
            )
            logger.warning(
                f"  Min/Max: {point_dict['feat'][~torch.isnan(point_dict['feat'])].min().item() if (~torch.isnan(point_dict['feat'])).any() else 'all NaN'}"
            )
        if "coord" in point_dict and torch.isnan(point_dict["coord"]).any():
            logger.warning(f"[NaN DETECTED] in PTv3 input coords (point_dict['coord'])")
            logger.warning(f"  Shape: {point_dict['coord'].shape}")
            logger.warning(
                f"  NaN count: {torch.isnan(point_dict['coord']).sum().item()}"
            )
        with torch.no_grad() if not self.training else torch.enable_grad():
            point = self.ptv3(point_dict, use_decoder=use_decoder)
        if torch.isnan(point.feat).any():
            logger.warning(f"[NaN DETECTED] in PTv3 output features (point.feat)")
            logger.warning(f"  Shape: {point.feat.shape}")
            logger.warning(f"  NaN count: {torch.isnan(point.feat).sum().item()}")
            logger.warning(
                f"  Min/Max (non-NaN): {point.feat[~torch.isnan(point.feat)].min().item() if (~torch.isnan(point.feat)).any() else 'all NaN'}/{point.feat[~torch.isnan(point.feat)].max().item() if (~torch.isnan(point.feat)).any() else 'all NaN'}"
            )
        return {
            "feat": point.feat,
            "offset": getattr(point, "offset", None),
            "batch_size": (
                getattr(point, "batch", None) if hasattr(point, "batch") else None
            ),
            "spatial_shape": getattr(point, "spatial_shape", None),
            "coord": getattr(point, "coord", None),
        }

    def get_feature_dim(self) -> int:
        return self.config["dec_channels"][0]

    def get_config(self) -> Dict[str, Any]:
        return self.config.copy()

    def freeze_backbone(self, freeze: bool = True):
        for param in self.ptv3.parameters():
            param.requires_grad = not freeze

    def load_pretrained_weights(self, checkpoint_path: str):
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if not k.startswith("dec."):
                    filtered_state_dict[k] = v
            self.ptv3.load_state_dict(filtered_state_dict, strict=False)
        else:
            pass
