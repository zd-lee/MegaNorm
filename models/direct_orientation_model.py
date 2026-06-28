import torch
import torch.nn as nn

from typing import Dict, Any, Tuple

from .ptv3_backbone import PTv3Backbone


class DirectOrientationModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        backbone_config = config["backbone"]
        self.backbone = PTv3Backbone(**backbone_config)
        self.backbone_dim = self.backbone.get_feature_dim()
        mlp_config = config.get("mlp_head", {})
        hidden_dim = mlp_config.get("hidden_dim", 256)
        num_layers = mlp_config.get("num_layers", 3)
        dropout = mlp_config.get("dropout", 0.1)
        num_classes = 1
        layers = []
        in_dim = self.backbone_dim
        for i in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.mlp_head = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, point_cloud_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        backbone_output = self.backbone(point_cloud_data)
        point_features = backbone_output["feat"]
        logits = self.mlp_head(point_features)
        return logits

    def predict(self, point_cloud_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self.forward(point_cloud_data)
        predictions = torch.argmax(logits, dim=1)
        return predictions

    def predict_proba(self, point_cloud_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self.forward(point_cloud_data)
        probs = torch.softmax(logits, dim=1)
        return probs

    def freeze_backbone(self, freeze: bool = True):
        self.backbone.freeze_backbone(freeze)

    def load_backbone_weights(self, checkpoint_path: str):
        self.backbone.load_pretrained_weights(checkpoint_path)

    def get_num_parameters(self) -> str:
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
        return {
            "backbone_dim": self.backbone_dim,
            "mlp_config": self.config.get("mlp_head", {}),
            "backbone_config": self.backbone.get_config(),
            "total_params": self.get_num_parameters(),
            "device": next(self.parameters()).device,
        }


def create_direct_orientation_model(config: Dict[str, Any]) -> DirectOrientationModel:
    return DirectOrientationModel(config)
