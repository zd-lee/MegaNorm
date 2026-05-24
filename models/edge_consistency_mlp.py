import torch
import torch.nn as nn


class EdgeConsistencyMLP(nn.Module):
    def __init__(self, input_dim=512, hidden_dims=[512, 256, 128], dropout=0.1, use_bn=True, use_pos_encoding=True):
        super().__init__()

        self.input_dim = input_dim
        concat_dim = input_dim * 2
        self.use_bn = use_bn
        self.use_pos_encoding = use_pos_encoding

        if use_pos_encoding:
            self.pos_encoder = nn.Sequential(
                nn.Linear(3, concat_dim),
                nn.ReLU(),
                nn.Linear(concat_dim, concat_dim)
            )

        if use_bn:
            self.input_bn = nn.BatchNorm1d(concat_dim)

        layers = []
        in_dim = concat_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features_A, features_B, patch_centers_A=None, patch_centers_B=None, batch_offsets=None):
        """
        Args:
            features_A: (N_edges_total, 512)
            features_B: (N_edges_total, 512)
            patch_centers_A: (N_edges_total, 3)
            patch_centers_B: (N_edges_total, 3)
            batch_offsets: (B,)

        Returns:
            logits: (N_edges_total, 1)
        """
        x = torch.cat([features_A, features_B], dim=-1)

        if self.use_pos_encoding and patch_centers_A is not None and patch_centers_B is not None:
            relative_pos = patch_centers_A - patch_centers_B
            pos_embed = self.pos_encoder(relative_pos)
            x = x + pos_embed

        if self.use_bn:
            x = self.input_bn(x)
        return self.mlp(x)


def create_edge_consistency_mlp(config):
    model_config = config['model']
    return EdgeConsistencyMLP(
        input_dim=model_config.get('input_dim', 512),
        hidden_dims=model_config.get('hidden_dims', [512, 256, 128]),
        dropout=model_config.get('dropout', 0.1),
        use_bn=model_config.get('use_bn', True),
        use_pos_encoding=model_config.get('use_pos_encoding', True)
    )
