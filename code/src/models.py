import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlockMLP(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super().__init__()

        self.lin1 = nn.Linear(in_features, out_features, bias=False)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.act1 = nn.GELU()

        self.lin2 = nn.Linear(out_features, out_features, bias=False)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.act2 = nn.GELU()

        self.dropout = nn.Dropout(dropout)

        self.shortcut = nn.Identity()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features, bias=False),
                nn.BatchNorm1d(out_features),
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.lin1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.lin2(out)
        out = self.bn2(out)
        out = self.act2(out)

        out = self.dropout(out)

        # https://github.com/shap/shap/issues/3466
        out = out + residual
        return out


class MultiResBlockMLP(nn.Module):
    def __init__(self, in_features, out_features, dropout, num_blocks):
        super().__init__()

        layers = []
        layers.append(ResBlockMLP(in_features, out_features, dropout))

        for _ in range(num_blocks - 1):
            layers.append(ResBlockMLP(out_features, out_features, dropout))

        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)

class TabularTransformer(nn.Module):
    """
    A Transformer architecture adapted for tabular/point-wise data (FT-Transformer style).
    Projects each feature into an embedding space and applies self-attention.
    """

    def __init__(
        self, in_features, out_features, d_model=64, nhead=4, num_layers=3, dropout=0.1
    ):
        super().__init__()

        # Feature Tokenizer: Project each scalar feature to a vector of size d_model
        # We use a ModuleList of Linear layers (1 -> d_model) to learn unique embeddings per feature
        self.feature_projectors = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(in_features)]
        )

        # CLS Token: Learnable embedding to aggregate information
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,  # Pre-LN helps significantly with convergence
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output Head
        self.head = nn.Linear(d_model, out_features)

    def forward(self, x):
        # x shape: (Batch, Features)
        batch_size = x.shape[0]

        # 1. Feature Tokenization: (Batch, Features) -> (Batch, Features, d_model)
        x_expanded = x.unsqueeze(-1)
        embeddings = [
            proj(x_expanded[:, i, :]) for i, proj in enumerate(self.feature_projectors)
        ]
        x_emb = torch.stack(embeddings, dim=1)

        # 2. Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_seq = torch.cat((cls_tokens, x_emb), dim=1)

        # 3. Transformer Processing
        x_tf = self.transformer(x_seq)

        # 4. Prediction using CLS token (index 0)
        out = self.head(x_tf[:, 0, :])

        return out

class ResidualPointBlock(nn.Module):
    def __init__(self, dim, dropout=0.5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return x + self.block(x)

class RheologyFCN(nn.Module):
    def __init__(self, in_channels=8, out_channels=2, hidden_dim=512, mlp_depth=6, dropout=0.5):
        super().__init__()

        self.spatial_context = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.GELU(),

            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.GELU(),

            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )

        mlp_layers = []
        for _ in range(mlp_depth):
            mlp_layers.append(ResidualPointBlock(hidden_dim, dropout))

        self.point_mlp = nn.Sequential(*mlp_layers)

        self.final_layer = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.spatial_context(x)
        x = self.point_mlp(x)
        return self.final_layer(x)

class StencilResMLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim, dropout, num_blocks, neighbourhood=3):
        super().__init__()

        # Enforce odd neighbourhood for clean padding
        assert neighbourhood % 2 != 0, "Neighbourhood size must be an odd number."

        self.neighbourhood = neighbourhood
        self.padding = neighbourhood // 2

        self.num_features = neighbourhood * neighbourhood * in_features

        self.mlp = nn.Sequential(
            MultiResBlockMLP(self.num_features, hidden_dim, dropout, num_blocks - 1),
            ResBlockMLP(hidden_dim, out_features, dropout)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. Extract local neighbourhoods
        # Shape: (Batch, Channels, H, W) -> (Batch, Channels * N * N, H * W)
        x = F.unfold(x, kernel_size=self.neighbourhood, padding=self.padding)
        spatial_size = x.size(2)

        # 2. Transpose AND Flatten
        # Shape: (B, Features, L) -> (B, L, Features) -> (B * L, Features)
        x = x.transpose(1, 2).reshape(-1, self.num_features)

        # 3. Apply the deep physics simulation independently to every point
        # Shape: (Batch, L, Features) -> (Batch, L, Out_Features)
        x = self.mlp(x)
        out_features = x.size(1)

        # 4. Transpose back
        # Shape: (Batch, L, Out_Features) -> (Batch, Out_Features, L)
        x = x.view(B, spatial_size, out_features).transpose(1, 2)

        # 5. Reshape back into a 2D image map
        # Shape: (Batch, Out_Features, H*W) -> (Batch, Out_Features, H, W)
        x = x.view(B, -1, H, W)

        return x