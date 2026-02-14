import torch
import torch.nn as nn


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
    def __init__(self, in_features, out_features, d_model=64, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()

        # Feature Tokenizer: Project each scalar feature to a vector of size d_model
        # We use a ModuleList of Linear layers (1 -> d_model) to learn unique embeddings per feature
        self.feature_projectors = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(in_features)
        ])

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
            norm_first=True  # Pre-LN helps significantly with convergence
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output Head
        self.head = nn.Linear(d_model, out_features)

    def forward(self, x):
        # x shape: (Batch, Features)
        batch_size = x.shape[0]
        
        # 1. Feature Tokenization: (Batch, Features) -> (Batch, Features, d_model)
        x_expanded = x.unsqueeze(-1)
        embeddings = [proj(x_expanded[:, i, :]) for i, proj in enumerate(self.feature_projectors)]
        x_emb = torch.stack(embeddings, dim=1)

        # 2. Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_seq = torch.cat((cls_tokens, x_emb), dim=1)

        # 3. Transformer Processing
        x_tf = self.transformer(x_seq)

        # 4. Prediction using CLS token (index 0)
        out = self.head(x_tf[:, 0, :])

        return out
