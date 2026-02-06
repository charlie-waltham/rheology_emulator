import torch.nn as nn


class SIVTransformer(nn.Transformer):
    def __init__(self, num_features, num_labels, num_heads=8, activation="relu"):
        super(SIVTransformer, self).__init__(
            d_model=num_features,
            nhead=num_heads,
            batch_first=True,
            activation=activation,
        )

        self.decoder = nn.Linear(num_features, num_labels)

    def forward(self, src):
        x = self.encoder(src)
        x = self.decoder(x)

        return x


class ResBlockMLP(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(ResBlockMLP, self).__init__()

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

        out += residual
        return out


class MultiResBlockMLP(nn.Module):
    def __init__(self, in_features, out_features, dropout, num_blocks):
        super(MultiResBlockMLP, self).__init__()

        layers = []
        layers.append(ResBlockMLP(in_features, out_features, dropout))

        for _ in range(num_blocks - 1):
            layers.append(ResBlockMLP(out_features, out_features, dropout))

        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)
