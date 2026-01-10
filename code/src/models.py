import torch
import torch.nn as nn

class SIVTransformer(nn.Transformer):
    def __init__(self, num_features, num_labels, num_heads=8, activation="relu"):
        super(SIVTransformer, self).__init__(d_model=num_features, nhead=num_heads, batch_first=True, activation=activation)

        self.decoder = nn.Linear(num_features, num_labels)
    
    def forward(self, src):
        x = self.encoder(src)
        x = self.decoder(x)

        return x