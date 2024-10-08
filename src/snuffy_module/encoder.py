import torch.nn as nn
from src.snuffy_module.utils import clones


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, c):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x, attntions = layer(x, c)
        return self.norm(x), attntions
