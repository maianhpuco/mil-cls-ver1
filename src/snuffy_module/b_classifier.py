import torch.nn as nn


class BClassifier(nn.Module):
    def __init__(self, encoder, num_classes, input_size: int):
        super(BClassifier, self).__init__()
        self.encoder = encoder
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x, c):
        "Pass the input (and mask) through each layer in turn."
        x, attentions = self.encoder(x, c)
        return self.linear(x.mean(dim=1)), attentions
