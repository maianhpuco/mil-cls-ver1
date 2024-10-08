import torch.nn as nn


class PositionwiseFeedForward(nn.Module):  # mikham
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, activation, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        activation_dictionary = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "leakyrelu": nn.LeakyReLU(),
            "selu": nn.SELU(),
        }
        self.activation = activation_dictionary[activation]

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
