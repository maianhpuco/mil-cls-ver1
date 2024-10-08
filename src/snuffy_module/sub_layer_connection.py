import torch.nn as nn
import torch


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x, sublayer, c, top_big_lambda_indices, random_indices, mode
    ):
        "Apply residual connection to any sublayer with the same size."
        if mode == "attn":
            top_big_lambdas = torch.index_select(
                x, dim=1, index=top_big_lambda_indices
            )
            random_big_lambda = (
                torch.index_select(x, dim=1, index=random_indices)
                if random_indices != None
                else None
            )
            top_big_lambda = (
                torch.cat((top_big_lambdas, random_big_lambda), dim=1)
                if random_indices != None
                else top_big_lambdas
            )
            multiheadedattn = sublayer(self.norm(x))
            return (
                top_big_lambda + self.dropout(multiheadedattn[0]),
                multiheadedattn[1],
            )
        elif mode == "ff":
            return x + self.dropout(sublayer(self.norm(x)))
