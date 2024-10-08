import torch
import torch.nn as nn
from src.snuffy_module.sub_layer_connection import SublayerConnection
from src.snuffy_module.utils import clones
import math


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        dropout,
        big_lambda,
        random_patch_share,
    ):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        self.big_lambda = big_lambda
        self.random_patch_share = random_patch_share
        self.top_big_lambda_share = 1.0 - random_patch_share

    def forward(self, x, c):
        "Follow Figure 1 (left) for connections."
        _, m_indices = torch.sort(c, 1, descending=True)
        top_big_lambda_share_indices = m_indices[
            :, 0 : math.ceil(self.big_lambda * self.top_big_lambda_share), :
        ].squeeze()
        top_big_lambdas = torch.index_select(
            x, dim=1, index=top_big_lambda_share_indices
        )

        if (
            top_big_lambda_share_indices.dim() == 0
        ):  # If topk_share_indices is a Scalar tensor, convert it to 1-D tensor
            top_big_lambda_share_indices = (
                top_big_lambda_share_indices.unsqueeze(0)
            )

        remaining_indices = list(
            set(range(x.shape[1])) - set(top_big_lambda_share_indices.tolist())
        )
        randoms_share = min(
            int(self.big_lambda * self.random_patch_share),
            max(
                0,
                x.shape[1]
                - math.ceil(self.big_lambda * self.top_big_lambda_share),
            ),
        )
        random_indices = (
            torch.from_numpy(
                np.random.choice(
                    remaining_indices, randoms_share, replace=False
                )
            ).to(device)
            if randoms_share != 0
            else None
        )

        random_big_lambda = (
            torch.index_select(x, dim=1, index=random_indices)
            if randoms_share != 0
            else None
        )
        top_big_lambda = (
            torch.cat((top_big_lambdas, random_big_lambda), dim=1)
            if randoms_share != 0
            else top_big_lambdas
        )
        x_big_lambda, attentions = self.sublayer[0](
            x,
            lambda x: self.self_attn(x, top_big_lambda, x),
            c,
            top_big_lambda_share_indices,
            random_indices,
            "attn",
        )

        selected_indices = (
            torch.hstack((top_big_lambda_share_indices, random_indices))
            if randoms_share != 0
            else top_big_lambda_share_indices
        )
        y = x.clone()
        y[:, selected_indices, :] = x_big_lambda

        return (
            self.sublayer[1](y, self.feed_forward, c, None, None, "ff"),
            attentions,
        )
