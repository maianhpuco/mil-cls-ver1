import torch.nn as nn
import torch
import copy
from typing import Tuple, Optional
import snuffy_module.fc_layer as fc_layer
import snuffy_module.multi_head_attention as multi_head_attention
import snuffy_module.pff as pff
import snuffy_module.b_classifier as bclassifier
import snuffy_module.encoder_layer as encoder_layer
import snuffy_module.encoder as encoder
import snuffy_module.mil_net as mil_net


device = "cuda" if torch.cuda.is_available() else "cpu"


class Snuffy(nn.Module):
    def __init__(self, 
                 feats_size,
                 num_classes,
                 depth,
                 big_lambda,
                 soft_average,
                 num_heads,
                 mlp_multiplier,
                 encoder_dropout,
                 activation,
                 random_patch_share,

                 weight_init__weight_init_i__weight_init_b,
                 ):
        self.feats_size = feats_size
        self.num_classes = num_classes
        self.depth = depth,
        self.big_lambda = big_lambda,
        self.soft_average = soft_average,
        self.num_heads = num_heads,
        self.mlp_multiplier = mlp_multiplier,
        self.encoder_dropout = encoder_dropout,
        self.activation = activation,
        self.random_patch_share = random_patch_share,
        self.weight_init__weight_init_i__weight_init_b = weight_init__weight_init_i__weight_init_b,
    def _get_milnet(self):
        i_classifier = fc_layer.FCLayer(in_size=self.feats_size,
                                        out_size=self.num_classes).to(device)
        c = copy.deepcopy
        attn = multi_head_attention.MultiHeadedAttention(
            self.num_heads,
            self.feats_size,
        ).to(device)
        ff = pff.PositionwiseFeedForward(
            self.feats_size,
            self.feats_size * self.mlp_multiplier,
            self.activation,
            self.encoder_dropout
        ).to(device)
        b_classifier = bclassifier.BClassifier(
            encoder.Encoder(
                encoder_layer.EncoderLayer(
                    self.feats_size,
                    c(attn),
                    c(ff),
                    self.encoder_dropout,
                    self.big_lambda,
                    self.random_patch_share
                ), self.depth
            ),
            self.num_classes,
            self.feats_size
        ).to(device)
        milnet = mil_net.MILNet(i_classifier, b_classifier).to(device)

        init_funcs_registry = {
            'trunc_normal': nn.init.trunc_normal_,
            'kaiming_uniform': nn.init.kaiming_uniform_,
            'kaiming_normal': nn.init.kaiming_normal_,
            'xavier_uniform': nn.init.xavier_uniform_,
            'xavier_normal': nn.init.xavier_normal_,
            'orthogonal': nn.init.orthogonal_
        }

        modules = [(self.weight_init__weight_init_i__weight_init_b[1], 'i_classifier'),
                   (self.weight_init__weight_init_i__weight_init_b[2], 'b_classifier')]
        print('modules:', modules)
        for init_func_name, module_name in modules:
            init_func = init_funcs_registry.get(init_func_name)
            print('init_func:', init_func)
            for name, p in milnet.named_parameters():
                if p.dim() > 1 and name.split(".")[0] == module_name:
                    init_func(p)

        return milnet


    def _run_model_previous(self, bag_feats, bag_label) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        ins_prediction, bag_prediction, attentions = self.milnet(bag_feats)
        if len(ins_prediction.shape) == 2:
            max_prediction, _ = torch.max(ins_prediction, 0)
        else:
            max_prediction, _ = torch.max(ins_prediction, 1)

        bag_loss = self.criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        max_loss = self.criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
        loss = self.single_weight_parameter * bag_loss + (1 - self.single_weight_parameter) * max_loss

        with torch.no_grad():
            bag_prediction = (
                    (1 - self.single_weight_parameter) * torch.sigmoid(max_prediction) +
                    self.single_weight_parameter * torch.sigmoid(bag_prediction)
            ).squeeze().cpu().numpy()

        return bag_prediction, loss, ins_prediction

    def _run_model(self, bag_feats, bag_label) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        bag_prediction, loss, ins_prediction = self._run_model_previous(bag_feats, bag_label)
        ins_prediction = ins_prediction.view(-1, 1)
        return bag_prediction, loss, torch.sigmoid(ins_prediction)

    def __str__(self) -> str:
        return f'Snuffy_k{self.big_lambda}_sa{self.soft_average}_depth{self.depth}'