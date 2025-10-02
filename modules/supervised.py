import os

import torch

from modules.loss_module import LossModule
from networks.heads import MLPHead
from utils.constants import LOSS, SIMILARITY_FUNCTIONS
from torch.nn import functional as F


class Supervised(LossModule):
    def __init__(self,args, fabric, net=None, net_target=None, n_classes=0, train_dataset=None, **kwargs):
        self.args = args
        self.parameters = []
        net.add_module("sup_projector",MLPHead(args,net.num_output, args.sup_hidden_dim, train_dataset.n_classes))
        net.register_buffer("sup_proj_output", torch.empty((2*args.batch_size, train_dataset.n_classes)), persistent=False)

    @classmethod
    def get_args(cls, parser):
        parser.add_argument('--sup_hidden_dim', default=256, type=int)
        return parser

    def apply(self, net, rep=None, net_target=None, data=None, **kwargs):
        net.sup_proj_output = net.sup_projector(rep)
        y1, y2 = net.proj_output.split(net.proj_output.shape[0]//2)
        loss = 0.5*F.cross_entropy(y1, data[1]).mean() + 0.5*F.cross_entropy(y2, data[1]).mean()
        return loss



