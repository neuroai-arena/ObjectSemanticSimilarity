import torch

from modules.loss_module import LossModule
from networks.heads import MLPHead
from utils.constants import LOSS, SIMILARITY_FUNCTIONS


class LabelSSL(LossModule):
    def __init__(self, args, fabric, net=None, net_target=None, train_dataset=None, **kwargs):
        self.args = args
        self.fabric = fabric
        self.n_classes = train_dataset.n_classes
        net.add_module("label_encoder", MLPHead(args, self.n_classes, args.hidden_dim, args.feature_dim, args.hidden_layers))
        net.add_module("label_head", MLPHead(args, net.num_output, args.hidden_dim, args.feature_dim, args.hidden_layers))
        self.loss = LOSS[args.main_loss](args, SIMILARITY_FUNCTIONS[args.similarity], fabric, temperature = self.args.label_temperature)


    @classmethod
    def get_args(cls, parser):
        parser.add_argument('--sparse_weight', default=1, type=float)
        parser.add_argument('--label_temperature', default=0.1, type=float)

        return parser

    def apply(self, net, rep=None, logger=None, step=None, data=None,**kwargs):
        y1 = net.label_head(rep[:rep.shape[0]//2])
        y2 = net.label_encoder(torch.nn.functional.one_hot(data[1], self.n_classes).to(torch.float))
        sparse_loss = self.args.sparse_weight * self.loss(y1, y2).mean()
        if logger is not None:
            logger.log("Loss/labels", sparse_loss, step)

        return sparse_loss
