

import torch

from modules.loss_module import LossModule
from torch.nn import functional as F
from torch import nn

from utils.general import str2table


class OnlineLinearEval(LossModule):

    def __init__(self,args, fabric, net=None, net_target=None, train_dataset=None, **kwargs):
        self.args = args
        self.fabric=fabric
        self.parameters = []
        self.losses = [self.fabric.to_device(torch.zeros((1,))) for _ in range(len(args.eval_labels))]
        self.train_accs = [self.fabric.to_device(torch.zeros((1,))) for _ in range(len(args.eval_labels))]
        self.cpt = 0.00001
        if not hasattr(train_dataset, "all_classes"):
            self.classes = [train_dataset.n_classes]*len(args.eval_labels)
        else:
            self.classes = train_dataset.all_classes

        for k in range(len(args.eval_labels)):
            label_type = int(self.args.eval_labels[k])
            net.add_module("sup_lin_projector"+str(label_type), nn.Linear(net.num_output, self.classes[label_type]))


    @classmethod
    def get_args(cls, parser):
        parser.add_argument('--eval_labels', default="0,1", type=str2table)
        parser.add_argument('--linear_wait', default=0, type=int)
        return parser

    def apply(self, net, rep=None, net_target=None, data=None, epoch=None, **kwargs):
        if epoch < self.args.linear_wait:
            return 0
        loss = 0
        for k in range(len(self.args.eval_labels)):
            label_type = int(self.args.eval_labels[k])

            label = data[1+label_type]
            y1 = net.get_submodule("sup_lin_projector" + str(label_type))(rep.detach()[:rep.size(0)//2])
            l = F.cross_entropy(y1, label).mean()
            loss = loss + l
            self.losses[k] = self.losses[k] + l.detach()
            self.train_accs[k] = self.train_accs[k] + (y1.argmax(dim=1) == label).float().mean()
        self.cpt += 1
        return loss

    @torch.no_grad()
    def eval(self, network, f_l_test = None):
        dict = {}
        for k in range(len(self.args.eval_labels)):
            dict["lin_loss"+self.args.eval_labels[k]] = self.losses[k].item()/self.cpt
            dict["lin_acc"+self.args.eval_labels[k]] = self.train_accs[k].item()/self.cpt
            self.losses[k][:]=0
            self.train_accs[k][:]=0

        self.cpt = 0.00001

        if f_l_test is not None:
            for k, v in f_l_test.items():
                lin_net = network.get_submodule(f"sup_lin_projector{self.args.eval_labels[int(k)]}")
                output = lin_net(v["features"])
                corrects = output.argmax(dim=-1) == v["labels"]
                dict["test_acc"+k] = corrects.float().mean().item()
        return dict