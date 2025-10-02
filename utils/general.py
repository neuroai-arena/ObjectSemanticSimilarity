#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import os
import random
from copy import deepcopy

import numpy as np
import torch

# from networks.heads import MLPHead
import lightning as L
from lightning.fabric.strategies import DDPStrategy
from lightning.pytorch.utilities.types import DistributedDataParallel


# configuration module
# -----


# custom functions
# -----

def prepare_device(args):
    # torch.set_float32_matmul_precision('high')
    # if args.dataset == "MVImgNet" and args.hdf5_mode == "partition":
    #     import scripts.idr_torch
    np.set_printoptions(linewidth=np.nan, precision=2)
    torch.set_printoptions(precision=3, linewidth=150)
    torch.set_float32_matmul_precision('medium')

    if args.precision == "mixed":
        precision = "16-mixed"
    else:
        precision = "32-true"

    if args.split_fwd:
        fabric = L.Fabric(accelerator=args.device, devices=args.num_devices, strategy=DDPStrategy(broadcast_buffers=False), num_nodes=args.num_nodes, precision=precision)
    else:
        fabric = L.Fabric(accelerator=args.device, devices=args.num_devices, strategy="ddp", num_nodes=args.num_nodes, precision=precision)

    if args.seed != -1:
        fabric.seed_everything(args.seed)

    # kwarg_launch={}
    # if args.num_devices > 1:
    #     import idr_torch
    #     import torch.distributed as dist
    #
    #     kwargs_launch={"backend":'nccl' ,"init_method":'env://', "world_size": idr_torch.size,"rank":idr_torch.rank}
    # else:
    #     fabric = L.Fabric(accelerator=args.device, devices=args.num_devices, strategy="ddp", main-address=)

    # fabric.launch(function=, **kwargs_launch)

    fabric.launch()
    print("start launch")
    fabric.barrier()
    print("end launch")


    return fabric
    # if args.device != "cpu":
    #     torch.cuda.init()


def run_forward(args, x, net):
    if args.split_fwd:
        x1, x2 = x.split(x.shape[0] // 2)
        rep1 = net(x1)
        rep2 = net(x2)
        return torch.cat((rep1, rep2), dim=0)
    return net(x)



def init_target_net(net, net_target):
    # initialize target network
    for param_online, param_target in zip(net.parameters(), net_target.parameters()):
        param_target.data.copy_(param_online.data)  # initialize
        param_target.requires_grad = False  # not update by gradient


def get_dataset_kwargs(d):
    d_new = deepcopy(d)
    del d_new["class"]
    return d_new


def is_target_needed(args):
    if args.main_loss in ['BYOL'] or "byol" in args.modules:
        return True
    return False


def update_target_net(net, net_target, tau):
    """
    function used to update the target net parameters to follow the running exponential average of online network.
        net: online network
        net_target: target network
        tau: hyper-parameter that controls the update
    """
    for param, target_param in zip(net.parameters(), net_target.parameters()):
        target_param.data.copy_((1 - tau) * param.data + tau * target_param.data)


def save_model(fabric, net, log_dir, epoch, optimizer=None, scheduler=None):
    """
        function used to save model parameters to the log directory
            net: network to be saved
            writer: summary writer to get the log directory
            epoch: epoch indicator
    """
    path = os.path.join(log_dir, 'models')
    if fabric.global_rank == 0:
        if not os.path.exists(path):
            os.mkdir(path)
    obj = {}
    obj["model"] = net
    if optimizer is not None:
        obj["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        obj["scheduler"] = scheduler.state_dict()
    fabric.save(os.path.join(path, f'epoch_{epoch}.pt'), obj)


def load_model(fabric, net, args, optimizer=None, scheduler=None,strict=True):
    checkpoint = fabric.load(os.path.join(args.path_load_model, "epoch_" + str(args.epoch_load_model)) + ".pt")
    net.load_state_dict(checkpoint["model"],strict=strict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        # optimizer = torch.load(checkpoint["optimizer"])
    if args.cosine_decay and scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])


def random_patch_selection(x_sel):
    pos_pair_ind = torch.randint(0, x_sel.shape[2] * x_sel.shape[3], (x_sel.shape[0], 1, 1), device=x_sel.device)
    pos_pair = x_sel.view(x_sel.shape[0], x_sel.shape[1], -1)
    pos_pair_ind = pos_pair_ind.expand((pos_pair.shape[0], pos_pair.shape[1], 1))
    gathering = torch.gather(pos_pair, 2, pos_pair_ind).squeeze()
    return gathering, pos_pair_ind


def get_neighbors(args, x, pos_pair_ind):
    pos_pair_ind_x = torch.div(pos_pair_ind,x.shape[3], rounding_mode="floor")
    pos_pair_ind_y = pos_pair_ind%x.shape[3]

    shift_ind_x = torch.clip(pos_pair_ind_x + shift1, min=0,max=x.shape[2]-1)
    shift_ind_y = torch.clip(pos_pair_ind_y + shift2, min=0,max=x.shape[3]-1)

    mask1 = (shift_ind_x == pos_pair_ind_x)
    mask1 = mask1.any(dim=1).squeeze()
    if mask1.nelement() != 0:
        shift_ind_x[mask1] += -shift1

    mask2 = (shift_ind_y == pos_pair_ind_y)
    mask2 = mask2.any(dim=1).squeeze()
    if mask2.nelement() != 0:
        shift_ind_y[mask2] += -shift2
    pos_pair = x.view(x.shape[0], x.shape[1], -1)
    pos_pair_ind_all = x.shape[2]*shift_ind_x+shift_ind_y
    return torch.gather(pos_pair, 2, pos_pair_ind_all).squeeze()

def retrieve_channel_number(layer_name, net):
    # We assume that  if this is a relu, we should check the output channel numberoff the block of the relu
    # This is true for resnet blocks but may be wrong in other cases.
    hierarchy = layer_name.split(".")
    current_name = hierarchy[-1]
    # if len(hierarchy) == 1 and current_name.startswith("conv"):
    #     return getattr(net.get_submodule("conv1"), "out_channels")
    if current_name.startswith("relu"):
        return retrieve_channel_number_loop(".".join(hierarchy[:-1]), net, 0)
    if current_name.startswith("conv"):
        return getattr(net.get_submodule(layer_name), "out_channels")
    if current_name.startswith("bn"):
        return getattr(net.get_submodule(layer_name), "num_features")
    # if hierarchy[0].startswith("conv"):
    #     return getattr(net.get_submodule(layer_name), "out_channels")
    # if hierarchy[0].startswith("bn"):
    #     return getattr(net.get_submodule(layer_name), "num_features")
    return retrieve_channel_number_loop(layer_name, net, 0)


def retrieve_channel_number_loop(layer_name, net, maximum=0):
    m = net.get_submodule(layer_name)
    for name, module in m.named_children():
        if name.startswith("conv"):
            maximum = max(maximum, getattr(module,"out_channels"))
        if name.startswith("bn"):
            maximum = max(maximum, getattr(module,"num_features"))
        maximum = max(maximum, retrieve_channel_number_loop(layer_name+"."+name, net, maximum))
    return maximum


#@torch.no_grad()
def normalize(x):
    return (x - x.mean(0, keepdim=True)) / (x.var(dim=0, keepdim=True) + 1e-5).sqrt()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        print(v, type(v))
        raise Exception('Boolean value expected.')

def str2table(v):
    return v.split(',')

# _____________________________________________________________________________

# Stick to 80 characters per line
# Use PEP8 Style
# Comment your code

# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
