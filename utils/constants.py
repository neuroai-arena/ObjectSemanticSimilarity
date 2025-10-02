import torch

from utils.datasets import MVImgNet
from utils.losses import SimCLR, BYOL, MultiPositiveSimCLR, Rince, VicReg
from torch.nn import functional as F

DATASETS = {
    "MVImgNet": {
        'class': MVImgNet,
        'size': 6500000,
        'img_size': (224, 224),
        'rgb_mean': (0.485, 0.456, 0.406),
        'rgb_std': (0.229, 0.224, 0.225),
        'action_size': 8
    }
}


SIMILARITY_FUNCTIONS = {
    'cosine': lambda x, x_pair: F.cosine_similarity(x.unsqueeze(1), x_pair.unsqueeze(0), dim=2),
    'RBF': lambda x, x_pair: -torch.cdist(x, x_pair)
}

SIMILARITY_FUNCTIONS_SIMPLE = {
    'cosine': lambda x, x_pair: F.cosine_similarity(x, x_pair, dim=1),
    'RBF': lambda x, x_pair: -torch.norm(x - x_pair, 2)
}

# loss dictionary for different losses
LOSS = {
    'SimCLR': SimCLR,
    'BYOL': BYOL,
    'MultiPosSimclr': MultiPositiveSimCLR,
    'Rince': Rince,
    'VicReg': VicReg
}