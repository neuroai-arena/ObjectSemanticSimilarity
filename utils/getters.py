import copy
import random

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR, SequentialLR, LambdaLR
from torchvision import transforms as T

from modules.labels_ssl import LabelSSL
from modules.linear_eval import OnlineLinearEval
from modules.ssl import Ssl
from networks.resnets import resnet18, resnet18_cifar, resnet50, resnet50_cifar
from utils.augmentations import get_transformations, CenterCropAndResize
from torch.utils.data import DataLoader
from tqdm import tqdm

# get transformations for validation and for training
from utils.constants import DATASETS, LOSS
from utils.general import get_dataset_kwargs, is_target_needed
from utils.lars import create_optimizer_lars, LARS2

from utils.losses import SimCLR


class RandomCenterCrop:
    def __init__(self, size=224, scale=(0.08, 1.0)):
        self.scale=scale
        self.size=size

    def __call__(self, img):
        crop_shape = int(random.uniform(self.scale[0], self.scale[1]))
        out = T.functional.center_crop(img, [crop_shape*self.size[0], crop_shape**self.size[1]])
        out = T.functional.resize(out, self.size)
        return out

def get_datasets(args, run_name, fabric, logger=True):

    transform_train = []
    transform_test_eval=[]
    if args.crop_first:
        # if not args.crop_center:
        transform_train.append(T.RandomResizedCrop(size=DATASETS[args.dataset]['img_size'], scale=(args.min_crop, args.max_crop)))
        # else:
        #     transform_train.append(RandomCenterCrop(size=DATASETS[args.dataset]['img_size'],scale=(args.min_crop, args.max_crop)))

        transform_test_eval.append(CenterCropAndResize(proportion=DATASETS[args.dataset]['img_size'][0]/256, size=DATASETS[args.dataset]['img_size'][0]))

    if not args.crop_first and not args.kornia and args.contrast != 'time':
        if args.min_crop != 1 and not args.one_crop:
            transform_train.append(T.RandomResizedCrop(size=DATASETS[args.dataset]['img_size'], scale=(args.min_crop, args.max_crop)))
        if args.flip:
            transform_train.append(T.RandomHorizontalFlip(p=0.5))
        if args.jitter != 0 and not args.unijit:
            s = args.jitter_strength
            transform_train.append(T.RandomApply([T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=args.jitter))
        if args.grayscale and not args.unijit:
            transform_train.append(T.RandomGrayscale(p=0.2))

    transform_train.append(T.ToTensor())
    transform_test_eval.append(T.ToTensor())

    remove_background = args.remove_background
    args.remove_background = False
    dataset_train = DATASETS[args.dataset]['class'](
        args=args,
        run_name=None,
        split='train',
        transform=T.Compose(transform_train),
        contrastive=True if (args.contrast in ['time','combined']) else False,
        fabric=fabric,
        **get_dataset_kwargs(DATASETS[args.dataset])
    )
    args.remove_background=remove_background

    # dataset_train_eval = DATASETS[args.dataset]['class'](
    #     args=args,
    #     run_name=None,
    #     split='train',
    #     transform=T.Compose(transform_test_eval),
    #     contrastive=False,
    #     fabric=fabric,
    #     eval = True,
    #     **get_dataset_kwargs(DATASETS[args.dataset])
    # )
    # dataloader_train_eval = DataLoader(dataset_train_eval, batch_size=args.batch_size, num_workers=0, shuffle=not args.drop_last_test, drop_last=args.drop_last_test, pin_memory=True)
    dataloader_train_eval=None
    dataset_train_eval=None


    dataset_test = DATASETS[args.dataset]['class'](
        args=args,
        run_name=run_name,
        split='test',
        transform=T.Compose(transform_test_eval),
        contrastive=False,
        fabric=fabric,
        logger=logger,
        **get_dataset_kwargs(DATASETS[args.dataset])
    )
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size,num_workers=args.n_workers, shuffle=not args.drop_last_test, drop_last=args.drop_last_test, pin_memory=True)


    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,num_workers=args.n_workers, shuffle=True, drop_last=True, pin_memory=True)

    # train_transform_after = transforms.Lambda(lambda x: torch.stack([train_transform(x_) for x_ in x]))
    return dataloader_train, dataloader_train_eval, dataloader_test, dataset_train, dataset_train_eval, dataset_test


def get_train_iterator(args, dataloader_train):
    return tqdm(dataloader_train)

def get_inputs_channels(args):
    num = 3
    if args.dataset in ["Stereo3DShapeE"] and args.stereo_layer == "pixels":
        num *= 2
    # if args.pe:
    #     num += 2
    return num


def apply_transform(args, transform, x, x_pair):
    if args.one_aug:
        return torch.cat([x, transform(x_pair)], 0)
    return transform(torch.cat([x, x_pair], 0))

def get_network(args):
    kwargs = {"input_channels": get_inputs_channels(args)}

    if args.model == "resnet18":
        model = resnet18(**kwargs)
    elif args.model == "resnet50":
        model = resnet50(**kwargs)
    else:
        raise Exception(args.model, "not found")


    #### Add stereo visual fusion at an arbitrary stereo_layer
    if args.dataset in ["Stereo3DShapeE"]:
        if args.stereo_layer != "pixels":
            model.register_forward_pre_hook(lambda m, x: x[0].view(2 * x[0].shape[0], 3, x[0].shape[2], x[0].shape[3]))
            if "stereo" not in args.modules:
                model.get_submodule(args.stereo_layer).register_forward_hook(fusion_hook)

    return model

def get_networks(args, fabric, dataset_train):
    net = get_network(args)
    method_modules = get_modules(args, fabric, net=net, train_dataset=dataset_train)
    if args.num_devices > 1:
        # pytorch_lightning.plugins.
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

    if fabric.global_rank == 0:
        print(net)

    net_target = None
    if is_target_needed(args):
        net_target = copy.deepcopy(net)
        if args.compile:
            net_target = torch.compile(net_target)
        net_target.train()
        net_target = fabric.setup(net_target)

    if args.compile:
        net = torch.compile(net)

    return net, net_target, method_modules


def get_optimizer(args, net):
    # for i, k in net.named_parameters():
    #     print(i)
    #     print(k)
    if args.optimizer == "adam":
        return torch.optim.Adam(net.parameters(), lr=args.lrate, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(net.parameters(), lr=args.lrate, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(net.parameters(), lr=args.lrate, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
    if args.optimizer == "lars":
        # return create_optimizer_lars(net.parameters(), args.lrate, 0.9, args.weight_decay, True, 0)
        return create_optimizer_lars(net, args.lrate, 0.9, args.weight_decay, True, 0)
    if args.optimizer == "lars2":
        def exclude_from_wd_and_adaptation(name):
            if 'bn' in name:
                return True
            if args.optimizer == 'lars2' and 'bias' in name:
                return True
        param_groups = [
            {
                'params': [p for name, p in net.named_parameters() if not exclude_from_wd_and_adaptation(name)],
                'weight_decay': args.weight_decay,
                'layer_adaptation': True,
            },
            {
                'params': [p for name, p in net.named_parameters() if exclude_from_wd_and_adaptation(name)],
                'weight_decay': 0.,
                'layer_adaptation': False,
            },
        ]
        return LARS2(torch.optim.SGD(param_groups, lr=args.lrate, weight_decay=args.weight_decay, momentum=0.9),net.parameters(),None)
        # return torch.optim.LARS(net.parameters(), lr=args.lrate, weight_decay=args.weight_decay, momentum=0.9)

def get_scheduler(args, optimizer, n_epochs):
    # decrease learning rate by a factor of 0.3 every 10 epochs
    # scheduler = StepLR(optimizer, 10, 0.3)
    if args.cosine_decay:
        #eta_min: minimum learning rate
        # scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=args.lrate * (args.lrate_decay ** 3))
        # scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=args.eta_min, last_epoch=-1)
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=args.eta_min, last_epoch=-1)
    else:
        scheduler = ExponentialLR(optimizer, 1.0)

    if args.warmup:
        def warmup(current_epoch):
            # return 1 / (10 ** (float(args.warmup - current_epoch)))
            return (1+current_epoch) / args.warmup
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup)
        scheduler = SequentialLR(optimizer, [warmup_scheduler, scheduler], [args.warmup])
    return scheduler


MODULES = {
    "classic": Ssl,
    "labels": LabelSSL,
    "linear_eval": OnlineLinearEval,
}

def get_modules(args, fabric,  **kwargs):
    modules = []
    for m in args.modules:
        modules.append(MODULES[m](args, fabric, **kwargs))
    return modules

def get_arguments(parser):
    for _, m in MODULES.items():
        parser = m.get_args(parser)
    for _, l in LOSS.items():
        parser = l.get_args(parser)
    for _,d in DATASETS.items():
        parser = d["class"].get_args(parser)
    return parser

