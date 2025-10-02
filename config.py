#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import argparse
import datetime

from utils.general import str2bool, str2table

# parse args that correspond to configurations to be experimented on
parser = argparse.ArgumentParser()

# General
parser.add_argument('--name',default='',type=str)
parser.add_argument('--dataset',default='MVImgNet',type=str)
parser.add_argument('--data_root',default='data', type=str)
parser.add_argument('--device',default='cuda', type=str)
parser.add_argument('--num_devices',default=8, type=int)
parser.add_argument('--num_nodes',default=1, type=int)
parser.add_argument('--seed',default=0, type=int)
parser.add_argument('--compile',default=False, type=str2bool)
parser.add_argument('--n_workers',default=4, type=int)
parser.add_argument('--mode',default="train", type=str)
parser.add_argument('--hdf5', default=True, type=str2bool)
parser.add_argument('--precision', default="mixed", type=str)

# Model
parser.add_argument('--modules', default=["classic"], type=str2table)
parser.add_argument('--model', default="resnet50", type=str)
parser.add_argument('--split_fwd', default=True, type=str2bool)

# Augmentations
parser.add_argument('--imgnet_stats', default=True, type=str2bool)
parser.add_argument('--aug_params', default=[], type=str2table)
parser.add_argument('--contrast', default='time', choices=['time', 'classic', 'combined'], type=str)
parser.add_argument('--crop_first', default=True, type=str2bool)
parser.add_argument('--crop_size', default=32, type=int)
parser.add_argument('--crop_ratio', default=4., type=float)
# parser.add_argument('--crop_center', default=False, type=str2bool)
parser.add_argument('--pcrop', default=1, type=float)
parser.add_argument('--one_crop', default=False, type=str2bool)
parser.add_argument('--min_crop', default=0.5, type=float)
parser.add_argument('--max_crop', default=1, type=float)
parser.add_argument('--flip', default=0, type=float)
parser.add_argument('--unijit', default=False, type=str2bool)
parser.add_argument('--punijit', default=1, type=float)
parser.add_argument('--jitter', default=0, type=float)
parser.add_argument('--jitter_strength', default=1, type=float)
parser.add_argument('--grayscale', default=0, type=float)
parser.add_argument('--solarize', default=0, type=float)
parser.add_argument('--blur', default=0, type=int)
parser.add_argument('--pblur', default=0.2, type=float)
parser.add_argument('--image_padding', default=0, type=int, help="Add some zero padding around the image")
parser.add_argument('--uni_aug', default=False, type=str2bool)
parser.add_argument('--one_aug', default=False, type=str2bool)
parser.add_argument('--rotation_range', default=2, type=int)
parser.add_argument('--max_views', default=180, type=int)
parser.add_argument('--canonical', default=1, type=int)
parser.add_argument('--planars', default=[], type=str2table)
parser.add_argument('--p_planar', default=1, type=float)
parser.add_argument('--kornia', default=True, type=str2bool)

# Optimizer hyperparameters
parser.add_argument('--optimizer',default="adamw", type=str)
parser.add_argument('--lrate',default=1e-3,type=float)
parser.add_argument('--weight_decay',default=0.000001,type=float)#Should be 1e-6
parser.add_argument('--cosine_decay',default=False, type=str2bool)
parser.add_argument('--last_epoch',default=-1, type=int)
parser.add_argument('--eta_min',default=1e-5, type=float)
parser.add_argument('--warmup',default=0, type=int)
parser.add_argument('--n_epochs',default=100,type=int)
parser.add_argument('--batch_size',default=64,type=int)
parser.add_argument('--finetune_labels',default=-1,type=int)


# Test/Evaluation
parser.add_argument('--test_every', default=10, type=int)
parser.add_argument('--knn_batch_size', default=64, type=int,
                    help="batch size used for knn evaluation, rules the trade-off between RAM - speed")
parser.add_argument('--finetune', default=True, type=str2bool)
parser.add_argument('--fine_eval', default=False, type=str2bool)
parser.add_argument('--fine_label', default=0, type=int)
parser.add_argument('--fine_proj', default=0, type=int)
parser.add_argument('--drop_last_test', default=True, type=str2bool, help="Makes the evaluation steps more stable on ROCm")

# Losses
parser.add_argument('--main_loss', default='SimCLR', choices=['SimCLR', 'BYOL', 'Rince', 'VicReg'], type=str)
parser.add_argument('--tau', default=0.996, type=float)
parser.add_argument('--similarity', default='cosine', choices=['cosine', 'RBF'], type=str)
parser.add_argument('--sampling_mode', default='interval', type=str)
parser.add_argument('--n_fix', default=2, type=int)

# Load/save
parser.add_argument('--path_load_model', default="", type=str)
parser.add_argument('--epoch_load_model', default=-1, type=int)
parser.add_argument('--save_model', default=True, type=str2bool)
parser.add_argument('--save_embedding', default=False, type=str2bool)
parser.add_argument('--log_dir', default="save", type=str)
parser.add_argument('--save_every', default=50, type=int)


# Actions used by action-based modules
parser.add_argument("--action_set", type=str, default="all")
parser.add_argument('--action_normalize', default=True, type=str2bool)

