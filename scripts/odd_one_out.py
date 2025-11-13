import datetime
import json
import os
from argparse import Namespace

import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from utils.augmentations import get_transformations
from utils.constants import DATASETS
from utils.finetuning import finetune
from utils.general import prepare_device, load_model
from utils.getters import get_networks, get_datasets
from utils.logger import EpochLogger


def get_test_embeddings(net, val_t, dataloader_test):
    latent_rep = []
    proj_rep = []
    proj_rep2 = []
    context_labels = []
    for i_batch, batch in tqdm(enumerate(dataloader_test)):
        # print(batch[0][0])
        # print(i_batch)
        e = net(val_t(batch[0][0]))
        context_labels.append(batch[2])
        latent_rep.append(torch.clone(e))
        for l in range(len(net.projector.net)):
            e = net.projector.net[l](e)
            if l == 2:
                proj_rep.append(torch.clone(e))
            if l == 5:
                proj_rep2.append(e)
                break
    # return torch.cat(latent_rep, dim=0), torch.cat(proj_rep, dim=0), torch.cat(proj_rep2, dim=0), torch.cat(context_labels, dim=0)
    return torch.cat(latent_rep, dim=0).cpu(), torch.cat(proj_rep, dim=0).cpu(), torch.cat(proj_rep2, dim=0).cpu(), torch.cat(context_labels, dim=0).cpu()

def compute_metric(args, fabric, contexts, context_values, e):
    count, cos_sim, euc_sim = 0, 0, 0
    for c in context_values:
        mask_in = contexts == c
        e_in = e[mask_in]
        e_out = e[~mask_in]

        # dataset = TensorDataset(e_in)
        # dataloader_test = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.n_workers)
        # dataloader_test = fabric.setup_dataloaders(dataloader_test, move_to_device=True)
        for n in range(5):
            for e_curr in iter(e_in.split(args.batch_size)):
            # for data in dataloader_test:
                same_idx = torch.randint(0, e_in.shape[0], (e_curr.shape[0],), dtype=torch.long, device=e_in.device)
                diff_idx = torch.randint(0, e_out.shape[0], (e_curr.shape[0],), dtype=torch.long, device=e_in.device)

                e_same = e_in[same_idx]
                e_diff = e_out[diff_idx]
                cos_sim += (torch.nn.functional.cosine_similarity(e_curr, e_same,dim=1) >= torch.nn.functional.cosine_similarity(e_curr,e_diff,dim=1)).float().sum()
                euc_sim += (torch.norm(e_curr - e_same, dim=1) <= torch.norm(e_curr - e_diff, dim=1)).float().sum()
                count += e_curr.size(0)


    # all_euc = fabric.all_reduce(euc_sim, reduce_op="sum")
    # all_cos = fabric.all_reduce(cos_sim, reduce_op="sum")
    # all_cpt = fabric.all_reduce(count, reduce_op="sum")
    # return all_euc, all_cos, all_cpt
    return euc_sim, cos_sim, count

@torch.no_grad()
def odd_one_out(args):

    if args.path_load_model != "":
        print(f"LOAD {os.path.join(args.path_load_model,'../config.json')}")
        with open(os.path.join(args.path_load_model,'../config.json'), 'r') as f:
            config = json.load(f)
            config = config[next(iter(config))]
            new_args = Namespace(**config)
            new_args.data_root = args.data_root
            new_args.log_dir = args.log_dir
            args = new_args

    fabric = prepare_device(args)
    run_name = f'{datetime.datetime.now().strftime("%d-%m-%y_%H-%M")}_{args.name}_{args.seed}'
    dataloader_train, dataloader_train_eval, dataloader_test, train_set, dataset_train_eval, dataset_test = get_datasets(args, run_name, fabric, logger=False)
    net, _, method_modules = get_networks(args, fabric, train_set)
    net = fabric.setup(net)
    # dataloader_train, dataset_train_eval, dataloader_test = fabric.setup_dataloaders(dataloader_train, dataloader_train_eval, dataloader_test, move_to_device=True)
    _, dataloader_test = fabric.setup_dataloaders(dataloader_train, dataloader_test, move_to_device=False)

    if args.path_load_model != "":
        load_model(fabric, net, args, strict=True)
    _, val_t = get_transformations(args, crop_size=DATASETS[args.dataset]['img_size'])
    save_dir = os.path.join(args.log_dir, run_name)

    # Prepare model
    if fabric.global_rank == 0:
        logger = EpochLogger(output_dir=save_dir, exp_name="Seed-" + str(args.seed), output_fname='odd_one_out.txt')
    print(val_t)
    en1, en2, en3, contextsn = get_test_embeddings(net, val_t, dataloader_test)

    alls = fabric.all_gather((en1, en2, en3, contextsn))
    e1, e2, e3, contexts = alls

    e1 = e1.view(-1, en1.shape[1])
    e2 = e2.view(-1, en2.shape[1])
    e3 = e3.view(-1, en3.shape[1])
    contexts = contexts.view(-1)

    if fabric.global_rank == 0:
        # contexts = torch.randint(0, 8, (contexts.shape[0],), dtype=torch.long, device=contexts.device)
        context_values = torch.unique(contexts)
        all_e = [e1, e2, e3]
        for i, e in enumerate(all_e):
            all_euc, all_cos, all_cpt = compute_metric(args, fabric, contexts, context_values, e)
            if fabric.global_rank == 0:
                logger.log_tabular(f"euc{i}", (all_euc/all_cpt).item())
                logger.log_tabular(f"cos{i}", (all_cos/all_cpt).item())
                logger.log_tabular(f"count{i}", all_cpt)
        if fabric.global_rank == 0: logger.dump_tabular()

