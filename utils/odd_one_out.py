import datetime
import json
import os
from argparse import Namespace

import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from utils.augmentations import get_transformations
from utils.constants import DATASETS
from utils.datasets import MVImgNet
from utils.finetuning import finetune
from utils.general import prepare_device, load_model
from utils.getters import get_networks, get_datasets
from utils.logger import EpochLogger


def get_test_embeddings(net, val_t, dataloader_test):
    latent_rep = []
    proj_rep = []
    proj_rep2 = []
    proj_rep3 = []
    proj_rep4 = []
    proj_rep5 = []
    proj_rep6 = []
    context_labels = []
    category_labels = []
    obj_labels = []
    for i_batch, batch in tqdm(enumerate(dataloader_test)):
        # print(batch[0][0])
        # print(i_batch)
        e = net(val_t(batch[0][0]))
        context_labels.append(batch[2])
        category_labels.append(batch[1])
        obj_labels.append(batch[3])
        # print(batch[2][0].item(), MVImgNet.reverse_mapping[batch[1][0].item()])

        latent_rep.append(torch.clone(e))
        if hasattr(net, "projector"):
            p = e
            for l in range(len(net.projector.net)):
                p = net.projector.net[l](p)
                if l == 2:
                    proj_rep.append(torch.clone(p))
                if l == 5:
                    proj_rep2.append(torch.clone(p))
                if l == 6:
                    proj_rep3.append(p)
                    break
        else:
            proj_rep.append(e)
            proj_rep2.append(e)
            proj_rep3.append(e)

        if hasattr(net, "label_head"):
            p = e
            for l in range(len(net.label_head.net)):
                p = net.label_head.net[l](p)
                if l == 2:
                    proj_rep4.append(torch.clone(p))
                if l == 5:
                    proj_rep5.append(torch.clone(p))
                if l == 6:
                    proj_rep6.append(p)
                    break
        else:
            proj_rep4.append(e)
            proj_rep5.append(e)
            proj_rep6.append(e)
    # return torch.cat(latent_rep, dim=0), torch.cat(proj_rep, dim=0), torch.cat(proj_rep2, dim=0), torch.cat(context_labels, dim=0)
    return (torch.cat(latent_rep, dim=0), torch.cat(proj_rep, dim=0), torch.cat(proj_rep2, dim=0),
            torch.cat(proj_rep3, dim=0), torch.cat(proj_rep4, dim=0), torch.cat(proj_rep5, dim=0),
            torch.cat(proj_rep6, dim=0), torch.cat(context_labels, dim=0), torch.cat(category_labels, dim=0),
            torch.cat(obj_labels, dim=0))

def compute_metric_objects(args, contexts, categories, objects, context_values, e):
    count, cos_sim, sparsity, collapse, norms = 0, 0, 0, 0, 0
    for c in context_values:
        mask_cin = contexts == c
        cat_values = torch.unique(categories[mask_cin])
        for cat in cat_values:
            mask_ccin = mask_cin & (categories == cat)
            objects_values = torch.unique(objects[mask_ccin])
            obj_values = torch.unique(objects_values)
            for o in obj_values:
                mask_in = mask_ccin & (objects == o)
                mask_out = mask_ccin & (objects != o)

                e_in = e[mask_in]
                e_out = e[mask_out]
                #We run the procedure n times to compare a given embeddings to more other embeddings
                for n in range(10):
                    for e_curr in iter(e_in.split(args.batch_size)):
                    # for data in dataloader_test:
                        same_idx = torch.randint(0, e_in.shape[0], (e_curr.shape[0],), dtype=torch.long, device=e_in.device)
                        diff_idx = torch.randint(0, e_out.shape[0], (e_curr.shape[0],), dtype=torch.long, device=e_in.device)

                        e_same = e_in[same_idx]
                        e_diff = e_out[diff_idx]

                        cos_sim1 = (torch.nn.functional.cosine_similarity(e_curr, e_same,dim=1) > torch.nn.functional.cosine_similarity(e_curr,e_diff,dim=1))
                        cos_sim2 = (torch.nn.functional.cosine_similarity(e_curr, e_same,dim=1) > torch.nn.functional.cosine_similarity(e_same,e_diff,dim=1))
                        cos_sim += (cos_sim1 & cos_sim2).float().sum()

                        count += e_curr.size(0)
                        sparsity += e_curr.count_nonzero().float().sum()
                        collapse += (torch.norm(e_curr, dim=1) > 0.1).float().sum()
                        norms += torch.norm(e_curr).sum()
    return sparsity, cos_sim, count, collapse, norms

def compute_metric_category(args, contexts, categories, context_values, e):
    count, cos_sim, sparsity, collapse, norms = 0, 0, 0, 0, 0
    for c in context_values:
        mask_cin = contexts == c
        cat_in = categories[mask_cin]
        cat_values = torch.unique(cat_in)
        for cat in cat_values:
            mask_in = mask_cin & (categories == cat)
            mask_out = mask_cin & (categories != cat)
            e_in = e[mask_in]
            e_out = e[mask_out]

            #We run the procedure n times to compare a given embeddings to more other embeddings
            for n in range(10):
                for e_curr in iter(e_in.split(args.batch_size)):
                # for data in dataloader_test:
                    same_idx = torch.randint(0, e_in.shape[0], (e_curr.shape[0],), dtype=torch.long, device=e_in.device)
                    diff_idx = torch.randint(0, e_out.shape[0], (e_curr.shape[0],), dtype=torch.long, device=e_in.device)

                    e_same = e_in[same_idx]
                    e_diff = e_out[diff_idx]

                    cos_sim1 = (torch.nn.functional.cosine_similarity(e_curr, e_same,dim=1) > torch.nn.functional.cosine_similarity(e_curr,e_diff,dim=1))
                    cos_sim2 = (torch.nn.functional.cosine_similarity(e_curr, e_same,dim=1) > torch.nn.functional.cosine_similarity(e_same,e_diff,dim=1))
                    cos_sim += (cos_sim1 & cos_sim2).float().sum()

                    count += e_curr.size(0)
                    sparsity += e_curr.count_nonzero().float().sum()
                    collapse += (torch.norm(e_curr, dim=1) > 0.1).float().sum()
                    norms += torch.norm(e_curr).sum()
    return sparsity, cos_sim, count, collapse, norms


def compute_metric_context(args, contexts, context_values, e):
    count, cos_sim, sparsity, collapse, norms = 0, 0, 0, 0, 0
    for c in context_values:
        mask_in = contexts == c
        e_in = e[mask_in]
        e_out = e[~mask_in]
        # We run the procedure n times to compare a given embeddings to more other embeddings
        for n in range(10):
            for e_curr in iter(e_in.split(args.batch_size)):
                # for data in dataloader_test:
                same_idx = torch.randint(0, e_in.shape[0], (e_curr.shape[0],), dtype=torch.long, device=e_in.device)
                diff_idx = torch.randint(0, e_out.shape[0], (e_curr.shape[0],), dtype=torch.long, device=e_in.device)

                e_same = e_in[same_idx]
                e_diff = e_out[diff_idx]

                cos_sim1 = (torch.nn.functional.cosine_similarity(e_curr, e_same,
                                                                  dim=1) > torch.nn.functional.cosine_similarity(e_curr,
                                                                                                                 e_diff,
                                                                                                                 dim=1))
                cos_sim2 = (torch.nn.functional.cosine_similarity(e_curr, e_same,
                                                                  dim=1) > torch.nn.functional.cosine_similarity(e_same,
                                                                                                                 e_diff,
                                                                                                                 dim=1))
                cos_sim += (cos_sim1 & cos_sim2).float().sum()

                count += e_curr.size(0)
                sparsity += e_curr.count_nonzero().float().sum()
                collapse += (torch.norm(e_curr, dim=1) > 0.1).float().sum()
                norms += torch.norm(e_curr).sum()
    return sparsity, cos_sim, count, collapse, norms


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
            new_args.num_devices = args.num_devices
            new_args.batch_size = args.batch_size
            new_args.path_load_model = args.path_load_model
            new_args.epoch_load_model = args.epoch_load_model
            new_args.device = args.device
            new_args.remove_background = args.remove_background
            new_args.precision = args.precision
            new_args.fine_label = args.fine_label
            # new_args.context_json = args.context_json
            # new_args.hdf5 = args.hdf5
            # new_args.sampling_mode = args.sampling_mode
            args = new_args
    # print(args.fine_label)
    #args.fine_label = args.fine_label + 2
    # print("FINE LABEL", args.fine_label)
    fabric = prepare_device(args)
    run_name = f'{datetime.datetime.now().strftime("%d-%m-%y_%H-%M")}_{args.name}_{args.seed}'
    dataloader_train, dataloader_train_eval, dataloader_test, train_set, dataset_train_eval, dataset_test = get_datasets(args, run_name, fabric, logger=False)
    net, _, _ = get_networks(args, fabric, train_set)
    net = fabric.setup(net)
    # dataloader_train, dataset_train_eval, dataloader_test = fabric.setup_dataloaders(dataloader_train, dataloader_train_eval, dataloader_test, move_to_device=True)
    # _, dataloader_test = fabric.setup_dataloaders(dataloader_train, dataloader_test, move_to_device=True)
    dataloader_test = fabric.setup_dataloaders(dataloader_test, move_to_device=True)

    if args.path_load_model != "":
        load_model(fabric, net, args, strict=True)

    _, val_t = get_transformations(args, crop_size=DATASETS[args.dataset]['img_size'])
    # save_dir = os.path.join(args.log_dir, run_name)
    save_dir = os.path.join(args.path_load_model,'../')

    if fabric.global_rank == 0:
        logger = EpochLogger(output_dir=save_dir, exp_name="Seed-" + str(args.seed), output_fname=f'odd_one_out{"_"+str(args.fine_label) if args.fine_label != 2 else ""}{"_noback" if args.remove_background else ""}.txt')

    #We pre-compute all embeddings at different layers
    net.eval()
    en1, en2, en3, en4, en5, en6, en7, contextsn, catsn, objsn = get_test_embeddings(net, val_t, dataloader_test)

    alls = fabric.all_gather((en1, en2, en3, en4, en5, en6, en7, contextsn, catsn, objsn))
    # e1, e2, e3, e4, e5, e6, e7, contexts = alls

    e = []
    for i in range(len(alls)-3):
        if not hasattr(net, "projector") and i < 4 and i > 0:
            e.append("pass")
            continue
        elif not hasattr(net, "label_head") and i >= 4:
            e.append("pass")
            continue
        e.append(alls[i].view(-1, alls[i].shape[2]).cpu())

    contexts = alls[-3].view(-1).cpu()
    categories = alls[-2].view(-1).cpu()
    objs = alls[-1].view(-1).cpu()


    if fabric.global_rank == 0:
        # contexts = torch.randint(0, 8, (contexts.shape[0],), dtype=torch.long, device=contexts.device)
        context_values = torch.unique(contexts)
        for i, e in enumerate(e):
            if e == "pass":
                continue
            if args.fine_label == 2:
                all_sparsity, all_cos, all_cpt, all_coll, all_norms = compute_metric_context(args, contexts, context_values, e)
            if args.fine_label == 1:
                all_sparsity, all_cos, all_cpt, all_coll, all_norms = compute_metric_category(args, contexts, categories, context_values, e)
            if args.fine_label == 3:
                all_sparsity, all_cos, all_cpt, all_coll, all_norms = compute_metric_objects(args, contexts, categories, objs, context_values, e)
            if fabric.global_rank == 0:
                logger.log_tabular(f"spars{i}", (all_sparsity/all_cpt).item())
                logger.log_tabular(f"cos{i}", (all_cos/all_cpt).item())
                logger.log_tabular(f"coll{i}", (all_coll/all_cpt).item())
                logger.log_tabular(f"norm{i}", (all_norms/all_cpt).item())
                logger.log_tabular(f"count{i}", all_cpt)
        if fabric.global_rank == 0: logger.dump_tabular()

