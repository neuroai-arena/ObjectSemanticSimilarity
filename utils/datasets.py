import glob
import io
import json
import math
import os
import time
import random
from functools import cached_property

import scipy
import h5py
import pandas as pd
import numpy as np
import torch
import torchvision
from numpy.random import choice
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from uuid import uuid4
from collections import defaultdict

import config

from utils.evaluation import get_representations, lls_fit, lls_eval
from utils.general import str2bool, str2table
from utils.logger import EpochLogger


class SimpleDataset(Dataset):

    def __init__(self, args, run_name, split='train', transform=None, target_transform=None, contrastive=True,
                 logger=True, fabric=None, eval=False, **kwargs):
        self.args = args
        self.contrastive = contrastive
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.fabric = fabric
        self.run_name = run_name
        self.eval_mode = eval

        if split == "test" and logger and fabric.global_rank == 0:
            self.epoch_logger = EpochLogger(output_dir=os.path.join(args.log_dir, run_name),
                                            exp_name="Seed-" + str(args.seed), output_fname='progress.txt')
            self.epoch_logger.save_config(args)
        if self.args.unijit:
            s = self.args.jitter_strength
            jit = transforms.RandomApply([transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)],
                                         p=self.args.jitter)
            grayscale = transforms.RandomGrayscale(p=0.2)
            self.jit = transforms.Compose([jit, grayscale])
        # if self.args.one_crop:
        #     self.crop = []
        #     if self.args.image_padding:
        #         self.crop.append(transforms.Pad(self.args.image_padding))
        #     self.crop.append(transforms.RandomApply([transforms.RandomResizedCrop(size=128, scale=(self.args.min_crop, self.args.max_crop))],p=self.args.pcrop))
        #     self.crop = transforms.Compose(self.crop)
        self.time = time.time()

    @classmethod
    def get_args(cls, parser):
        return parser

    @torch.no_grad()
    def eval(self, net, dataloader_train_eval, dataloader_test, epoch=0, modules=[], tv=None, **kwargs):
        if self.fabric.global_rank == 0:
            test_time = time.time()

        data_test = get_representations(self.args, net, dataloader_test, tv)
        data_test = self.fabric.all_gather(data_test)

        if self.fabric.global_rank == 0:
            self.epoch_logger.log_tabular("epoch", epoch)
            self.epoch_logger.log_tabular("all_time", (time.time() - self.time) / self.args.test_every)
            self.epoch_logger.log_tabular("test_time", (time.time() - test_time) / self.args.test_every)

            f_l_test = {}
            if "0" in self.args.eval_labels:
                f_l_test["0"] = {"features": data_test[0], "labels": data_test[1]}

            if "1" in self.args.eval_labels:
                f_l_test["1"] = {"features": data_test[0], "labels": data_test[2]}

            for m in modules:
                for k, v in m.eval(net, f_l_test).items():
                    # for k, v in m.eval(net, net).items():
                    self.epoch_logger.log_tabular(k, v)
            self.time = time.time()
            self.epoch_logger.dump_tabular()
        self.fabric.barrier()

    def get_log_dir(self):
        return os.path.join(self.args.log_dir, self.run_name)

class MVImgNet(SimpleDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.args.remove_background:
            self.hdf5_file = h5py.File(os.path.join(self.args.data_root, 'masked_dataset', f"{self.split}_linker.h5"),
                                       "r")
            self.dataset = pd.read_parquet(os.path.join(self.args.data_root, 'masked_dataset', f"dataset_{self.split}.parquet"))
        else:
            if self.args.hdf5 and self.args.hdf5_mode == "category":
                # self.hdf5_file = h5py.File(os.path.join(self.args.data_root, "data2.h5"), "r")
                if not os.path.exists(os.path.join(self.args.data_root, "data_all.h5")):
                    raise Exception("You should build the merged data hdf5 file")
                self.hdf5_file = h5py.File(os.path.join(self.args.data_root, "data_all.h5"), "r")
                split_rep = "" if self.split == "train" else f"_{self.split}"
                file_list = [os.path.join(self.args.data_root, f"dataset_{c}{split_rep}.parquet") for c in category_list]
                self.dataset = pd.concat([pd.read_parquet(f) for f in file_list if os.path.exists(f)], ignore_index=True)
            elif self.args.hdf5 and self.args.hdf5_mode == "partition":
                if not os.path.exists(os.path.join(self.args.data_root, "data_all.h5")):
                    raise Exception("You should build the merged data hdf5 file")
                self.hdf5_file = h5py.File(os.path.join(self.args.data_root, "data_all.h5"), "r")
                self.dataset = pd.read_parquet(os.path.join(self.args.data_root, f"dataset_{self.split}_all3.parquet"))
            elif self.args.hdf5:
                self.hdf5_file = h5py.File(os.path.join(self.args.data_root, "data2.h5"), "r")
                self.dataset = pd.read_parquet(os.path.join(self.args.data_root, "dataset_" + self.split + "2.parquet"))
            else:
                self.dataset = pd.read_parquet(os.path.join(self.args.data_root, "dataset_" + self.split + "2.parquet"))
                # self.dataset = pd.read_csv(os.path.join(self.args.data_root, f"datasetT_0.1_1.csv"))

            # Map each category to its context and remove all object with 'Uncertain' category
            context_path = os.path.join('./utils/', getattr(self.args, 'context_json', 'room_assignment.json'))
            contexts_json = json.load(open(context_path, "r"))
            self.cat_to_context = {}
            uncertain_ctx_id, num_uncertain_cats = None, None
            for context_int, (cont_name, cont_cats) in enumerate(contexts_json.items()):
                print(context_int, cont_name)
                for cat in cont_cats:
                    self.cat_to_context[int(cat)] = context_int
                if cont_name == "Uncertain":
                    uncertain_ctx_id = context_int
                    num_uncertain_cats = len(cont_cats)

            self.context_to_string = {i: c for i, c in enumerate(contexts_json.keys())}
            self.dataset["context"] = self.dataset['category'].apply(lambda c: self.cat_to_context[c])
            self.num_contexts = len(contexts_json) - 1
            assert uncertain_ctx_id is not None, "Uncertain category not found"
            print(f"Found {self.num_contexts} contexts.")

            cats_not_in_split = set(self.cat_to_context.keys()) - set(self.dataset["category"])
            print(f"Warning: {len(cats_not_in_split)} categories not found in split {self.split}. "
                  f"{len(cats_not_in_split.intersection(set(contexts_json['Uncertain'])))} of them are 'Uncertain' and can be ignored.")

            # Remove all objects with 'Uncertain' category
            num_cats_old = self.dataset["category"].nunique()
            self.dataset = self.dataset.loc[self.dataset["context"] != uncertain_ctx_id].reset_index(drop=True)
            self.categories = self.dataset["category"].unique()
            print(f"Removed {num_uncertain_cats} uncertain categories. Num. cats. {num_cats_old} -> {len(self.categories)}")
            # assert num_cats_old - num_uncertain_cats + len(cats_not_in_split) == len(self.categories), \
            #     "Some categories were lost during removal of uncertain categories. Maybe some categories are not present in the split?"

            if self.args.imgnet_subset and self.split == "train":
                if self.args.imgnet_subset_balanced_sampling:
                    # weight each object by the inverse of the number of categories it belongs to
                    objects_per_category = self.dataset.groupby(by='category').object.nunique().to_dict()
                    uniqs = self.dataset.groupby(by='object').category.apply(lambda x: objects_per_category[x.iloc[0]])
                    weights = (1 / uniqs.values)
                    weights = weights / weights.sum()
                    sampled_objects = np.random.choice(uniqs.index, p=weights,
                                                       size=int(len(uniqs) * self.args.imgnet_subset))
                else:
                    uniqs = self.dataset["object"].unique()
                    sampled_objects = np.random.choice(uniqs, int(len(uniqs) * self.args.imgnet_subset))

                self.dataset = self.dataset.query('object in @sampled_objects').reset_index(drop=True)
                # ensure that all categories are present
                assert len(self.categories) == len(self.dataset["category"].unique()), \
                    "Lost some categories during sampling, consider increasing the sample size"
                print(f"Sampled {self.args.imgnet_subset * 100}% of the dataset."
                      f" Num. objects {len(uniqs):,} -> {len(sampled_objects):,}")

            # map the category to an index in [0, n_classes]
            if self.split == "train" and not self.eval_mode:
                print("Creating category mapping")
                MVImgNet.category_mapping = {cat: i for i, cat in enumerate(self.categories)}
            else:
                if not hasattr(self, "category_mapping"):
                    raise ValueError("Category mapping not found. Create train dataset first.")

            self.n_classes = len(MVImgNet.category_mapping)
            self.dataset["category_int"] = self.dataset["category"].apply(lambda x: np.int64(MVImgNet.category_mapping[x]))
            self.objects = self.dataset["object"].unique()
            object_mapping = {obj: i for i, obj in enumerate(self.objects)}
            self.dataset["object_int"] = self.dataset["object"].apply(lambda x: np.int64(object_mapping[x]))


            self.all_classes = [self.n_classes, self.num_contexts]
            self.dataset['original_frame'] = self.dataset["frame"]
            if self.args.sampling_mode == "interval" and self.split == "train":
                exp_identifier = "_" + os.environ.get("SLURM_JOB_ID", "")
                if not self.args.crop_number:
                    path_dataset = os.path.join(self.args.data_root,f"datasetT_{self.args.p_change_room}_{self.args.seed}{exp_identifier}.csv")
                else:
                    path_dataset = os.path.join(self.args.data_root,
                                                f"datasetT_{self.args.p_change_room}_{self.args.crop_number}_"
                                                f"{self.args.seed}{exp_identifier}.csv")

                    if self.fabric.global_rank == 0:
                        self.create_temporal_sequence()
                        self.dataset.to_csv(path_dataset)
                    self.fabric.barrier()
                    self.dataset = pd.read_csv(path_dataset)
                    self.dataset.index = self.dataset["original_index"]
                    self.prev_obj = self.dataset.loc[self.dataset["prev_obj"] > -1]["prev_obj"].values
                    self.next_obj = self.dataset.loc[self.dataset["next_obj"] > -1]["next_obj"].values

        if self.args.finetune_labels != -1 and self.split == "train" and not self.eval_mode:
            self.dataset = self.dataset.groupby("object").apply(
                lambda x: x.sample(self.args.finetune_labels)).reset_index(drop=True)

        print(f"Dataset length {len(self.dataset):,}")

    def crop_objects(self):
        self.dataset["crop_id"] = -1
        groups = self.dataset.groupby(["category", "object"])
        crop_id = -1
        last = 0

        crop_ids = np.zeros((len(self.dataset),), dtype=np.int64)
        frames = np.zeros((len(self.dataset),), dtype=np.int64)
        lengths = np.zeros((len(self.dataset),), dtype=np.int64)

        for _, gr in groups:
            total_val = 0
            val = 0
            for index, row in gr.iterrows():
                if row["frame"] == 0 or last >= val:
                    total_val += val
                    val = min(1 + int(np.random.poisson(self.args.crop_number)), row["length"] - total_val)
                    last = 0
                    crop_id += 1
                crop_ids[index] = crop_id
                frames[index] = last
                lengths[index] = val
                last += 1
        print(f"Number of crops {len(groups):,} -> {crop_id + 1:,}")

        self.dataset["crop_id"] = crop_ids
        self.dataset["frame"] = frames
        self.dataset["length"] = lengths

    def create_temporal_sequence(self):
        self.dataset["original_index"] = self.dataset.index
        # split the object into multiple crops
        if self.args.crop_number:
            self.crop_objects()

        self.dataset["next_obj"] = -1
        self.dataset["prev_obj"] = -1
        rooms_v = self.dataset["context"].unique()
        rooms = [r for r in range(len(rooms_v))]

        # contain the information of the first frame of every object
        rooms_to_obj_first = {}  # {room_index: [DataFrame, ...]}
        # contains all objects, that are connected to one room
        room_to_objects = {}  # {room_index: [(dataset_index, object_name), ...]}
        n_objects_in_room = []

        def sample_room() -> int:
            # sample new room weighted by the amount of objects in each room
            return random.choices(rooms, weights=n_objects_in_room, k=1)[0]

        def get_next_object(room):
            objorder = n_objects_in_room[room] - 1
            new_obj_name = room_to_objects[room][objorder]
            new_obj = rooms_to_obj_first[room].loc[new_obj_name]
            # reduce the count of the object in that room by one
            n_objects_in_room[room] -= 1
            return new_obj

        # Preprocess temporal segments into groups
        for i, r in enumerate(rooms_v):
            # select a groups of frames that belong to an object
            groups = self.dataset.loc[self.dataset["context"] == r].groupby(
                ["category", "object"] if not self.args.crop_number else "crop_id")
            # first frame of every object
            rooms_to_obj_first[i] = groups.first()
            # last frames of every object
            lasts = groups.last()
            # mark the index of the last frame for every object in the first frame
            rooms_to_obj_first[i]["original_last_index"] = lasts["original_index"]
            # shuffle the object that are connected to a room
            room_to_objects[i] = np.random.permutation(rooms_to_obj_first[i].index.values)
            # count the number of objects in the rooms
            n_objects_in_room.append(len(lasts))
        total_length = sum(n_objects_in_room)

        print(n_objects_in_room)

        # Flag images that are already processed
        images_keep = np.zeros((len(self.dataset),), dtype=bool)
        count_per_room = defaultdict(int)
        limit_object_in_room = [self.args.chunk_ratio_limit * x for x in n_objects_in_room]

        # sample starting room and first object
        new_room = sample_room()
        new_obj = get_next_object(new_room)
        first_obj = new_obj
        # mark all images that are related to the chosen image
        images_keep[new_obj["original_index"]:new_obj["original_last_index"] + 1] = 1

        # number of rooms that still have objects
        cpt_nonzero = sum(o != 0 for o in n_objects_in_room)
        cpt = 0

        # Create the temporal sequence
        # process only a part of all rooms
        while cpt < self.args.chunk_ratio_limit * total_length:
            actual_obj = new_obj
            actual_room = new_room

            # to fix the problem of missing rooms if p is 0, we limit the max object per room
            consider_room_count = count_per_room[new_room] > limit_object_in_room[new_room] if self.args.p_change_room == 0.0 else False
            # Change room if hit prob. or if same room is empty
            if random.random() < self.args.p_change_room or n_objects_in_room[actual_room] == 0 or consider_room_count:
                new_room = sample_room()
                if cpt_nonzero > 1:  # ensure that there is any room left
                    tmp = n_objects_in_room[actual_room]
                    n_objects_in_room[actual_room] = 0  # mask out current room
                    # try to find a non-empty room
                    while n_objects_in_room[new_room] == 0 or new_room == actual_room:
                        new_room = sample_room()
                    n_objects_in_room[actual_room] = tmp

            # get a new object from the new room
            new_obj = get_next_object(new_room)
            count_per_room[new_room] += 1

            images_keep[new_obj["original_index"]:new_obj["original_last_index"] + 1] = 1

            # assign the index of the new object to the next_obj attribute of the current object
            self.dataset.loc[actual_obj["original_last_index"], "next_obj"] = new_obj["original_index"]
            # assign the index of the current object as prev. object of the new object
            self.dataset.loc[new_obj["original_index"], "prev_obj"] = actual_obj["original_last_index"]

            cpt += 1
            cpt_nonzero = sum(o != 0 for o in n_objects_in_room)
            if cpt_nonzero == 1:
                raise Exception("Error")

        self.dataset.loc[new_obj["original_last_index"], "next_obj"] = -2
        self.dataset.loc[first_obj["original_index"], "prev_obj"] = -2
        self.dataset = self.dataset.loc[images_keep.tolist()]
        print("Crops selected number:", cpt, "Crop number:", total_length)

    def interval_sampling(self, idx, frame_index, size):
        time_shift = random.randint(-self.args.rotation_range, self.args.rotation_range)

        new_frame_index = frame_index + time_shift
        med_idx = idx
        new_idx = med_idx + time_shift

        # Now we will browse objects until consuming the whole time shift
        while new_frame_index < 0:
            # The very first image sample of the dataset, no previous objects thus we clamp
            if self.dataset.loc[med_idx - frame_index]["prev_obj"] == -2:
                new_idx = med_idx - frame_index
                break

            if not self.args.interval_uniform:
                med_idx = self.dataset.loc[med_idx - frame_index]["prev_obj"]
            else:
                change_room = random.random() < self.args.p_change_room
                med_idx = random.choice(self.prev_obj)
                while ((self.dataset.loc[med_idx, "context"] == self.dataset.loc[idx, "context"] and change_room) or
                       (self.dataset.loc[med_idx, "context"] != self.dataset.loc[idx, "context"] and not change_room)):
                    med_idx = random.choice(self.prev_obj)

            time_shift = new_frame_index
            frame_index = self.dataset.loc[med_idx, "frame"]

            size = self.dataset.loc[med_idx, "length"]

            new_frame_index = frame_index + time_shift
            new_idx = med_idx + time_shift

        while new_frame_index > size - 1:
            # The very last image samples of the dataset, no next objects thus we clamp
            if self.dataset.loc[med_idx - frame_index + size - 1]["next_obj"] == -2:
                new_idx = med_idx - frame_index + size - 1
                break
            if not self.args.interval_uniform:
                med_idx = self.dataset.loc[med_idx - frame_index + size - 1]["next_obj"]
            else:
                change_room = random.random() < self.args.p_change_room
                med_idx = random.choice(self.next_obj)
                while ((self.dataset.loc[med_idx, "context"] == self.dataset.loc[idx, "context"] and change_room) or
                       (self.dataset.loc[med_idx, "context"] != self.dataset.loc[idx, "context"] and not change_room)):
                    med_idx = random.choice(self.next_obj)

            time_shift = time_shift + frame_index - size + 1
            frame_index = self.dataset.loc[med_idx, "frame"]
            size = self.dataset.loc[med_idx, "length"]
            new_frame_index = frame_index + time_shift
            new_idx = med_idx + time_shift

        return self.dataset.loc[new_idx, "category"], self.dataset.loc[new_idx, "object"], new_idx

    def __len__(self):
        return len(self.dataset)

    def open_image(self, category, obj, index, path, idx, datapoint):
        if not self.args.hdf5:
            # path = f"{args.data_root}/{category}/{obj}/images/"+"%04d"%index}"
            # path = "%s/%s/%s/images/"%(self.args.data_root, category, obj)
            # return Image.open(os.path.join(self.args.data_root, "200/0000a8d9/images/001.jpg"))
            return Image.open(os.path.join(self.args.data_root, path))

        if self.args.remove_background:
            raw = self.hdf5_file.get(datapoint.partition_mask).get('image')[datapoint.partition_index]
            return Image.open(io.BytesIO(raw)).convert('RGB')

        if self.args.hdf5_mode == "partition":
            partition = self.dataset.loc[idx, "partition"]
            try:
                return Image.open(io.BytesIO(self.hdf5_file.get(partition).get(category).get(obj)[index]))
            except:
                print(str(category), str(obj), index, path, idx)
                print(self.hdf5_file.get(partition))
                print(self.hdf5_file.get(partition).get(category))
                print(self.hdf5_file.get(partition).get(category).get(obj))
                print(self.hdf5_file.get(partition).get(category).get(obj)[index])
        # print(self.hdf5_file)
        try:
            c = self.hdf5_file.get(category)
            o = c.get(obj)
            return Image.open(io.BytesIO(o[index]))
        except:
            print(category, obj, index, path, self.hdf5_file)
            raise Exception("not ok")

    @classmethod
    def get_args(cls, parser):
        parser.add_argument("--hdf5_mode", type=str, default="partition")
        parser.add_argument("--action_clamp", type=str2bool, default=True)
        parser.add_argument("--imgnet_subset", type=float, default=0.3)
        parser.add_argument("--imgnet_subset_balanced_sampling", type=bool, default=True, help="Weight each object by the inverse of the number of categories it belongs to")
        parser.add_argument("--p_change_room", type=float, default=0.1)
        parser.add_argument("--interval_uniform", type=str2bool, default=False)
        parser.add_argument("--crop_number", type=int, default=7)
        parser.add_argument("--chunk_ratio_limit", type=float, default=0.8)
        parser.add_argument("--context_json", type=str, default='room_assignment.json', help='Path to context json file')
        parser.add_argument("--remove_background", type=str2bool, default=False, help='Use images without background')
        return parser

    def __getitem__(self, idx):

        if self.args.sampling_mode == "interval" and self.split == "train" and not self.eval_mode:
            sample = self.dataset.iloc[idx]
            idx = sample["original_index"]

        sample = self.dataset.loc[idx]
        category, obj, frame_index, size = sample["category"], sample["object"], sample["frame"], sample["length"]
        room_label = sample["context"]
        image = self.open_image(str(category), str(obj), sample["original_frame"], sample["path"], idx, sample)
        label = np.int64(self.category_mapping[category])

        # print(self.transform)

        if self.transform:
            image_t = self.transform(image)

        if not self.contrastive:
            return (
            image_t, self.transform(image) if self.split == "train" else image_t, torch.zeros((8,))), label, room_label, sample["object_int"]

        # if self.args.sampling_mode == "interval":
        #     category, obj, new_idx = self.interval_sampling(idx, frame_index, size)
        if self.args.sampling_mode == "interval":
            category, obj, new_idx = self.interval_sampling(idx, frame_index, size)
        elif self.args.sampling_mode == "uniform":
            new_frame_index = random.randint(0, size - 1)
            new_idx = idx + new_frame_index - frame_index
        elif self.args.sampling_mode == "randomwalk+":
            new_frame_index = frame_index + 1 if frame_index < size - 1 else frame_index - 1
            new_idx = idx + new_frame_index - frame_index
        elif self.args.sampling_mode == "last":
            # new_frame_index = frame_index + 1 if frame_index < size - 1 else frame_index - 1
            new_frame_index = sample["length"]-1
            new_idx = idx + new_frame_index - frame_index
        else:
            new_frame_index = max(0, min(size - 1, (frame_index - 1 if random.random() < 0.5 else frame_index + 1)))
            new_idx = idx + new_frame_index - frame_index

        new_sample = self.dataset.loc[new_idx]
        if self.args.sampling_mode != "interval":
            assert str(obj) == str(new_sample[
                                       "object"]), f"{str(obj)}, {str(new_sample['object'])}, {idx}, {new_idx}, {frame_index}, {new_frame_index}, {size}"
        image_pair = self.open_image(str(category), str(obj), self.dataset.loc[new_idx, "original_frame"], self.dataset.loc[new_idx, "path"], new_idx, new_sample)
        if self.transform:
            image_pair = self.transform(image_pair)
        return (image_t, image_pair, None), label, room_label, sample["object_int"]
