import os
from compress_pickle import load, dump
import random
from pathlib import Path

import albumentations as A
import cv2 as cv
import einops
import numpy as np
import numpy.ma as ma
import pandas as pd
import richdem as rd
import rioxarray as rio
import torch
import torchvision
from torchio.transforms import RescaleIntensity
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
import pyjson5 as json

import utilities.utilities as utilities
from utilities import augmentations


def get_grids(pickle_path):
    if not os.path.isfile(pickle_path):
        print("Pickle file not found! ", pickle_path)
        exit(2)
    with open(pickle_path, "rb") as file:
        grid_dict = load(file)
    return grid_dict


class Dataset(torch.utils.data.Dataset):
    def __init__(self, mode="train", configs=None):
        self.train_acts = configs["train_acts"]
        self.val_acts = configs["val_acts"]
        self.test_acts = configs["test_acts"]
        self.mode = mode
        self.configs = configs
        self.root_path = os.path.join(self.configs["root_path"], "data")
        
        if (
            self.configs["task"] == "self-supervised"
            or self.configs["data_augmentations"]
        ):
            self.augmentations = augmentations.get_augmentations(self.configs)
        else:
            self.augmentations = None

        self.non_valids = []

        # Load precomputed min-max stats for each SAR image or calculate them anew
        self.min_max_random_events = self.update_min_max_stats()
        self.clz_stats = {1: 0, 2: 0, 3: 0}
        self.act_stats = {}
        if self.mode == "train":
            self.valid_acts = self.train_acts
            self.pickle_path = configs["train_pickle"]
        elif self.mode == "val":
            self.valid_acts = self.val_acts
            self.pickle_path = configs["test_pickle"]
        else:
            self.valid_acts = self.test_acts
            self.pickle_path = configs["test_pickle"]

        self.negative_grids = None
        total_grids = {}
        self.positive_records = []
        self.negative_records = []

        if not configs["oversampling"] or self.mode != "train":
            self.grids = get_grids(pickle_path=self.pickle_path)
            total_grids = self.grids
        else:
            self.grids = get_grids(pickle_path=self.pickle_path)
            self.negative_grids = get_grids(pickle_path=configs["negative_pickle"])
            total_grids.update(self.grids)
            total_grids.update(self.negative_grids)
            print("=" * 20)
            print("Enabling oversampling")
            print("Length of positive grids: ", len(self.grids))
            print("Length of negative grids: ", len(self.negative_grids))
            print("Total grids: ", len(total_grids))
            print("=" * 20)

        all_activations = []
        all_activations.extend(self.train_acts)
        all_activations.extend(self.val_acts)
        all_activations.extend(self.test_acts)
        self.records = []
        for key in total_grids:
            record = {}
            record["id"] = key
            record["path"] = total_grids[key]["path"]

            record["info"] = total_grids[key]["info"]
            record["type"] = None
            record["clz"] = total_grids[key]["clz"]
            activation = total_grids[key]["info"]["actid"]
            aoi = total_grids[key]["info"]["aoiid"]
            if configs["track"] == "Climatic":
                act_aoi = str(activation) + "_" + f"{aoi:02}"
            else:
                act_aoi = activation

            record["activation"] = activation
            if act_aoi in self.valid_acts:
                self.clz_stats[record["clz"]] += 1
                if act_aoi in self.act_stats:
                    self.act_stats[act_aoi] += 1
                else:
                    self.act_stats[act_aoi] = 1
                if self.configs["task"] == "diffusion-unsup":
                    # We will create a separate record per observation (pre1, pre2, flood) in order to
                    # ensure that the model will see every image during an epoch
                    # This will also allow us to compute appropriate weights for the loss functions
                    for t in ["pre1", "pre2", "flood"]:
                        tmp = record.copy()
                        tmp["type"] = t
                        self.records.append(tmp)
                        if key in self.grids:
                            self.positive_records.append(tmp)
                        else:
                            self.negative_records.append(tmp)
                else:
                    self.records.append(record)
                    if key in self.grids:
                        self.positive_records.append(record)
                    else:
                        self.negative_records.append(record)

            if act_aoi not in all_activations and act_aoi not in self.non_valids:
                print("Activation: ", activation, " not in Activations")
                self.non_valids.append(act_aoi)

        print("Samples per Climatic zone for mode: ", self.mode)
        print(self.clz_stats)
        print("Samples per Activation for mode: ", self.mode)
        print(self.act_stats)
        self.num_examples = len(self.records)
        self.activations = set([record["activation"] for record in self.records])

    def __len__(self):
        return self.num_examples

    def concat(self, image1, image2):
        image1_exp = np.expand_dims(image1, 0)  # vv
        image2_exp = np.expand_dims(image2, 0)  # vh

        if set(self.configs["channels"]) == set(["vv", "vh", "vh/vv"]):
            eps = 1e-7
            image = np.vstack(
                (image1_exp, image2_exp, image2_exp / (image1_exp + eps))
            )  # vv, vh, vh/vv
        elif set(self.configs["channels"]) == set(["vv", "vh"]):
            image = np.vstack((image1_exp, image2_exp))  # vv, vh
        elif self.configs["channels"] == ["vh"]:
            image = image2_exp  # vh

        image = torch.from_numpy(image).float()

        if self.configs["clamp_input"] is not None:
            image = torch.clamp(image, min=0.0, max=self.configs["clamp_input"])
            image = torch.nan_to_num(image, self.configs["clamp_input"])
        else:
            image = torch.nan_to_num(image, 200)
        return image

    def create_views(self, event, mask=None):
        if mask is None:
            transform = self.augmentations(image=event.permute(1, 2, 0).numpy())
            view_1 = transform["image"]
            transform_2 = self.augmentations(image=event.permute(1, 2, 0).numpy())
            view_2 = transform_2["image"]
            return torch.from_numpy(view_1).permute(2, 0, 1), torch.from_numpy(
                view_2
            ).permute(2, 0, 1)

        else:
            transform = self.augmentations(
                image=event.permute(1, 2, 0).numpy(), masks=mask
            )
            view_1 = transform["image"]
            mask1 = transform["masks"]
            return torch.from_numpy(view_1).permute(2, 0, 1), [
                torch.from_numpy(mask1[0]),
                torch.from_numpy(mask1[1]),
            ]

    def scale_img(self, img, valid_mask, img_name, activation):
        if self.configs["scale_input"] == "normalize":
            # Read dataset mean and std from configs
            means = self.configs["data_mean"]
            stds = self.configs["data_std"]

            return means, stds, torchvision.transforms.Normalize(means, stds)(img)
        elif self.configs["scale_input"] == "min-max":
            if len(self.configs["channels"]) > 1:
                mins = {}
                maxs = {}

                if "vv" in self.configs["channels"]:
                    mins["vv"] = self.min_max_random_events[activation][
                        f"{img_name}_vv"
                    ][0]
                    if self.configs["clamp_input"] is not None:
                        maxs["vv"] = self.configs["clamp_input"]
                    else:
                        maxs["vv"] = self.min_max_random_events[activation][
                            f"{img_name}_vv"
                        ][1]
                if "vh" in self.configs["channels"]:
                    mins["vh"] = self.min_max_random_events[activation][
                        f"{img_name}_vh"
                    ][0]
                    if self.configs["clamp_input"] is not None:
                        maxs["vh"] = self.configs["clamp_input"]
                    else:
                        maxs["vh"] = self.min_max_random_events[activation][
                            f"{img_name}_vh"
                        ][1]
                if "vh/vv" in self.configs["channels"]:
                    mins["vh/vv"] = (
                        self.min_max_random_events[activation][f"{img_name}_vh"][0]
                        / self.min_max_random_events[activation][f"{img_name}_vv"][0]
                    )
                    if self.configs["clamp_input"] is not None:
                        maxs["vh/vv"] = 1.0
                    else:
                        maxs["vh/vv"] = (
                            self.min_max_random_events[activation][f"{img_name}_vh"][1]
                            / self.min_max_random_events[activation][f"{img_name}_vv"][
                                1
                            ]
                        )

                new_ch = []
                for ch_i, ch in enumerate(self.configs["channels"]):
                    new_ch.append(
                        ((img[ch_i, :, :] - mins[ch]) / (maxs[ch] - mins[ch]))[
                            None, :, :
                        ]
                    )

                return (
                    list(mins.values()),
                    list(maxs.values()),
                    torch.cat(new_ch, dim=0),
                )
            else:
                if self.configs["channels"] == ["vh/vv"]:
                    mins = (
                        self.min_max_random_events[activation][f"{img_name}_vh"][0]
                        / self.min_max_random_events[activation][f"{img_name}_vv"][0]
                    )
                else:
                    mins = self.min_max_random_events[activation][
                        f'{img_name}_{self.configs["channels"][0]}'
                    ][0]

                if self.configs["clamp_input"] is not None:
                    maxs = self.configs["clamp_input"]
                elif self.configs["channels"] == ["vh/vv"]:
                    maxs = (
                        self.min_max_random_events[activation][f"{img_name}_vh"][1]
                        / self.min_max_random_events[activation][f"{img_name}_vv"][1]
                    )
                else:
                    maxs = self.min_max_random_events[activation][
                        f'{img_name}_{self.configs["channels"][0]}'
                    ][1]
                return mins, maxs, (img - mins) / (maxs - mins)
        elif isinstance(self.configs["scale_input"], list):
            # The required min and max values are given
            new_min, new_max = [torch.tensor(i) for i in self.configs["scale_input"]]

            if len(self.configs["channels"]) > 1:
                mins = {}
                maxs = {}

                if "vv" in self.configs["channels"]:
                    mins["vv"] = self.min_max_random_events[activation][
                        f"{img_name}_vv"
                    ][0]
                    if self.configs["clamp_input"] is not None:
                        maxs["vv"] = self.configs["clamp_input"]
                    else:
                        maxs["vv"] = self.min_max_random_events[activation][
                            f"{img_name}_vv"
                        ][1]
                if "vh" in self.configs["channels"]:
                    mins["vh"] = self.min_max_random_events[activation][
                        f"{img_name}_vh"
                    ][0]
                    if self.configs["clamp_input"] is not None:
                        maxs["vh"] = self.configs["clamp_input"]
                    else:
                        maxs["vh"] = self.min_max_random_events[activation][
                            f"{img_name}_vh"
                        ][1]
                if "vh/vv" in self.configs["channels"]:
                    mins["vh/vv"] = (
                        self.min_max_random_events[activation][f"{img_name}_vh"][0]
                        / self.min_max_random_events[activation][f"{img_name}_vv"][0]
                    )
                    if self.configs["clamp_input"] is not None:
                        maxs["vh/vv"] = 1.0
                    else:
                        maxs["vh/vv"] = (
                            self.min_max_random_events[activation][f"{img_name}_vh"][1]
                            / self.min_max_random_events[activation][f"{img_name}_vv"][
                                1
                            ]
                        )

                new_ch = []
                for ch_i, ch in enumerate(self.configs["channels"]):
                    new_ch.append(
                        ((img[ch_i, :, :] - mins[ch]) / (maxs[ch] - mins[ch]))[
                            None, :, :
                        ]
                    )

                new_img = torch.cat(new_ch, dim=0)
                new_img = torch.mul(new_img, (new_max - new_min)) + new_min

                return list(mins.values()), list(maxs.values()), new_img
            else:
                if self.configs["channels"] == ["vh/vv"]:
                    mins = (
                        self.min_max_random_events[activation][f"{img_name}_vh"][0]
                        / self.min_max_random_events[activation][f"{img_name}_vv"][0]
                    )
                else:
                    mins = self.min_max_random_events[activation][
                        f'{img_name}_{self.configs["channels"][0]}'
                    ][0]

                if self.configs["clamp_input"] is not None:
                    maxs = self.configs["clamp_input"]
                elif self.configs["channels"] == ["vh/vv"]:
                    maxs = (
                        self.min_max_random_events[activation][f"{img_name}_vh"][1]
                        / self.min_max_random_events[activation][f"{img_name}_vv"][1]
                    )
                else:
                    maxs = self.min_max_random_events[activation][
                        f'{img_name}_{self.configs["channels"][0]}'
                    ][1]
                return (
                    torch.mul((img - mins) / (maxs - mins), (new_max - new_min))
                    + new_min
                )
        elif self.configs["scale_input"] == "custom":
            # 1) Bring all values to > 0 (to avoid NaNs after the logarithm below)
            eps = 1e-7
            offset = torch.tensor(
                [torch.masked_select(x, valid_mask).min().item() for x in img]
            )

            mins = {}
            maxs = {}
            if len(self.configs["channels"]) > 1:
                if "vv" in self.configs["channels"]:
                    mins["vv"] = self.min_max_random_events[activation][
                        f"{img_name}_vv"
                    ][0]
                    if self.configs["clamp_input"] is not None:
                        maxs["vv"] = self.configs["clamp_input"]
                    else:
                        maxs["vv"] = self.min_max_random_events[activation][
                            f"{img_name}_vv"
                        ][1]
                if "vh" in self.configs["channels"]:
                    mins["vh"] = self.min_max_random_events[activation][
                        f"{img_name}_vh"
                    ][0]
                    if self.configs["clamp_input"] is not None:
                        maxs["vh"] = self.configs["clamp_input"]
                    else:
                        maxs["vh"] = self.min_max_random_events[activation][
                            f"{img_name}_vh"
                        ][1]
                if "vh/vv" in self.configs["channels"]:
                    mins["vh/vv"] = (
                        self.min_max_random_events[activation][f"{img_name}_vh"][0]
                        / self.min_max_random_events[activation][f"{img_name}_vv"][0]
                    )
                    if self.configs["clamp_input"] is not None:
                        maxs["vh/vv"] = 1.0
                    else:
                        maxs["vh/vv"] = (
                            self.min_max_random_events[activation][f"{img_name}_vh"][1]
                            / self.min_max_random_events[activation][f"{img_name}_vv"][
                                1
                            ]
                        )
            else:
                if self.configs["channels"] == ["vh/vv"]:
                    mins["vv/vh"] = (
                        self.min_max_random_events[activation][f"{img_name}_vh"][0]
                        / self.min_max_random_events[activation][f"{img_name}_vv"][0]
                    )
                else:
                    mins[self.configs["channels"][0]] = self.min_max_random_events[
                        activation
                    ][f'{img_name}_{self.configs["channels"][0]}'][0]

                if self.configs["clamp_input"] is not None:
                    maxs[self.configs["channels"][0]] = self.configs["clamp_input"]
                elif self.configs["channels"] == ["vh/vv"]:
                    maxs["vh/vv"] = (
                        self.min_max_random_events[activation][f"{img_name}_vh"][1]
                        / self.min_max_random_events[activation][f"{img_name}_vv"][1]
                    )
                else:
                    maxs[self.configs["channels"][0]] = self.min_max_random_events[
                        activation
                    ][f'{img_name}_{self.configs["channels"][0]}'][1]

            if len(self.configs["channels"]) > 1:
                new_ch = []
                for ch_i, ch in enumerate(self.configs["channels"]):
                    new_ch.append(
                        ((img[ch_i, :, :] - mins[ch]) / (maxs[ch] - mins[ch]))[
                            None, :, :
                        ]
                    )
                    if offset[0] <= 0:
                        new_ch.append(
                            (img[ch_i, :, :] + (-offset[ch_i]) + eps)[None, :, :]
                        )
                    else:
                        new_ch.append(img[ch_i, :, :][None, :, :])

                img_scaled = torch.cat(new_ch, dim=0)
            else:
                if offset <= 0:
                    img_scaled = img + (-offset) + eps
                else:
                    img_scaled = img.clone()

            # 2) Log scaling
            img_scaled = torch.log(img_scaled)

            # 3) Min-max scaling to [0, 1]
            if len(self.configs["channels"]) > 1:
                new_ch = []
                for ch_i, ch in enumerate(self.configs["channels"]):
                    new_ch.append(
                        ((img_scaled[ch_i, :, :] - mins[ch]) / (maxs[ch] - mins[ch]))[
                            None, :, :
                        ]
                    )

                img_scaled = torch.cat(new_ch, dim=0)
            else:
                img_scaled = (img_scaled - mins) / (maxs - mins)

            # 4) Contrast stretching
            intens_scaling = RescaleIntensity(out_min_max=(0, 1), percentiles=(2, 98))

            if len(self.configs["channels"]) > 1:
                new_ch = []
                for ch in range(len(self.configs["channels"])):
                    new_ch.append(
                        intens_scaling(
                            img_scaled[ch, :, :][None, None, :, :]
                        ).squeeze()[None, :, :]
                    )

                return (
                    list(mins.values()),
                    list(maxs.values()),
                    torch.cat(new_ch, dim=0),
                )
            else:
                return (
                    list(mins.values()),
                    list(maxs.values()),
                    intens_scaling(img_scaled[None, :, :]).squeeze()[None, :, :],
                )

    def update_min_max_stats(self):
        if Path("stats.pkl").exists():
            print(f"({self.mode}) Using precalculated stats for dataset...")
            return load(open("stats.pkl", "rb"))

        print(f"({self.mode}) Calculating stats for dataset...")

        records = []
        for mode in ["train", "val", "test"]:
            valid = self.configs[f"{mode}_acts"]

            if mode == "test" or mode == "val":
                pickle_path = self.configs["test_pickle"]
            else:
                pickle_path = self.configs["train_pickle"]

            grids = get_grids(pickle_path=pickle_path)

            for key in grids.keys():
                record = {}
                record["path"] = grids[key]["path"]

                activation = grids[key]["info"]["actid"]
                aoi = grids[key]["info"]["aoiid"]
                if self.configs["track"] == "Climatic":
                    act_aoi = str(activation) + "_" + f"{aoi:02}"
                else:
                    act_aoi = activation
                record["activation"] = activation

                if act_aoi in self.non_valids:
                    continue

                if act_aoi in valid:
                    records.append(record)

        min_max_random_events = {}
        for record in records:
            filespath = Path(record["path"])
            filespath = self.root_path / filespath

            for file in filespath.glob("*"):
                if "xml" not in file.name:
                    if file.stem.startswith("MK0_MNA"):
                        # Get mask of valid pixels
                        valid_mask = cv.imread(str(file), cv.IMREAD_ANYDEPTH)
                    elif file.stem.startswith("MS1_IVV"):
                        # Get master ivv channel
                        flood_vv = cv.imread(str(file), cv.IMREAD_ANYDEPTH)
                    elif file.stem.startswith("MS1_IVH"):
                        # Get master ivh channel
                        flood_vh = cv.imread(str(file), cv.IMREAD_ANYDEPTH)
                    elif file.stem.startswith("SL1_IVV"):
                        # Get slave1 vv channel
                        sec1_vv = cv.imread(str(file), cv.IMREAD_ANYDEPTH)
                    elif file.stem.startswith("SL1_IVH"):
                        # Get sl1 vh channel
                        sec1_vh = cv.imread(str(file), cv.IMREAD_ANYDEPTH)
                    elif file.stem.startswith("SL2_IVV"):
                        # Get sl2 vv channel
                        sec2_vv = cv.imread(str(file), cv.IMREAD_ANYDEPTH)
                    elif file.stem.startswith("SL2_IVH"):
                        # Get sl2 vh channel
                        sec2_vh = cv.imread(str(file), cv.IMREAD_ANYDEPTH)
            
            invalid_mask = valid_mask != 1

            # Mask images with valid pixels
            flood_vv = ma.masked_array(flood_vv, invalid_mask)
            flood_vh = ma.masked_array(flood_vh, invalid_mask)
            sec1_vv = ma.masked_array(sec1_vv, invalid_mask)
            sec1_vh = ma.masked_array(sec1_vh, invalid_mask)
            sec2_vv = ma.masked_array(sec2_vv, invalid_mask)
            sec2_vh = ma.masked_array(sec2_vh, invalid_mask)

            if record["activation"] not in min_max_random_events.keys():
                min_max_random_events[record["activation"]] = {
                    "pre1_vv": (ma.min(sec1_vv), ma.max(sec1_vv)),
                    "pre1_vh": (ma.min(sec1_vh), ma.max(sec1_vh)),
                    "pre2_vv": (ma.min(sec2_vv), ma.max(sec2_vv)),
                    "pre2_vh": (ma.min(sec2_vh), ma.max(sec2_vh)),
                    "flood_vv": (ma.min(flood_vv), ma.max(flood_vv)),
                    "flood_vh": (ma.min(flood_vh), ma.max(flood_vh)),
                }
            else:
                min_max_random_events[record["activation"]] = {
                    "pre1_vv": (
                        min(
                            ma.min(sec1_vv),
                            min_max_random_events[record["activation"]]["pre1_vv"][0],
                        ),
                        max(
                            ma.max(sec1_vv),
                            min_max_random_events[record["activation"]]["pre1_vv"][1],
                        ),
                    ),
                    "pre1_vh": (
                        min(
                            ma.min(sec1_vh),
                            min_max_random_events[record["activation"]]["pre1_vh"][0],
                        ),
                        max(
                            ma.max(sec1_vh),
                            min_max_random_events[record["activation"]]["pre1_vh"][1],
                        ),
                    ),
                    "pre2_vv": (
                        min(
                            ma.min(sec2_vv),
                            min_max_random_events[record["activation"]]["pre2_vv"][0],
                        ),
                        max(
                            ma.max(sec2_vv),
                            min_max_random_events[record["activation"]]["pre2_vv"][1],
                        ),
                    ),
                    "pre2_vh": (
                        min(
                            ma.min(sec2_vh),
                            min_max_random_events[record["activation"]]["pre2_vh"][0],
                        ),
                        max(
                            ma.max(sec2_vh),
                            min_max_random_events[record["activation"]]["pre2_vh"][1],
                        ),
                    ),
                    "flood_vv": (
                        min(
                            ma.min(flood_vv),
                            min_max_random_events[record["activation"]]["flood_vv"][0],
                        ),
                        max(
                            ma.max(flood_vv),
                            min_max_random_events[record["activation"]]["flood_vv"][1],
                        ),
                    ),
                    "flood_vh": (
                        min(
                            ma.min(flood_vh),
                            min_max_random_events[record["activation"]]["flood_vh"][0],
                        ),
                        max(
                            ma.max(flood_vh),
                            min_max_random_events[record["activation"]]["flood_vh"][1],
                        ),
                    ),
                }

        print(f"({self.mode}) New stats:")
        print(min_max_random_events)

        dump(min_max_random_events, open("stats.pkl", "wb"))

        return min_max_random_events

    def __getitem__(self, index):
        if self.configs["oversampling"] and self.mode == "train":
            choice = random.randint(0, 1)
            if choice == 0:
                index = random.randint(0, len(self.positive_records) - 1)
                sample = self.positive_records[index]
            elif choice == 1:
                index = random.randint(0, len(self.negative_records) - 1)
                sample = self.negative_records[index]
        else:
            sample = self.records[index]

        path = sample["path"]
        path = os.path.join(self.root_path, path)
        files = os.listdir(path)
        clz = sample["clz"]
        activation = sample["activation"]
        mask = None

        for file in files:
            current_path = str(os.path.join(path, file))
            if "xml" not in file:
                if file.startswith("MK0_MLU") and (sample["type"] is None):
                    # Get mask of flooded/perm water pixels
                    mask = cv.imread(current_path, cv.IMREAD_ANYDEPTH)
                elif file.startswith("MK0_MNA") and (sample["type"] is None):
                    # Get mask of valid pixels
                    valid_mask = cv.imread(current_path, cv.IMREAD_ANYDEPTH)
                elif file.startswith("MS1_IVV") and (
                    sample["type"] not in ["pre1", "pre2"]
                ):
                    # Get master ivv channel
                    flood_vv = cv.imread(current_path, cv.IMREAD_ANYDEPTH)
                    if self.configs["uint8"]:
                        flood_vv /= flood_vv.max()
                        flood_vv *= 255
                        flood_vv = flood_vv.astype(np.uint8)
                    if flood_vv is None:
                        print(current_path)

                elif file.startswith("MS1_IVH") and (
                    sample["type"] not in ["pre1", "pre2"]
                ):
                    # Get master ivh channel
                    flood_vh = cv.imread(current_path, cv.IMREAD_ANYDEPTH)
                    if self.configs["uint8"]:
                        flood_vh /= flood_vh.max()
                        flood_vh *= 255
                        flood_vh = flood_vh.astype(np.uint8)

                elif file.startswith("SL1_IVV") and (
                    sample["type"] not in ["flood", "pre2"]
                ):
                    # Get slave1 vv channel
                    sec1_vv = cv.imread(current_path, cv.IMREAD_ANYDEPTH)
                    if self.configs["uint8"]:
                        sec1_vv /= sec1_vv.max()
                        sec1_vv *= 255
                        sec1_vv = sec1_vv.astype(np.uint8)

                elif file.startswith("SL1_IVH") and (
                    sample["type"] not in ["flood", "pre2"]
                ):
                    # Get sl1 vh channel
                    sec1_vh = cv.imread(current_path, cv.IMREAD_ANYDEPTH)
                    if self.configs["uint8"]:
                        sec1_vh /= sec1_vh.max()
                        sec1_vh *= 255
                        sec1_vh = sec1_vh.astype(np.uint8)

                elif file.startswith("SL2_IVV") and (
                    sample["type"] not in ["flood", "pre1"]
                ):
                    # Get sl2 vv channel
                    sec2_vv = cv.imread(current_path, cv.IMREAD_ANYDEPTH)
                    if self.configs["uint8"]:
                        sec2_vv /= sec2_vv.max()
                        sec2_vv *= 255
                        sec2_vv = sec2_vv.astype(np.uint8)

                elif file.startswith("SL2_IVH") and (
                    sample["type"] not in ["flood", "pre1"]
                ):
                    # Get sl2 vh channel
                    sec2_vh = cv.imread(current_path, cv.IMREAD_ANYDEPTH)
                    if self.configs["uint8"]:
                        sec2_vh /= sec2_vh.max()
                        sec2_vh *= 255
                        sec2_vh = sec2_vh.astype(np.uint8)

                elif file.startswith("MK0_DEM"):
                    # Get DEM
                    dem = rio.open_rasterio(current_path)
                    nans = dem.isnull()
                    if nans.any():
                        dem = dem.rio.interpolate_na()
                        nans = dem.isnull()

                    nodata = dem.rio.nodata
                    dem = dem.to_numpy()
                    if not self.configs["dem"] and self.configs["slope"]:
                        print(
                            "To return the slope the DEM option must be enabled. Validate the config file!"
                        )
                        exit(2)

                    # Get slope before normalization
                    if self.configs["slope"]:
                        rd_dem = rd.rdarray(dem.squeeze(), no_data=nodata)
                        slope = rd.TerrainAttribute(rd_dem, attrib="slope_riserun")
                        slope = np.asarray(slope.data)
                        slope = einops.rearrange(slope, "h w -> 1 h w")
                        if self.configs["scale_input"] is not None:
                            # Only support standarization for DEMs
                            normalization = transforms.Normalize(
                                mean=self.configs["slope_mean"],
                                std=self.configs["slope_std"],
                            )
                            dem = normalization(torch.from_numpy(slope))
                        else:
                            dem = slope
                    else:
                        if self.configs["scale_input"] is not None:
                            normalization = transforms.Normalize(
                                mean=self.configs["dem_mean"],
                                std=self.configs["dem_std"],
                            )
                            dem = normalization(torch.from_numpy(dem))

        # Concat channels
        if sample["type"] not in ("pre1", "pre2"):
            flood = self.concat(flood_vv, flood_vh)
        if sample["type"] not in ("flood", "pre2"):
            pre_event_1 = self.concat(sec1_vv, sec1_vh)
        if sample["type"] not in ("flood", "pre1"):
            pre_event_2 = self.concat(sec2_vv, sec2_vh)

        if sample["type"] is None:
            if mask is None:
                mask = np.zeros((224, 224))

        mask = torch.from_numpy(mask).long()

        # Return record given training options
        if sample["type"] == "pre1":
            return pre_event_1
        if sample["type"] == "pre2":
            return pre_event_2
        if sample["type"] == "flood":
            return flood

        if self.augmentations is not None and self.mode == "train":
            events = torch.cat((pre_event_1, pre_event_2, flood), dim=0)
            events_A, masks_A = self.create_views(
                events, mask=[mask.numpy(), valid_mask]
            )
            if torch.sum(masks_A[1]) > 0:
                # Certain augmentation pipelines may return no valid pixels, so we discard them
                pre_event_1 = events_A[: pre_event_1.shape[0], :, :]
                pre_event_2 = events_A[
                    pre_event_1.shape[0] : 2 * pre_event_1.shape[0], :, :
                ]
                flood = events_A[2 * pre_event_1.shape[0] :, :, :]
                mask = masks_A[0]
                valid_mask = masks_A[1]
        else:
            valid_mask = torch.from_numpy(valid_mask)

        mask = mask.long()

        # Scale images if necessary
        if self.configs["scale_input"] is not None:
            valid_mask = valid_mask == 1
            flood_scale_var_1, flood_scale_var_2, flood = self.scale_img(
                flood, valid_mask, "flood", activation
            )
            pre1_scale_var_1, pre1_scale_var_2, pre_event_1 = self.scale_img(
                pre_event_1, valid_mask, "pre1", activation
            )
            pre2_scale_var_1, pre2_scale_var_2, pre_event_2 = self.scale_img(
                pre_event_2, valid_mask, "pre2", activation
            )

        if not self.configs["dem"]:
            if self.configs["scale_input"] is not None:
                return (
                    flood_scale_var_1,
                    flood_scale_var_2,
                    flood,
                    mask,
                    pre1_scale_var_1,
                    pre1_scale_var_2,
                    pre_event_1,
                    pre2_scale_var_1,
                    pre2_scale_var_2,
                    pre_event_2,
                    clz,
                    activation,
                )
            else:
                return flood, mask, pre_event_1, pre_event_2, clz, activation
        else:
            if self.configs["scale_input"] is not None:
                return (
                    flood_scale_var_1,
                    flood_scale_var_2,
                    flood,
                    mask,
                    pre1_scale_var_1,
                    pre1_scale_var_2,
                    pre_event_1,
                    pre2_scale_var_1,
                    pre2_scale_var_2,
                    pre_event_2,
                    dem,
                    clz,
                    activation,
                )
            else:
                return flood, mask, pre_event_1, pre_event_2, dem, clz, activation


# Dataset class for SSL MAE training
class SSLDataset(torch.utils.data.Dataset):
    def __init__(self, configs=None):
        resized_crop = A.augmentations.RandomResizedCrop(
            height=224, width=224, p=1.0, scale=(0.2, 1.0), interpolation=3
        )
        flip = A.augmentations.HorizontalFlip(p=0.5)
        self.augmentations = A.Compose([resized_crop, flip])
        self.root_path = os.path.join(configs["root_path"], "data")
        self.configs = configs
        self.samples = []
        events = os.listdir(self.root_path)
        if not os.path.isfile("ssl_samples.pkl"):
            for event in tqdm(events):
                folder_dir = os.path.join(self.root_path, event)
                folders = os.listdir(folder_dir)
                for folder in folders:
                    if ".gpkg" in folder:
                        continue
                    subfolder_dir = os.path.join(folder_dir, folder)
                    subfolders = os.listdir(subfolder_dir)
                    for subfolder in subfolders:
                        hashes_dir = os.path.join(subfolder_dir, subfolder)
                        hashes = os.listdir(hashes_dir)
                        for hash_folder in hashes:
                            hash_folder_dir = os.path.join(hashes_dir, hash_folder)
                            if os.path.isfile(hash_folder_dir):
                                self.samples.append(
                                    os.path.join(subfolder_dir, subfolder)
                                )
                            else:
                                self.samples.append(hash_folder_dir)
            with open("ssl_samples.pkl", "wb") as file:
                dump(self.samples, file)
        else:
            with open("ssl_samples.pkl", "rb") as file:
                self.samples = load(file)
        random.Random(999).shuffle(self.samples)
        self.num_examples = len(self.samples)

    def __len__(self):
        return self.num_examples

    def concat(self, image1, image2):
        image1_exp = np.expand_dims(image1, 0)  # vv
        image2_exp = np.expand_dims(image2, 0)  # vh

        if set(self.configs["channels"]) == set(["vv", "vh", "vh/vv"]):
            eps = 1e-7
            image = np.vstack(
                (image1_exp, image2_exp, image2_exp / (image1_exp + eps))
            )  # vv, vh, vh/vv
        elif set(self.configs["channels"]) == set(["vv", "vh"]):
            image = np.vstack((image1_exp, image2_exp))  # vv, vh
        elif self.configs["channels"] == ["vh"]:
            image = image2_exp  # vh

        image = torch.from_numpy(image).float()

        if self.configs["clamp_input"] is not None:
            image = torch.clamp(image, min=0.0, max=self.configs["clamp_input"])
            image = torch.nan_to_num(image, self.configs["clamp_input"])
        else:
            image = torch.nan_to_num(image, 200)
        return image

    def __getitem__(self, index):
        path = self.samples[index]

        files = os.listdir(path)
        for file in files:
            current_path = os.path.join(path, file)
            if "xml" not in file:
                if file.startswith("MS1_IVV"):
                    # Get master ivv channel
                    flood_vv = cv.imread(current_path, cv.IMREAD_ANYDEPTH)

                    if flood_vv is None:
                        print(current_path)

                elif file.startswith("MS1_IVH"):
                    # Get master ivh channel
                    flood_vh = cv.imread(current_path, cv.IMREAD_ANYDEPTH)

                elif file.startswith("SL1_IVV"):
                    # Get slave1 vv channel
                    sec1_vv = cv.imread(current_path, cv.IMREAD_ANYDEPTH)

                elif file.startswith("SL1_IVH"):
                    # Get sl1 vh channel
                    sec1_vh = cv.imread(current_path, cv.IMREAD_ANYDEPTH)

                elif file.startswith("SL2_IVV"):
                    # Get sl2 vv channel
                    sec2_vv = cv.imread(current_path, cv.IMREAD_ANYDEPTH)

                elif file.startswith("SL2_IVH"):
                    # Get sl2 vh channel
                    sec2_vh = cv.imread(current_path, cv.IMREAD_ANYDEPTH)

        # Concat channels
        flood = self.concat(flood_vv, flood_vh)
        pre_event_1 = self.concat(sec1_vv, sec1_vh)
        pre_event_2 = self.concat(sec2_vv, sec2_vh)

        # Hardcoded mean and std for all of Kuro Siwo (labeled + unlabeled part)
        mean = torch.tensor([0.0953, 0.0264])
        std = torch.tensor([0.0427, 0.0215])

        normalize = torchvision.transforms.Normalize(mean, std)
        flood = normalize(flood)
        pre_event_1 = normalize(pre_event_1)
        pre_event_2 = normalize(pre_event_2)

        image = torch.cat((flood, pre_event_1, pre_event_2), dim=0)
        image = einops.rearrange(image, "c h w -> h w c").numpy()
        transform = self.augmentations(image=image)
        image = transform["image"]
        image = einops.rearrange(image, "h w c -> c h w")
        image = torch.from_numpy(image)
        return image, flood, pre_event_1, pre_event_2


class SLCDataset(torch.utils.data.Dataset):
    def __init__(self, mode="train", configs=None):
        print('='*20)
        print('Initializing SLC Dataset')
        print('='*20)
        self.train_acts = configs["train_acts"]
        self.val_acts = configs["val_acts"]
        self.test_acts = configs["test_acts"]
        self.mode = mode
        self.configs = configs
        self.root_path = os.path.join(self.configs["slc_root_path"])
        if self.configs["task"] == "self-supervised" or self.configs["data_augmentations"]:
            self.augmentations = augmentations.get_augmentations(self.configs)
        else:
            self.augmentations = None

        self.non_valids = []

        # Load precomputed min-max stats for each SAR image or calculate them anew
        # self.min_max_random_events = self.update_min_max_stats()
        self.clz_stats = {1: 0, 2: 0, 3: 0}
        self.act_stats = {}
        if self.mode == "train":
            self.valid_acts = self.train_acts
            self.pickle_path = configs["train_json"]
        elif self.mode == "val":
            self.valid_acts = self.val_acts
            self.pickle_path = configs["test_json"]
        else:
            self.valid_acts = self.test_acts
            self.pickle_path = configs["test_json"]

        self.negative_grids = None
        total_grids = {}
        self.positive_records = []
        self.negative_records = []

        self.grids = json.load(open(self.pickle_path, "r"))  # get_grids(pickle_path=self.pickle_path)
        total_grids = self.grids

        all_activations = []
        all_activations.extend(self.train_acts)
        all_activations.extend(self.val_acts)
        all_activations.extend(self.test_acts)
        self.records = []
        for key in total_grids:
            record = {}
            record["id"] = key
            record["path"] = total_grids[key]["path"]

            record["type"] = None
            record["clz"] = total_grids[key]["clz"]
            activation = total_grids[key]["actid"]
            aoi = total_grids[key]["aoiid"]
            if configs["track"] == "Climatic":
                act_aoi = str(activation) + "_" + f"{aoi:02}"
            else:
                act_aoi = activation

            record["activation"] = activation
            if act_aoi in self.valid_acts:
                self.clz_stats[record["clz"]] += 1
                if act_aoi in self.act_stats:
                    self.act_stats[act_aoi] += 1
                else:
                    self.act_stats[act_aoi] = 1
                if self.configs["task"] == "diffusion-unsup":
                    # We will create a separate record per observation (pre1, pre2, flood) in order to
                    # ensure that the model will see every image during an epoch
                    # This will also allow us to compute appropriate weights for the loss functions
                    for t in ["pre1", "pre2", "flood"]:
                        tmp = record.copy()
                        tmp["type"] = t
                        self.records.append(tmp)
                        if key in self.grids:
                            self.positive_records.append(tmp)
                        else:
                            self.negative_records.append(tmp)
                else:
                    self.records.append(record)
                    if key in self.grids:
                        self.positive_records.append(record)
                    else:
                        self.negative_records.append(record)

            if act_aoi not in all_activations and act_aoi not in self.non_valids:
                print("Activation: ", activation, " not in Activations")
                self.non_valids.append(act_aoi)

        print("Samples per Climatic zone for mode: ", self.mode)
        print(self.clz_stats)
        print("Samples per Activation for mode: ", self.mode)
        print(self.act_stats)
        self.num_examples = len(self.records)
        self.activations = set([record["activation"] for record in self.records])

    def __len__(self):
        return self.num_examples

    def normalize(self, image):
        means = self.configs["slc_mean"]
        stds = self.configs["slc_std"]

        return means, stds, torchvision.transforms.Normalize(means, stds)(image)

    def __getitem__(self, idx):
        sample = self.records[idx]

        path = os.path.join(sample["path"])
        path = os.path.join(self.root_path, path)
        files = os.listdir(path)
        clz = sample["clz"]
        activation = sample["activation"]
        mask = None

        for file in files:
            current_path = str(os.path.join(path, file))
            if "xml" not in file:
                if file.startswith("MK0_MLU") and (sample["type"] is None):
                    # Get mask of flooded/perm water pixels
                    mask = cv.imread(current_path, cv.IMREAD_ANYDEPTH)
                elif file.startswith("MK0_MNA") and (sample["type"] is None):
                    # Get mask of valid pixels
                    valid_mask = cv.imread(current_path, cv.IMREAD_ANYDEPTH)
                elif file.startswith("MS1"):
                    # Get master ivv channel
                    flood = rio.open_rasterio(current_path).to_numpy()  # cv.imread(current_path, cv.IMREAD_ANYDEPTH)
                    if self.configs["uint8"]:
                        flood /= flood.max()
                        flood *= 255
                        flood = flood.astype(np.uint8)
                    if flood is None:
                        print(current_path)

                elif file.startswith("SL1"):
                    # Get slave1 vv channel
                    sec1 = rio.open_rasterio(current_path).to_numpy()
                    if self.configs["uint8"]:
                        sec1 /= sec1.max()
                        sec1 *= 255
                        sec1 = sec1.astype(np.uint8)

                elif file.startswith("SL2") and (sample["type"] not in ["flood", "pre1"]):
                    # Get sl2 vv channel
                    sec2 = rio.open_rasterio(current_path).to_numpy()
                    if self.configs["uint8"]:
                        sec2 /= sec2.max()
                        sec2 *= 255
                        sec2 = sec2.astype(np.uint8)
                elif file.startswith("MK0_DEM"):
                    # Get DEM
                    dem = rio.open_rasterio(current_path)

                    # NOTE: Nodata values are not NaN but a rather high float number
                    # Here we convert the nodata value to NaN and interpolate
                    nodata = dem.rio.nodata
                    dem_arr = dem.to_numpy()
                    nans = dem == nodata
                    if nans.any().item():
                        dem_arr[nans] = np.nan
                        dem[:] = dem_arr

                    dem = dem.rio.write_nodata(np.nan)
                    dem = dem.rio.interpolate_na()

                    dem = dem.to_numpy()

                    if self.configs['dem'] and not self.configs['slope']:
                        if self.configs["scale_input"] is not None:
                            normalization = transforms.Normalize(
                                mean=self.configs["slc_dem_mean"],
                                std=self.configs["slc_dem_std"],
                            )
                            dem = normalization(torch.from_numpy(dem))
                    elif self.configs['dem'] and self.configs['slope']:
                        # Get slope
                        rd_dem = rd.rdarray(dem.squeeze(), no_data=nodata)
                        slope = rd.TerrainAttribute(rd_dem, attrib="slope_riserun")
                        slope = np.asarray(slope.data)
                        dem = einops.rearrange(slope, "h w -> 1 h w")

                        if self.configs["scale_input"] is not None:
                            # Only support standarization for DEMs
                            normalization = transforms.Normalize(
                                mean=self.configs["slc_slope_mean"],
                                std=self.configs["slc_slope_std"],
                            )
                            dem = normalization(torch.from_numpy(dem))
        try:
            if flood.shape!= (4,224,224) or sec1.shape!= (4,224,224) or sec2.shape!= (4,224,224):
                #pad using albumentations
                pad = A.Compose([A.PadIfNeeded(
                    min_height=224,  # Optional[int]
                    min_width=224,  # Optional[int]
                    border_mode=cv.BORDER_CONSTANT,  # int
                    value=int(flood.mean()),  # Union[float, Sequence[float], NoneType]
                    mask_value=3,  # Union[float, Sequence[float], NoneType]
                    p=1.0,  # float
                )])
                flood = einops.rearrange(flood,"c h w -> h w c")
                sec1 = einops.rearrange(sec1,"c h w -> h w c")
                sec2 = einops.rearrange(sec2,"c h w -> h w c")

                try:
                    transform = pad(image=flood,mask=mask)
                except Exception as e:
                    print(e)
                    print(sample['path'])
                    print(flood.shape,sec1.shape,sec2.shape,mask.shape,flush=True)
                    exit(2)

                flood = transform["image"]

                mask = transform["mask"]

                transform = pad(image=sec1)
                sec1 = transform["image"]
                transform = pad(image=sec2)
                sec2 = transform["image"]
                flood = einops.rearrange(flood,"h w c -> c h w")
                sec1 = einops.rearrange(sec1,"h w c -> c h w")
                sec2 = einops.rearrange(sec2,"h w c -> c h w")
        except:
            print(sample['path'])

        if self.configs['scale_input'] == "normalize":
            flood = torch.from_numpy(flood).float()
            sec1 = torch.from_numpy(sec1).float()
            sec2 = torch.from_numpy(sec2).float()
            mask = torch.from_numpy(mask).long()
            sc_mean, scale_stdf , flood = self.normalize(flood)
            sc1, sc_std1 , sec1 = self.normalize(sec1)
            sc2, sc_std2, sec2 = self.normalize(sec2)

            if not self.configs["dem"]:
                return sc_mean, scale_stdf, flood, mask, sc1, sc_std1, sec1, sc2, sc_std2, sec2, clz, activation
            else:
                return sc_mean, scale_stdf, flood, mask, sc1, sc_std1, sec1, sc2, sc_std2, sec2, dem, clz, activation
        else:
            if not self.configs["dem"]:
                return flood, mask, sec1, sec2, clz, activation
            else:
                return flood, mask, sec1, sec2, dem, clz, activation


if __name__ == "__main__":
    configs = {
        "dem": False,
        "slope": False,
        "channels": ["vv", "vh"],
        "clamp_input": 0.15,
    }
    a = SSLDataset(configs=configs)
    a.__getitem__(0)