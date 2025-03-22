import pyjson5 as json
from datetime import datetime
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, JaccardIndex, Precision, Recall
from torchvision.transforms import Normalize

import dataset.Dataset as Dataset
from .bce_and_dice import BCEandDiceLoss


def create_checkpoint_directory(configs, model_configs):
    if "vit" in configs["method"].lower():
        checkpoint_path = (
            "checkpoints/"
            + configs["method"]
            + "_patch"
            + str(model_configs["patch_size"])
            + "_depth"
            + str(model_configs["depth"])
            + "_num_heads"
            + str(model_configs["num_heads"])
            + "/input_patches_"
            + str(len(configs["inputs"]))
            + "/"
            + configs["track"]
        )
    elif configs["task"] == "diffusion-unsup":
        checkpoint_path = "checkpoints/diffusion-unsup/"
    elif configs["task"] == "segmentation":
        if model_configs["backbone"]:
            checkpoint_path = (
                "checkpoints/"
                + model_configs["architecture"]
                + "/"
                + model_configs["backbone"]
                + "/"
                + "-".join(configs["channels"])
                + "_patches_"
                + str(len(configs["inputs"]))
                + "/"
                + configs["track"]
            )
        else:
            checkpoint_path = "checkpoints/" + model_configs["architecture"]
    elif configs["task"] == "mae":
        checkpoint_path = (
            "checkpoints/"
            + configs["method"].lower()
            + "/"
            + model_configs["backbone"].lower()
            + "/"
            + model_configs["backbone"].lower()
            + "_"
            + str(configs["num_channels"])
            + "/"
            + configs["track"]
        )
    elif configs["task"] == "cd":
        run_ts = datetime.now().strftime("%Y%m%d%H%M%S")
        checkpoint_path = (
            "checkpoints/" + configs["method"].lower() + f'/{configs["track"]}_{run_ts}'
        )
    elif configs["task"] == "finetune":
        checkpoint_path = "checkpoints/finetuning"
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    return checkpoint_path


def prepare_loaders(configs):
    if configs["track"] not in ["RandomEvents"]:
        print("=" * 20)
        print("No such track! We currently support only RandomEvents")
        print("=" * 20)
        exit(2)

    batch_size = configs["batch_size"]
    workers = configs["num_workers"]

    print("=" * 20)
    print("Initializing ", configs["track"])
    print("=" * 20)

    if "slc" in configs and configs["slc"]:
        train_dataset = Dataset.SLCDataset(mode="train", configs=configs)
        val_dataset = Dataset.SLCDataset(mode="val", configs=configs)
        test_dataset = Dataset.SLCDataset(mode="test", configs=configs)
    else:
        train_dataset = Dataset.Dataset(mode="train", configs=configs)
        val_dataset = Dataset.Dataset(mode="val", configs=configs)
        test_dataset = Dataset.Dataset(mode="test", configs=configs)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )

    print("Samples in Train Set: ", len(train_loader.dataset))
    print("Samples in Val Set: ", len(val_loader.dataset))
    print("Samples in Test Set: ", len(test_loader.dataset))
    return train_loader, val_loader, test_loader


def reverse_scale_img(img, x1, x2, configs):
    if configs["scale_input"] == "normalize":
        # x1 is the means, x2 the stds
        if img.ndim == 4:
            return (img * x2[:, :, None, None]) + x1[:, :, None, None]
        else:
            return (img * x2[:, None, None]) + x1[:, None, None]
    elif configs["scale_input"] in ["min-max", "custom"]:
        # x1 is the mins, x2 is the maxs
        if len(configs["channels"]) == 2:
            if img.ndim == 4:
                new_ch0 = (
                    img[:, 0, :, :]
                    * (
                        x2[:, :, None, None][:, 0, :, :]
                        - x1[:, :, None, None][:, 0, :, :]
                    )
                ) + x1[:, :, None, None][:, 0, :, :]
                new_ch1 = (
                    img[:, 1, :, :]
                    * (
                        x2[:, :, None, None][:, 1, :, :]
                        - x1[:, :, None, None][:, 1, :, :]
                    )
                ) + x1[:, :, None, None][:, 1, :, :]
                return torch.cat(
                    (new_ch0[:, None, :, :], new_ch1[:, None, :, :]), dim=1
                )
            else:
                new_ch0 = (
                    img[0, :, :]
                    * (x2[:, None, None][0, :, :] - x1[:, None, None][0, :, :])
                ) + x1[:, None, None][0, :, :]
                new_ch1 = (
                    img[1, :, :]
                    * (x2[:, None, None][1, :, :] - x1[:, None, None][1, :, :])
                ) + x1[:, None, None][1, :, :]
                return torch.cat((new_ch0[None, :, :], new_ch1[None, :, :]), dim=0)
        else:
            if img.ndim == 4:
                return (img * (x2[:, None, None] - x1[:, None, None])) + x1[
                    :, None, None
                ]
            else:
                return (img * (x2 - x1)) + x1
    elif isinstance(configs["scale_input"], list):
        # x1 is the mins, x2 is the maxs
        # The required min and max values are given
        new_min, new_max = [torch.tensor(i) for i in configs["scale_input"]]

        if len(configs["channels"]) == 2:
            mid_img = (img - new_min) / (new_max - new_min)

            if img.ndim == 4:
                new_ch0 = (
                    mid_img[:, 0, :, :]
                    * (
                        x2[:, :, None, None][:, 0, :, :]
                        - x1[:, :, None, None][:, 0, :, :]
                    )
                ) + x1[:, :, None, None][:, 0, :, :]
                new_ch1 = (
                    mid_img[:, 1, :, :]
                    * (
                        x2[:, :, None, None][:, 1, :, :]
                        - x1[:, :, None, None][:, 1, :, :]
                    )
                ) + x1[:, :, None, None][:, 1, :, :]

                new_img = torch.cat(
                    (new_ch0[:, None, :, :], new_ch1[:, None, :, :]), dim=1
                )
            else:
                new_ch0 = (
                    mid_img[0, :, :]
                    * (x2[:, None, None][0, :, :] - x1[:, None, None][0, :, :])
                ) + x1[:, None, None][0, :, :]
                new_ch1 = (
                    mid_img[1, :, :]
                    * (x2[:, None, None][1, :, :] - x1[:, None, None][1, :, :])
                ) + x1[:, None, None][1, :, :]

                new_img = torch.cat((new_ch0[None, :, :], new_ch1[None, :, :]), dim=0)

            return new_img
        else:
            if img.ndim == 4:
                return (
                    torch.mul(
                        (img - new_min[:, None, None])
                        / (new_max[:, None, None] - new_min[:, None, None]),
                        (x2[:, None, None] - x1[:, None, None]),
                    )
                    + x1[:, None, None]
                )
            else:
                return torch.mul((img - new_min) / (new_max - new_min), (x2 - x1)) + x1


def initialize_metrics(configs, mode="all"):
    accuracy = Accuracy(
        task="multiclass",
        average="none",
        multidim_average="global",
        num_classes=configs["num_classes"] + 1,
        ignore_index=3,
    ).to(configs["device"])
    fscore = F1Score(
        task="multiclass",
        num_classes=configs["num_classes"] + 1,
        average="none",
        multidim_average="global",
        ignore_index=3,
    ).to(configs["device"])
    precision = Precision(
        task="multiclass",
        average="none",
        num_classes=configs["num_classes"] + 1,
        multidim_average="global",
        ignore_index=3,
    ).to(configs["device"])
    recall = Recall(
        task="multiclass",
        average="none",
        num_classes=configs["num_classes"] + 1,
        multidim_average="global",
        ignore_index=3,
    ).to(configs["device"])
    iou = JaccardIndex(
        task="multiclass",
        num_classes=configs["num_classes"] + 1,
        average="none",
        multidim_average="global",
        ignore_index=3,
    ).to(configs["device"])

    return accuracy, fscore, precision, recall, iou


def init_lr_scheduler(optimizer, configs, model_configs, model_name=None, steps=None):
    # Get the required LR scheduling
    if model_name is not None:
        lr_schedule = model_configs[model_name]["lr_schedule"]
    else:
        lr_schedule = model_configs["lr_schedule"]

    # Initialize the LR scheduler
    if lr_schedule == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    elif lr_schedule is None:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda _: 1, last_epoch=-1
        )
    elif lr_schedule == "linear":

        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(configs["epochs"] + 1)
            return lr_l

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda_rule
        )
    elif lr_schedule == "step":
        step_size = configs["epochs"] // 3
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    else:
        raise NotImplementedError(
            f"{lr_schedule} LR scheduling is not yet implemented!"
        )

    # Load checkpoint (if any)
    if configs["resume_checkpoint"]:
        checkpoint = torch.load(configs["resume_checkpoint"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    return lr_scheduler


def create_loss(configs, mode="val"):
    if configs["loss_function"] == "cross_entropy":
        # Initialize class weights
        if "class_weights" in configs.keys():
            class_weights = torch.tensor(configs["class_weights"])
        else:
            class_weights = torch.tensor([1.0, 1.0, 1.0])
        if mode == "train":
            print("Creating cross entropy loss with class weights")
            print(class_weights)
            return nn.CrossEntropyLoss(weight=class_weights, ignore_index=3).to(
                configs["device"]
            )
        else:
            return nn.CrossEntropyLoss(ignore_index=3).to(configs["device"])

    elif configs["loss_function"] == "iou":
        return smp.losses.LovaszLoss(mode="multiclass", ignore_index=3)
    elif configs["loss_function"] == "dice":
        return smp.losses.DiceLoss(mode="multiclass", ignore_index=3)
    elif configs["loss_function"] == "focal":
        if "class_weights" in configs.keys():
            class_weights = torch.tensor(configs["class_weights"])
        else:
            class_weights = torch.tensor([1.0, 1.0, 1.0])

        return torch.hub.load(
            "adeelh/pytorch-multi-class-focal-loss",
            model="FocalLoss",
            alpha=class_weights,
            gamma=2,
            ignore_index=3,
            reduction="mean",
            force_reload=False,
        ).to(configs["device"])
    elif configs['loss_function'] == 'ce+dice':
        if 'class_weights' in configs.keys():
            class_weights = torch.tensor(configs['class_weights'])
        else:
            class_weights = torch.tensor([1.0, 1.0, 1.0])
        return BCEandDiceLoss(weights=class_weights, ignore_index=3, use_softmax=True).to(configs['device'])


def update_config(config, args=None):
    # Load data related configs
    data_config_path = "configs/train/data_config.json"
    data_config = json.load(open(data_config_path, "r"))
    config.update(data_config)

    if args is not None:
        if args.inputs is not None:
            config["inputs"] = args.inputs
        if args.dem:
            config["dem"] = args.dem
            if args.slope:
                config["slope"] = args.slope

    # Load train related configs
    train_config_path = "configs/train/train_config.json"
    train_config = json.load(open(train_config_path, "r"))
    config.update(train_config)

    if config["task"] == "self-supervised" or config["data_augmentations"]:
        # Load augmentation settings
        augmentation_config = json.load(
            open("configs/augmentations/augmentation.json", "r")
        )
        config.update(augmentation_config)

    # Compute total number of input channels
    if (config['task'] == 'cd') or (config['method'] == 'convlstm'):
        config['num_channels'] = len(config['channels'])
        if config['dem']:
            config['num_channels'] += 1
    else:
        config['num_channels'] = len(config['channels']) * len(config['inputs'])
        if config['dem']:
            config['num_channels'] += 1

    if "slc" in config and config["slc"]:
        if config['dem']:
            config["num_channels"] = ((config["num_channels"] - 1) * 2) + 1
        else:
            config["num_channels"] = config["num_channels"]*2

    if config["weighted"] and config["track"] == "RandomEvents":
        config["class_weights"] = [
            0.3715753140309927,
            14.009780283125977,
            8.20405370357821,
        ]
    else:
        config["class_weights"] = [1.0, 1.0, 1.0]

    # Get device
    if config["gpu"] is not None:
        device = f'cuda:{config["gpu"]}'
    else:
        device = "cpu"
    config["device"] = device

    # Prepare aois

    config = define_tracks(config)
    print("Configs updated")
    return config


def define_tracks(configs):        
    train_acts = configs['train_acts']
    val_acts = configs['val_acts']
    test_acts = configs['test_acts']
    print("train activations ", len(train_acts))
    print("val activations ", len(val_acts))
    print("test activations ", len(test_acts))
    return configs
