import math
import os
import time

import pyjson5 as json
import torch
import wandb
from tqdm import tqdm
from models.vision_transformer import ViT
import dataset.Dataset as Dataset
from models import mae as mae_model


def adjust_learning_rate(optimizer, epoch, configs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < configs["warmup_epochs"]:
        lr = configs["lr"] * epoch / configs["warmup_epochs"]
    else:
        lr = configs["min_lr"] + (configs["lr"] - configs["min_lr"]) * 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (epoch - configs["warmup_epochs"])
                / (configs["epochs"] - configs["warmup_epochs"])
            )
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def get_current_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_epoch(loader, mae, optimizer, epoch, configs, scaler):
    mae.train()
    configs["num_steps_per_epoch"] = (
        configs["num_samples_per_epoch"] // configs["batch_size"]
    )
    num_steps_per_epoch = configs["num_steps_per_epoch"]

    device = configs["gpu"]
    disable = False

    # Set up gradient accumulation
    if configs["accumulate_gradients"] is not None:
        batches_to_accumulate = configs["accumulate_gradients"]

    running_loss = 0.0
    number_of_batches = 0.0
    data_loading_time = 0
    loader_iter = loader.__iter__()
    for idx in tqdm(range(num_steps_per_epoch), disable=disable):
        start_time = time.time()
        batch = loader_iter.__next__()

        end_time = time.time()
        data_loading_time += end_time - start_time

        if (
            configs["accumulate_gradients"] is None
            or (idx + 1) % batches_to_accumulate == 0
            or (idx + 1) == num_steps_per_epoch
        ):
            optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=configs["mixed_precision"]):
            if (
                configs["accumulate_gradients"] is None
                or (idx + 1) % batches_to_accumulate == 0
                or (idx + 1) == num_steps_per_epoch
            ):
                # we use a per iteration (instead of per epoch) lr scheduler as done in official MAE implementation
                adjust_learning_rate(
                    optimizer, idx / num_steps_per_epoch + epoch, configs
                )

            image, flood, pre_event_1, pre_event_2 = batch

            image = image.to(device, non_blocking=True)

            loss = mae(image)

        running_loss += loss.item()
        number_of_batches += 1

        if idx % 100 == 0:
            log_dict = {
                "Epoch": epoch,
                "Iteration": idx,
                "train loss": running_loss / number_of_batches,
            }
            running_loss = 0.0
            number_of_batches = 0.0

            log_dict["Current Learning Rate"] = get_current_learning_rate(optimizer)

            if configs["wandb_activate"]:
                wandb.log(log_dict)
            else:
                print(log_dict)

        # Scale loss according to gradient accumulation
        if configs["accumulate_gradients"] is not None:
            loss = loss / batches_to_accumulate

        # If gradient accumulation is enabled, update weights every batches_to_accumulate iterations.
        if (
            configs["accumulate_gradients"] is None
            or (idx + 1) % batches_to_accumulate == 0
            or (idx + 1) == num_steps_per_epoch
        ):
            if configs["mixed_precision"]:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
    print("=" * 20)
    print("Epoch sampling statistics")
    print(number_of_batches)
    print("=" * 20)


def train(configs):
    print("=" * 20)
    print("Initializing MAE")
    print("=" * 20)
    mae_configs = json.load(open("configs/method/mae/mae.json", "r"))
    configs.update(mae_configs)
    if configs["mixed_precision"]:
        # Creates a GradScaler once at the beginning of training.
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    train_dataset = Dataset.SSLDataset(configs=configs)

    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configs["batch_size"],
        shuffle=False,
        num_workers=configs["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    # Calculate effective batch size
    if configs["accumulate_gradients"] is None:
        accumulated_batches = 1
    else:
        accumulated_batches = configs["accumulate_gradients"]

    configs["lr"] = configs["learning_rate"]

    # Scale learning rate
    configs["lr"] = configs["lr"] * accumulated_batches
    print("=" * 20)
    print("Scaled Learning Rate: ", configs["lr"])
    print("=" * 20)
    v = ViT(
        image_size=configs["image_size"],
        patch_size=configs["patch_size"],
        channels=configs["num_channels"],
        num_classes=configs["num_classes"],
        dim=configs["dim"],
        depth=configs["depth"],
        heads=configs["heads"],
        mlp_dim=configs["mlp_dim"],
    )
    model = mae_model.MAE(
        encoder=v,
        masking_ratio=configs[
            "masked_ratio"
        ],  # the paper recommended 75% masked patches
        decoder_dim=configs["decoder_dim"],  # paper showed good results with just 512
        decoder_depth=configs["decoder_depth"],  # anywhere from 1 to 8
        decoder_heads=configs["decoder_heads"],  # attention heads for decoder
    )
    model.to(configs["gpu"])
    optimizer = torch.optim.Adam(model.parameters(), lr=configs["lr"])

    if configs["wandb_activate"]:
        # Store wandb id to continue run
        id = wandb.util.generate_id()
        # json.dump({'run_id': id}, open(configs['checkpoint_path'] + '/id.json', 'w'))
        wandb.init(
            project=configs["wandb_project"],
            entity=configs["wandb_entity"],
            config=configs,
            id=id,
            resume="allow",
        )
        wandb.watch(model, log_freq=20)
    if configs["start_epoch"] is None:
        start_epoch = 0
    else:
        start_epoch = configs["start_epoch"]
    for epoch in range(start_epoch, configs["epochs"]):
        train_epoch(loader, model, optimizer, epoch, configs, scaler)
        if epoch % 1 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    configs["checkpoint_path"], "mae_" + str(epoch) + ".pt"
                ),
            )
            torch.save(
                model.encoder,
                os.path.join(
                    configs["checkpoint_path"], "vit_" + str(epoch) + ".pt"
                ),
            )

    torch.save(
        model.encoder.state_dict(),
        os.path.join(
            configs["checkpoint_path"], "mae_vit_" + str(configs["epochs"]) + ".pt"
        ),
    )
    torch.save(
        model.encoder,
        os.path.join(
            configs["checkpoint_path"],
            "trained_vit_" + str(configs["epochs"]) + ".pt",
        ),
    )
