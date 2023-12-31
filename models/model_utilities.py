from pathlib import Path

import einops
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from denoising_diffusion_pytorch import GaussianDiffusion, Unet

import models.upernet as upernet
import segmentation_models_siamese_pytorch as smsp

from .bit_cd import define_G
from .changeformer import ChangeFormerV6
from .hfanet import HFANet
from .siam_conc import SiamUnet_conc
from .siam_diff import SiamUnet_diff


class Decoder(nn.Module):
    def __init__(self, input_size, output_channels):
        super(Decoder, self).__init__()

        # Deconvolutional layers
        self.deconv1 = nn.ConvTranspose2d(1024, 128, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.up = nn.Upsample(scale_factor=2)

        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(
            64, output_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x):
        x = self.deconv1(x)
        x = self.relu(x)

        x = self.up(x)

        x = self.deconv2(x)

        x = self.relu(x)

        x = self.deconv3(x)

        return x


class FinetunerSegmentation(nn.Module):
    def __init__(self, encoder, configs=None, pool=False):
        super().__init__()
        self.configs = configs
        self.model = encoder
        self.model.pool = pool
        self.pool = pool
        if not self.pool:
            if configs["mlp"]:
                self.head = nn.Sequential(
                    nn.Conv2d(encoder.mlp_head.in_features, 512, kernel_size=1),
                    nn.ReLU(),
                    nn.Conv2d(512, configs["num_classes"], kernel_size=1),
                )
            elif configs["decoder"]:
                self.head = Decoder(
                    encoder.mlp_head.in_features, configs["num_classes"]
                )
            else:
                self.head = nn.Conv2d(
                    encoder.mlp_head.in_features, configs["num_classes"], kernel_size=1
                )
        else:
            self.head = nn.Linear(
                encoder.mlp_head.in_features,
                configs["num_classes"] * configs["image_size"] * configs["image_size"],
            )
        self.model.mlp_head = nn.Identity()

    def forward(self, x):
        x = x
        img_size = 224
        GS = img_size // self.configs["finetuning_patch_size"]
        x = self.model(x)

        if self.pool == False:
            x = einops.rearrange(x, "b (h w) c -> b (c) h w", h=GS, w=GS)
            if not self.configs["decoder"]:
                upsample = nn.Upsample(size=(img_size, img_size), mode="bilinear")

                x = upsample(x)

        x = self.head(x)
        return x


def initialize_segmentation_model(config, model_configs):
    if config["task"] == "diffusion-unsup":
        model = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=2)

        diffusion = GaussianDiffusion(
            model,
            channels=2,
            image_size=224,
            timesteps=1000,  # number of steps
            loss_type="l1",  # L1 or L2
        )
        return diffusion
    else:
        if config["method"].lower() == "unet":
            classes = config["num_classes"]
            model = smp.Unet(
                encoder_name=model_configs[
                    "backbone"
                ],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=model_configs[
                    "encoder_weights"
                ],  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=config[
                    "num_channels"
                ],  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=classes,  # model output channels (number of classes in your dataset)
            )
        elif config["method"].lower() == "upernet":
            model = upernet.UperNet(config)

        elif config["method"].lower() == "unetplusplus":
            model = smp.UnetPlusPlus(
                encoder_name=model_configs[
                    "backbone"
                ],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=model_configs[
                    "encoder_weights"
                ],  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=config[
                    "num_channels"
                ],  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=config[
                    "num_classes"
                ],  # model output channels (number of classes in your dataset)
            )

        elif config["method"].lower() == "deeplabv3":
            model = smp.DeepLabV3Plus(
                encoder_name=model_configs[
                    "backbone"
                ],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=model_configs[
                    "encoder_weights"
                ],  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=config[
                    "num_channels"
                ],  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=config[
                    "num_classes"
                ],  # model output channels (number of classes in your dataset)
            )
        elif config["method"] == "finetune":
            encoder = torch.load(config["encoder"], map_location="cpu")
            for param in encoder.parameters():
                param.requires_grad = not config["linear_eval"]
            model = FinetunerSegmentation(encoder=encoder, configs=config)
            from torchsummary import summary

            summary(model, input_size=((6, 224, 224)), device="cpu")

        return model


def initialize_cd_model(configs, model_configs, phase="train"):
    if configs["method"].lower() == "siam-conc":
        model = SiamUnet_conc(
            input_nbr=len(configs["channels"]), label_nbr=configs["num_classes"]
        )
    elif configs["method"].lower() == "siam-diff":
        model = SiamUnet_diff(
            input_nbr=len(configs["channels"]), label_nbr=configs["num_classes"]
        )
    elif configs["method"].lower() == "stanet":
        from .stanet import create_model

        if configs["resume_checkpoint"]:
            ckpt = Path(configs["resume_checkpoint"])
            if "best_segmentation" == ckpt.stem:
                with open(ckpt.parent / "best_segmentation.txt", "r") as f:
                    epoch = int(f.readline().strip())
            else:
                epoch = int(ckpt.stem.split("=")[1])
        else:
            ckpt = None
            epoch = 0

        opts = {
            # 'encoder_name': model_configs['encoder_name'],
            "pretrained": model_configs["pretrained"],
            # 'activation': model_configs['activation'],
            "in_channels": len(configs["channels"]),
            "num_classes": configs["num_classes"],
            "lr_policy": model_configs["lr_policy"],
            "lr_decay": model_configs["lr_decay"],
            "lr_decay_iters": model_configs["lr_decay_iters"],
            "epoch_count": configs["epochs"],
            "niter": model_configs["niter"],
            "niter_decay": model_configs["niter_decay"],
            "gpu_ids": [configs["gpu"]],
            "isTrain": (phase == "train"),
            "checkpoints_dir": configs["checkpoint_path"],
            "name": "STANet",
            "phase": phase,
            "continue_train": configs["resume_checkpoint"],
            "load_iter": ckpt,
            "verbose": True,
            "epoch": epoch,
            "model": model_configs["model"],
            "istest": (phase != "train"),
            "f_c": model_configs["f_c"],
            "arch": model_configs["arch"],
            "lr": model_configs["lr"],
            "beta1": model_configs["beta1"],
            "ds": model_configs["ds"],
            "SA_mode": model_configs["sa_mode"],
            "device": configs["device"],
        }

        model = create_model(opts)
    elif configs["method"].lower() == "bit-cd":
        model = define_G(model_configs, in_channels=len(configs["channels"]))
    elif configs["method"].lower() == "unet-siam-conc":
        model = smsp.Unet(
            encoder_name=model_configs[
                "backbone"
            ],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=model_configs[
                "encoder_weights"
            ],  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=len(configs["channels"]),
            classes=configs[
                "num_classes"
            ],  # model output channels (number of classes in your dataset)
            fusion_mode="conc",
        )
    elif configs["method"].lower() == "unet-siam-diff":
        model = smsp.Unet(
            encoder_name=model_configs[
                "backbone"
            ],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=model_configs[
                "encoder_weights"
            ],  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=len(configs["channels"]),
            classes=configs[
                "num_classes"
            ],  # model output channels (number of classes in your dataset)
            fusion_mode="diff",
        )
    elif configs["method"].lower() == "hfa-net":
        model = HFANet(
            input_channel=len(configs["channels"]),
            input_size=224,
            num_classes=configs["num_classes"],
        )
    elif configs["method"].lower() == "changeformer":
        model = ChangeFormerV6(
            embed_dim=model_configs["embed_dim"],
            input_nc=len(configs["channels"]),
            output_nc=configs["num_classes"],
            decoder_softmax=model_configs["decoder_softmax"],
        )

    if configs["resume_checkpoint"]:
        if configs["method"].lower() == "stanet":
            model.load_networks(epoch=epoch)
        else:
            checkpoint = torch.load(configs["resume_checkpoint"])
            model.load_state_dict(checkpoint["model_state_dict"])

    print(model)
    return model
