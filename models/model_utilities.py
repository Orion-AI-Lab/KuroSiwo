from pathlib import Path

import einops
#import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from denoising_diffusion_pytorch import GaussianDiffusion, Unet

import models.upernet as upernet
from .adhr_cdnet import ADHR
from .bit_cd import define_G
from .changeformer import ChangeFormerV6
from .hfanet import HFANet
from .siam_conc import SiamUnet_conc
from .siam_diff import SiamUnet_diff
import segmentation_models_pytorch as smp
from .snunet import SNUNet_ECAM 
import segmentation_models_pytorch as smp
from .seg_finetuner import FinetunerSegmentation, Decoder


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
            input_nbr=len(configs["num_channels"]), label_nbr=configs["num_classes"]
        )
    elif configs["method"].lower() == "siam-diff":
        model = SiamUnet_diff(
            input_nbr=len(configs["num_channels"]), label_nbr=configs["num_classes"]
        )
    elif configs["method"].lower() == "bit-cd":
        model = define_G(model_configs, in_channels=len(configs["num_channels"]))
    elif configs["method"].lower() == "hfa-net":
        model = HFANet(
            input_channel=len(configs["num_channels"]),
            input_size=224,
            num_classes=configs["num_classes"],
        )
    elif configs["method"].lower() == "changeformer":
        model = ChangeFormerV6(
            embed_dim=model_configs["embed_dim"],
            input_nc=len(configs["num_channels"]),
            output_nc=configs["num_classes"],
            decoder_softmax=model_configs["decoder_softmax"],
        )
    elif configs['method'].lower() == 'snunet':
        model = SNUNet_ECAM(
            configs['num_channels'],
            configs['num_classes'],
            base_channel=model_configs['base_channel']
        )
    elif configs['method'].lower() == 'adhr-cdnet':
        model = ADHR(
            in_channels=configs['num_channels'],
            num_classes=configs['num_classes']
        )
    elif configs['method'].lower() == 'transunet-cd':
        model = TransUNet_CD(
            img_dim=224,
            in_channels=configs['num_channels'],
            out_channels=configs['out_channels'],
            head_num=configs['head_num'],
            mlp_dim=configs['mlp_dim'],
            block_num=configs['block_num'],
            patch_dim=configs['patch_dim'],
            class_num=configs['num_classes'],
            siamese=configs['siamese']
        )

    if configs["resume_checkpoint"]:
        checkpoint = torch.load(configs["resume_checkpoint"], map_location=configs['device'])
        model.load_state_dict(checkpoint["model_state_dict"])

    print(model)
    return model
