dependencies = ["einops"]

import torch
import torch.nn as nn
import torchvision

from models.seg_finetuner import FinetunerSegmentation
from models.snunet import SNUNet_ECAM
from models.vision_transformer import ViT


# Hard-coded because these should be decoupled from changing configs
IMG_MEANS = [0.0953, 0.0264]
IMG_STDS = [0.0427, 0.0215]
EPS = 1e-7


class PreprocessingWrapper(nn.Module):
    """
    A model wrapper which accepts a list of images shaped [B, C, H, W]
    where C=2, as (vv, vh) Sentinel-1 images. It will normalise and
    reshape as necessary, and provides a consistent interface across models.

    The list can be either 2 or 3 elements long, representing the timesteps:
        [pre1] pre2 post
    """

    def __init__(self, model, means, stds, append_ratio=False, cat_input=True):
        super().__init__()
        self.model = model
        self.means = means
        self.stds = stds
        self.append_ratio = append_ratio
        self.cat_input = cat_input
        self.transform = torchvision.transforms.Normalize(means, stds)

    def forward(self, images, dem=None):
        if dem is not None:
            raise NotImplementedError("DEM not implemented for model wrapper")

        # Normalise images
        norm_images = [self.transform(image) for image in images]

        if self.append_ratio:
            # Append vh/vv to existing image (vv, vh) channels
            norm_images = [torch.cat((image, (image[:, 1:2] / (image[:, 0:1] + EPS))), dim=1) for image in norm_images]

        # Input concat on channels, or as separate inputs
        if self.cat_input:
            return self.model(torch.cat(norm_images, dim=1))
        else:
            return self.model(*norm_images)


def vit_decoder(pretrained=False):
    vit = ViT(
        image_size=224,
        patch_size=16,
        channels=6,
        num_classes=1000,
        dim=1024,
        depth=24,
        heads=16,
        mlp_dim=2048,
    )
    configs = {
        "finetuning_patch_size": 16,
        "num_classes": 3,
        "mlp": False,
        "decoder": True,
    }
    model = FinetunerSegmentation(encoder=vit, configs=configs)
    model.input_hint = "Input is 3x Image(VV, VH) as (pre1, pre2, post)"
    if pretrained:
        # TODO: Replace with real release URL
        release_url = (
            "https://github.com/Multihuntr/KuroSiwo/releases/download/v0.0.1-alpha/kurosiwo_vit_decoder_weights.pt"
        )
        model.load_state_dict(torch.hub.load_state_dict_from_url(release_url))

    wrapped = PreprocessingWrapper(model, IMG_MEANS, IMG_STDS, append_ratio=False, cat_input=True)
    return wrapped.eval()


def snunet(pretrained=False):
    model = SNUNet_ECAM(3, 3, 32)
    model.input_hint = "Input is 2x Image(VV, VH) as (pre1, post)"
    if pretrained:
        # TODO: Replace with real release URL
        release_url = "https://github.com/Multihuntr/KuroSiwo/releases/download/v0.0.1-alpha/kurosiwo_snunet_weights.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(release_url))

    wrapped = PreprocessingWrapper(model, IMG_MEANS, IMG_STDS, append_ratio=True, cat_input=False)
    return wrapped.eval()
