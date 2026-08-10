dependencies = ["einops", "richdem"]

import einops
import richdem as rd
import numpy as np
import torch
import torch.nn as nn
import torchvision

from models.seg_finetuner import FinetunerSegmentation
from models.snunet import SNUNet_ECAM
from models.vision_transformer import ViT


# Hard-coded because these should be decoupled from changing configs
IMG_MEANS = [0.0953, 0.0264]
IMG_STDS = [0.0427, 0.0215]
DEM_MEAN = 67.0293
DEM_STD = 1765.0062
SLOPE_MEAN = 2.9482
SLOPE_STD = 79.2493
EPS = 1e-7


def _calculate_slope(dem, nodata):
    """
    Calculates the slope of the DEM. DEM should be shaped [b h w]
    """
    res = np.empty_like(dem)
    for i in range(dem.shape[0]):
        dem_i = rd.rdarray(dem[i], no_data=nodata)
        dem_i = rd.TerrainAttribute(dem_i, attrib="slope_riserun")
        res[i] = np.asarray(dem_i.data)
    return res


class PreprocessingWrapper(nn.Module):
    """
    A model wrapper which accepts a list of images shaped [B, C, H, W]
    where C=2, as (vv, vh) Sentinel-1 images. It will normalise and
    reshape as necessary, and provides a consistent interface across models.

    The list can be either 2 or 3 elements long, representing the timesteps:
        [pre2] pre1 post

    If providing DEM, it must be provided as a numpy array.
    """

    def __init__(
        self,
        model,
        means,
        stds,
        append_ratio=False,
        cat_input=True,
        img_order_forward=True,
        requires_dem=False,
        requires_slope=False,
        dem_mean=None,
        dem_std=None,
        slope_mean=None,
        slope_std=None,
    ):
        super().__init__()
        self.model = model
        self.means = means
        self.stds = stds
        self.append_ratio = append_ratio
        self.cat_input = cat_input
        self.img_order_forward = img_order_forward
        self.requires_dem = requires_dem
        self.requires_slope = requires_slope
        self.dem_mean = dem_mean
        self.dem_std = dem_std
        self.transform = torchvision.transforms.Normalize(means, stds)
        if (requires_dem) and (dem_mean is None or dem_std is None):
            raise Exception("Using a DEM requires normalisation parameters.")
        self.dem_transform = torchvision.transforms.Normalize(dem_mean, dem_std)
        if (requires_slope) and (slope_mean is None or slope_std is None):
            raise Exception("Using a slope requires normalisation parameters.")
        self.slope_transform = torchvision.transforms.Normalize(slope_mean, slope_std)

    def forward(self, images, dem=None, dem_nodata=np.nan):
        if (self.requires_dem or self.requires_slope) and dem is None:
            raise Exception("This model requires a DEM.")

        # Normalise images
        norm_images = [self.transform(image) for image in images]

        if self.append_ratio:
            # Append vh/vv to existing image (vv, vh) channels
            norm_images = [torch.cat((image, (image[:, 1:2] / (image[:, 0:1] + EPS))), dim=1) for image in norm_images]

        if self.requires_dem or self.requires_slope:
            dtype = norm_images[0].dtype
            device = norm_images[0].device
            if self.requires_slope:
                slope = _calculate_slope(dem, dem_nodata)
                slope = torch.tensor(slope, dtype=dtype, device=device)
                norm_dem = self.slope_transform(slope)
            else:
                dem = torch.tensor(dem, dtype=dtype, device=device)
                norm_dem = self.dem_transform(dem)
            # Append DEM to existing image channels
            norm_dem = einops.rearrange(norm_dem, "b h w -> b 1 h w")
            norm_images = [torch.cat((image, norm_dem), dim=1) for image in norm_images]

        if not self.img_order_forward:
            norm_images = list(reversed(norm_images))

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
    model.input_hint = "Input is 3x Image(VV, VH) as (pre2, pre1, post)"
    if pretrained:
        release_url = (
            "https://github.com/Multihuntr/KuroSiwo/releases/download/v0.0.1-alpha/kurosiwo_vit_decoder_weights.pt"
        )
        model.load_state_dict(torch.hub.load_state_dict_from_url(release_url))

    wrapped = PreprocessingWrapper(model, IMG_MEANS, IMG_STDS, append_ratio=False, cat_input=True, img_order_forward=False)
    return wrapped.eval()


def snunet(pretrained=False):
    model = SNUNet_ECAM(3, 3, 32)
    model.input_hint = "Input is 2x Image(VV, VH) as (pre1, post) and dem as a np.ndarray"
    if pretrained:
        release_url = "https://github.com/Multihuntr/KuroSiwo/releases/download/v0.0.1-alpha/kurosiwo_snunet_weights.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(release_url))

    wrapped = PreprocessingWrapper(
        model,
        IMG_MEANS,
        IMG_STDS,
        append_ratio=False,
        cat_input=False,
        requires_slope=True,
        slope_mean=SLOPE_MEAN,
        slope_std=SLOPE_STD,
    )
    return wrapped.eval()
