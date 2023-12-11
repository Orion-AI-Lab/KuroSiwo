import albumentations as A


def get_augmentations(config):
    augmentations = config["augmentations"]
    independend_aug = []
    for k, v in augmentations.items():
        if k == "RandomResizedCrop":
            aug = A.augmentations.RandomResizedCrop(
                height=v["value"],
                width=v["value"],
                p=v["p"],
                scale=tuple(v["scale"]),
                interpolation=v["interpolation"],
            )
        elif k == "HorizontalFlip":
            aug = A.augmentations.HorizontalFlip(p=v["p"])
        elif k == "VerticalFlip":
            aug = A.augmentations.VerticalFlip(p=v["p"])
        elif k == "GaussianBlur":
            aug = A.augmentations.GaussianBlur(sigma_limit=v["sigma_limit"], p=v["p"])
        elif k == "ElasticTransform":
            aug = A.augmentations.ElasticTransform(
                alpha=v["alpha"],
                sigma=v["sigma"],
                alpha_affine=v["alpha_affine"],
                interpolation=v["interpolation"],
                border_mode=v["border_mode"],
                value=v["value"],
                mask_value=v["mask_value"],
                same_dxdy=v["same_dxdy"],
                approximate=v["approximate"],
                p=v["p"],
            )
        elif k == "Cutout":
            aug = A.augmentations.CoarseDropout(p=v["p"])
        elif k == "GaussianNoise":
            aug = A.augmentations.GaussNoise(p=v["p"])
        elif k == "MultNoise":
            aug = A.augmentations.MultiplicativeNoise(p=v["p"])
        independend_aug.append(aug)
    return A.Compose(independend_aug)
