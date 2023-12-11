from transformers import (
    ConvNextConfig,
    UperNetConfig,
    UperNetForSemanticSegmentation,
    AutoConfig,
)
import torch
import torch.nn as nn

# backbone_config = ConvNextConfig(out_features=["stage1", "stage2", "stage3", "stage4"])
backbones = {
    "swin_base": "openmmlab/upernet-swin-base",
    "swin_tiny": "openmmlab/upernet-swin-tiny",
    "swin_small": "openmmlab/upernet-swin-small",
    "convnext_tiny": "openmmlab/upernet-convnext-tiny",
    "convnext_small": "openmmlab/upernet-convnext-small",
    "convnext_base": "openmmlab/upernet-convnext-base",
}


class UperNet(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        if config["backbone"] not in backbones:
            print("Backbone: ", config["backbone"], " Not supported!")
            exit(2)

        if "convnext" in config["backbone"]:
            self.model = UperNetForSemanticSegmentation.from_pretrained(
                backbones[config["backbone"]]
            )
            self.model.num_labels = config["num_classes"]

            out_channels = self.model.backbone.embeddings.patch_embeddings.out_channels
            kernel_size = self.model.backbone.embeddings.patch_embeddings.kernel_size
            stride = self.model.backbone.embeddings.patch_embeddings.stride
            if config["num_channels"] != 3:
                self.model.backbone.embeddings.num_channels = config["num_channels"]
                self.model.config.backbone_config.num_channels = config["num_channels"]
                self.model.backbone.embeddings.patch_embeddings = nn.Conv2d(
                    config["num_channels"],
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                )

            self.model.decode_head.classifier = nn.Conv2d(
                512, config["num_classes"], kernel_size=(1, 1), stride=(1, 1)
            )

        elif "swin" in config["backbone"]:
            self.model = UperNetForSemanticSegmentation.from_pretrained(
                backbones[config["backbone"]]
            )

            out_channels = (
                self.model.backbone.embeddings.patch_embeddings.projection.out_channels
            )
            kernel_size = (
                self.model.backbone.embeddings.patch_embeddings.projection.kernel_size
            )
            stride = self.model.backbone.embeddings.patch_embeddings.projection.stride
            if config["num_channels"] != 3:
                self.model.backbone.embeddings.patch_embeddings.num_channels = config[
                    "num_channels"
                ]
                self.model.backbone.embeddings.patch_embeddings.projection = nn.Conv2d(
                    config["num_channels"],
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            self.model.num_labels = config["num_classes"]
            self.model.decode_head.classifier = nn.Conv2d(
                512, config["num_classes"], kernel_size=(1, 1), stride=(1, 1)
            )
            print(self.model)

    def forward(self, x):
        x = self.model(x, return_dict=True)
        return x["logits"]


"""config= {'num_channels':3,'num_classes':5,'backbone':'convnext_tiny'}
k = torch.randn((4,config['num_channels'],224,224))

model = UperNet(config)
logits = model(k)
print(logits.shape)
print(type(logits))"""
