# model/mobilenet_v4.py
import torch.nn as nn
import timm

class CustomMobileNetV4(nn.Module):
    def __init__(self, num_classes=33, pretrained=True, freeze_backbone=False):
        super().__init__()
        # Tên model KHÔNG có .e2405
        self.backbone = timm.create_model(
            "mobilenetv4_conv_small",  # hoặc _medium / _large
            pretrained=pretrained,
            num_classes=num_classes,   # để timm tự gắn classifier đúng
            global_pool="avg"
        )

        # Đóng băng backbone (chỉ train classifier)
        if freeze_backbone:
            for name, p in self.backbone.named_parameters():
                if "classifier" not in name:
                    p.requires_grad = False

    def forward(self, x):
        return self.backbone(x)
