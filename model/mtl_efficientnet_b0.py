# model/mtl_efficientnet_b0.py
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class MTLEfficientNetB0(nn.Module):
    """
    Fine-tuned EfficientNet-B0 (pretrained on ImageNet).
    - Freeze phần backbone đầu để giữ đặc trưng low-level
    - Thay classifier bằng head mới gồm Dropout + Linear + ReLU
    - num_classes: số lớp (ví dụ 33 món ăn)
    """

    def __init__(self, num_classes: int = 33,
                 pretrained: bool = True,
                 freeze_backbone: bool = True,
                 dropout_rate: float = 0.4):
        super().__init__()

        # === 1. Load backbone EfficientNet B0 ===
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        # === 2. Freeze các layer đầu (conv1..MBConv4) nếu được yêu cầu ===
        if freeze_backbone:
            freeze_list = [
                "features.0",  # stem conv
                "features.1",  # MBConv1
                "features.2",
                "features.3",  # MBConv4
            ]
            for name, param in self.backbone.named_parameters():
                if any(name.startswith(block) for block in freeze_list):
                    param.requires_grad = False

        # === 3. Xây head classifier mới ===
        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

        # === 4. Lưu meta cho trainer ===
        self._export_name = "MTL-EfficientNetB0-FT"
        self._imagenet_normalize = {
            "mean": [0.485, 0.456, 0.406],
            "std":  [0.229, 0.224, 0.225]
        }

    def forward(self, x):
        return self.backbone(x)


def mtl_efficientnet_b0_model(num_classes: int = 33,
                              pretrained: bool = True,
                              freeze_backbone: bool = True) -> nn.Module:
    """
    Factory để gọi từ classifi_main.py
    """
    return MTLEfficientNetB0(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )



