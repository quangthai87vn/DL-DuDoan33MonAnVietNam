# model/efficientnet_b0.py
import torch
import torch.nn as nn

class CustomEfficientNetB0(nn.Module):
    """
    EfficientNet-B0 (pretrained ImageNet) – tương thích trainer/classifi_main hiện tại.
    - num_classes: số lớp đầu ra (ví dụ 33)
    - pretrained: True => fine-tune từ ImageNet, False => train từ đầu
    - freeze_backbone: True => chỉ train classifier 5-10 epoch đầu (gợi ý)
    """
    def __init__(self, num_classes: int = 33,
                 pretrained: bool = True,
                 freeze_backbone: bool = False):
        super().__init__()

        # Ưu tiên torchvision (có sẵn ở môi trường bạn)
        try:
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = efficientnet_b0(weights=weights)
            in_features = backbone.classifier[1].in_features
            backbone.classifier[1] = nn.Linear(in_features, num_classes)
            self.backbone = backbone
            self._imagenet_normalize = {
                "mean": [0.485, 0.456, 0.406],
                "std":  [0.229, 0.224, 0.225]
            }
        except Exception:
            # Fallback: dùng timm nếu cần
            import timm
            backbone = timm.create_model(
                "efficientnet_b0",
                pretrained=pretrained,
                num_classes=num_classes,   # timm tự gắn classifier
                global_pool="avg"
            )
            self.backbone = backbone
            self._imagenet_normalize = {
                "mean": [0.485, 0.456, 0.406],
                "std":  [0.229, 0.224, 0.225]
            }

        # (Tùy chọn) đóng băng backbone – chỉ train classifier
        if freeze_backbone:
            for name, p in self.backbone.named_parameters():
                if "classifier" not in name:
                    p.requires_grad = False

        # tên xuất ra file checkpoint (trainer của bạn đang hỗ trợ _export_name)
        self._export_name = "MTL-EfficientNetB0"

    def forward(self, x):
        return self.backbone(x)


def efficientnet_b0_model(num_classes: int = 33,
                          pretrained: bool = True,
                          freeze_backbone: bool = False) -> nn.Module:
    """
    Factory để dùng trong classifi_main.py
    """
    return CustomEfficientNetB0(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )
