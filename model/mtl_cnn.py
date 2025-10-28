# model/mobilenet.py
import torch
import torch.nn as nn

class CustomMobileNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        def conv_dw(in_ch, out_ch, stride):
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            *[conv_dw(512, 512, 1) for _ in range(5)],
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)          # [B, 1024, 1, 1]
        x = x.view(x.size(0), 1024)   # [B, 1024]
        return self.classifier(x)

def mtl_cnn_v1(num_classes: int) -> nn.Module:
    """Factory để classifi_main gọi giống cnn.miniVGG."""
    return CustomMobileNet(num_classes=num_classes)
