# model/mtl_efficientnet_b0.py
'''
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
'''


import os
from pathlib import Path
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# =========================
# 1. Model EfficientNet-B0
# =========================

class MTLEfficientNetB0(nn.Module):
    """
    Fine-tuned EfficientNet-B0 (pretrained on ImageNet).
    - Dùng backbone EfficientNet-B0
    - Thay classifier bằng head mới: Dropout -> Linear 512 -> BN -> ReLU -> Dropout -> Linear num_classes
    """

    def __init__(
        self,
        num_classes: int = 33,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.4,
    ):
        super().__init__()

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        # Freeze một phần backbone nếu muốn
        if freeze_backbone:
            freeze_list = [
                "features.0",  # stem conv
                "features.1",
                "features.2",
                "features.3",
            ]
            for name, param in self.backbone.named_parameters():
                if any(name.startswith(block) for block in freeze_list):
                    param.requires_grad = False

        # Thay classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),
        )

        self._export_name = "MTL-EfficientNetB0-FT"
        self._imagenet_normalize = {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }

    def forward(self, x):
        return self.backbone(x)


def build_model(
    num_classes: int,
    dropout: float = 0.4,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    device: torch.device | None = None,
) -> nn.Module:
    """
    Khởi tạo model + chuyển về channels_last như notebook.
    """
    model = MTLEfficientNetB0(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout_rate=dropout,
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device).to(memory_format=torch.channels_last)

    # Thử compile giống notebook (nếu PyTorch 2.x)
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        print("✅ torch.compile enabled")
    except Exception as e:
        print("⚠️ torch.compile skipped:", e)

    return model


# ==========================================
# 2. Focal CrossEntropy với class weights
# ==========================================

class FocalCrossEntropyWithWeights(nn.Module):
    """
    Focal CE + label smoothing + class weights (giống ý tưởng notebook).
    - class_weights: Tensor [C]
    - gamma: hệ số focal
    - smooth: label smoothing (0 -> one-hot cứng)
    """

    def __init__(self, class_weights: torch.Tensor, gamma: float = 1.5, smooth: float = 0.05):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: [B, C], targets: [B]
        num_classes = logits.size(1)

        # Label smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smooth / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smooth)

        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        # p_t = prob của class đúng (sau smoothing)
        pt = (probs * true_dist).sum(dim=1)
        focal_factor = (1.0 - pt).pow(self.gamma)

        # cross-entropy từng mẫu (đã smoothing)
        ce = -(true_dist * log_probs).sum(dim=1)

        # trọng số theo label thật
        w = self.class_weights[targets]

        loss = (w * focal_factor * ce).mean()
        return loss


def build_criterion_from_counts(
    class_counts: List[int],
    gamma: float = 1.5,
    smooth: float = 0.05,
    device: torch.device | None = None,
) -> nn.Module:
    """
    Tạo FocalLoss với trọng số lớp dựa trên tần suất trong train set.
    class_counts: list số mẫu mỗi class trong train set.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    counts = torch.tensor(class_counts, dtype=torch.float32, device=device)
    # Trọng số ngược tần suất (normalize cho dễ nhìn)
    inv_freq = 1.0 / (counts + 1e-6)
    class_weights = inv_freq / inv_freq.mean()

    print("Class weights:", class_weights.cpu().numpy())

    return FocalCrossEntropyWithWeights(class_weights=class_weights, gamma=gamma, smooth=smooth)


# =================================
# 3. Dataloaders + transforms giống NB
# =================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def create_dataloaders(
    data_dir: Path,
    img_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 8,
) -> Tuple[Dict[str, DataLoader], List[str], List[int]]:
    """
    Tạo dataloader cho Train/Val/Test + trả về class_names + class_counts(train).
    data_dir: folder gốc chứa 3 thư mục con Train/Validate/Test (giống notebook).
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "Train"
    val_dir = data_dir / "Validate"
    test_dir = data_dir / "Test"

    # Ép bất cứ ảnh nào cũng về RGB
    to_rgb = transforms.Lambda(lambda im: im.convert("RGB"))

    # Augment tương tự notebook: crop + jitter + blur + affine
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            shear=5,
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tfms)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tfms)

    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes:", class_names)

    # Đếm số mẫu từng lớp từ train_ds
    class_counts = [0] * num_classes
    for _, label in train_ds.samples:
        class_counts[label] += 1
    print("Class counts:", class_counts)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
    return loaders, class_names, class_counts


# ==============================
# 4. Optimizer + CosineAnnealing
# ==============================

def build_optim_sched(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    epochs: int = 100,
):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    return optimizer, scheduler


# =======================
# 5. Accuracy helper
# =======================

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()
