from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# =========================
# 1. Model EfficientNet-B0
# =========================

class MTLEfficientNetB0(nn.Module):
    """
    Fine-tuned EfficientNet-B0 (pretrained on ImageNet) cho bài toán 33 món ăn.
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

        # Freeze 1 phần backbone nếu muốn
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

        # Thay classifier mặc định bằng head custom
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_model(
    num_classes: int,
    dropout: float = 0.4,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    device: torch.device | None = None,
) -> nn.Module:
    """
    Khởi tạo model + chuyển về channels_last, compile nếu có thể.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MTLEfficientNetB0(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout_rate=dropout,
    )

    model = model.to(device).to(memory_format=torch.channels_last)

    # Thử torch.compile (PyTorch >= 2.0)
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
    Focal CrossEntropy + label smoothing + class weights.
    - class_weights: Tensor [C]
    - gamma: hệ số focal
    - smooth: label smoothing (0 -> one-hot cứng)
    """

    def __init__(self, class_weights: torch.Tensor, gamma: float = 1.5, smooth: float = 0.05):
        super().__init__()
        # Lưu class_weights dưới dạng buffer để tự move theo model.to(device)
        self.register_buffer("class_weights", class_weights)
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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

        # trọng số theo nhãn thật
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
    inv_freq = 1.0 / (counts + 1e-6)
    class_weights = inv_freq / inv_freq.mean()

    print("Class weights (normalized):", class_weights.detach().cpu().numpy())

    return FocalCrossEntropyWithWeights(class_weights=class_weights, gamma=gamma, smooth=smooth)


# =================================
# 3. Dataloaders + transforms
# =================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def create_dataloaders(
    data_dir: Path,
    img_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 8,
) -> Tuple[Dict[str, DataLoader], List[str], List[int]]:
    """
    Tạo dataloader cho Train/Validate/Test + trả về class_names + class_counts(train).

    Train loader dùng:
    - Augmentation mạnh (crop, flip, rotate, jitter, affine, blur, erasing)
    - WeightedRandomSampler: oversample lớp ít, giảm lớp nhiều.
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "Train"
    val_dir   = data_dir / "Validate"
    test_dir  = data_dir / "Test"

    # Ép mọi ảnh về RGB
    to_rgb = transforms.Lambda(lambda im: im.convert("RGB"))

    # ==== Data Augmentation cho TRAIN ====
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(
            img_size,
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),

        transforms.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.20,
            hue=0.05,
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            shear=5,
        ),

        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
            p=0.2,
        ),

        to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),

        transforms.RandomErasing(
            p=0.25,
            scale=(0.02, 0.08),
            ratio=(0.3, 3.3),
            value='random',
            inplace=False,
        ),
    ])

    # ==== Transform cho VAL/TEST (không augment) ====
    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Dataset
    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds   = datasets.ImageFolder(val_dir,   transform=eval_tfms)
    test_ds  = datasets.ImageFolder(test_dir,  transform=eval_tfms)

    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes:", class_names)

    # Đếm số mẫu từng lớp từ train_ds
    class_counts = [0] * num_classes
    for _, label in train_ds.samples:
        class_counts[label] += 1
    print("Class counts:", class_counts)

    # ===== WeightedRandomSampler cho TRAIN =====
    counts = np.array(class_counts, dtype=np.float32)
    inv_freq = 1.0 / (counts + 1e-6)
    inv_freq = inv_freq / inv_freq.sum()  # normalize

    sample_weights = [inv_freq[label] for _, label in train_ds.samples]
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),   # mỗi epoch ~ size train set
        replacement=True,
    )

    # DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,          # dùng sampler, không shuffle
        shuffle=False,
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

    loaders: Dict[str, DataLoader] = {
        "train": train_loader,
        "val":   val_loader,
        "test":  test_loader,
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

