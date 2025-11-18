import os
import sys
import json
import csv
import time
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Đảm bảo import được models/efficientnet_b0.py
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR / "models"))

from efficientnet_b0 import (
    build_model,
    create_dataloaders,
    build_criterion_from_counts,
    build_optim_sched,
    accuracy_from_logits,
)


# ========================
# 1. CẤU HÌNH TOÀN CỤC
# ========================

DATA_DIR = Path("/media/mtl/DATA 6TB/AI DATASET/vietnamese-foods/Images")
RUN_NAME = "MTL_TGFOOD_01"

IMG_SIZE = 224
EPOCHS = 100
BATCH_SIZE = 64
NUM_WORKERS = 8

SEED = 1337
LR = 3e-4
WEIGHT_DECAY = 1e-4

GAMMA = 1.5        # focal
SMOOTH = 0.05      # label smoothing
DROPOUT = 0.4
PRETRAINED = True
FREEZE_BACKBONE = False

# Thư mục run
RUN_DIR = ROOT_DIR / "runs" / RUN_NAME
CKPT_DIR = RUN_DIR / "checkpoints"
IMAGES_DIR = ROOT_DIR / "images"
RUN_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = RUN_DIR / "history.csv"
HISTORY_JSON = RUN_DIR / "history.json"

# SAVE META GLOBAL
RUNS_META_DIR = ROOT_DIR.parent / "runs_meta"
RUNS_META_DIR.mkdir(parents=True, exist_ok=True)


# =====================
# 2. SET SEED + DEVICE
# =====================

def set_seed(seed: int = 1337):
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 2.1. Dataloaders
    loaders, class_names, class_counts = create_dataloaders(
        DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    num_classes = len(class_names)

    # 2.2. SAVE runs_meta (giống notebook)
    with open(RUNS_META_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    with open(RUNS_META_DIR / "mean_std.json", "w") as f:
        json.dump({"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}, f, indent=2)
    with open(RUNS_META_DIR / "img_size.json", "w") as f:
        json.dump({"img_size": IMG_SIZE}, f, indent=2)
    print("✅ Saved runs_meta/")

    # 2.3. Model + criterion + optimizer
    model = build_model(
        num_classes=num_classes,
        dropout=DROPOUT,
        pretrained=PRETRAINED,
        freeze_backbone=FREEZE_BACKBONE,
        device=device,
    )

    criterion = build_criterion_from_counts(
        class_counts,
        gamma=GAMMA,
        smooth=SMOOTH,
        device=device,
    )

    optimizer, scheduler = build_optim_sched(
        model,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        epochs=EPOCHS,
    )

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # 2.4. Training loop
    history = []
    best_val_acc = 0.0
    best_ckpt_name = "mtl_effb0_best.pt"

    # Chuẩn bị CSV
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")

        # ---- Train ----
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        n_train = 0

        pbar = tqdm(
            loaders["train"],
            desc=f"[Train] {epoch}/{EPOCHS}",
            dynamic_ncols=True,
        )
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type):
                logits = model(imgs)
                loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = labels.size(0)
            n_train += bs
            train_loss += loss.item() * bs
            train_acc += accuracy_from_logits(logits, labels) * bs

            pbar.set_postfix(
                loss=train_loss / n_train,
                acc=train_acc / n_train,
                lr=optimizer.param_groups[0]["lr"],
            )

        train_loss /= n_train
        train_acc /= n_train

        # ---- Val ----
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_val = 0
        with torch.no_grad():
            pbar_val = tqdm(
                loaders["val"],
                desc=f"[Valid] {epoch}/{EPOCHS}",
                dynamic_ncols=True,
            )
            for imgs, labels in pbar_val:
                imgs = imgs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type):
                    logits = model(imgs)
                    loss = criterion(logits, labels)

                bs = labels.size(0)
                n_val += bs
                val_loss += loss.item() * bs
                val_acc += accuracy_from_logits(logits, labels) * bs

                pbar_val.set_postfix(
                    loss=val_loss / n_val,
                    acc=val_acc / n_val,
                )

        val_loss /= n_val
        val_acc /= n_val

        scheduler.step()

        print(
            f"Epoch {epoch:03d}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        # Lưu CSV + history
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

        # Lưu best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = CKPT_DIR / best_ckpt_name
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "class_names": class_names,
                },
                ckpt_path,
            )
            print(f"💾 New BEST (val_acc={best_val_acc:.4f}) → {ckpt_path}")

    # Save history JSON
    with open(HISTORY_JSON, "w") as f:
        json.dump(history, f, indent=2)
    print("✅ Training done. History saved at:", CSV_PATH, "and", HISTORY_JSON)


if __name__ == "__main__":
    main()
