import sys
import json
import csv
import time
import random
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import torch
from tqdm.auto import tqdm

# ===== Import model & utils từ models/efficientnet_b0.py =====
ROOT_DIR = Path(__file__).resolve().parent  # thư mục Jupyter/
sys.path.append(str(ROOT_DIR / "models"))

from efficientnet_b0 import (  # type: ignore
    build_model,
    create_dataloaders,
    build_criterion_from_counts,
    build_optim_sched,
    accuracy_from_logits,
)

# ========================
# 1. CẤU HÌNH TOÀN CỤC
# ========================

# Đường dẫn dataset ĐÃ THÊM 1 CLASS MỚI (34 lớp)
DATA_DIR = Path("/media/mtl/DATA 6TB/AI DATASET/vietnamese-foods/Images")

# Run gốc 33 class dùng làm nền để copy trọng số
# !!! NHỚ ĐỔI TÊN NÀY THÀNH RUN 33 LỚP TỐT NHẤT CỦA ÔNG !!!
BASE_RUN_NAME = "MTL_TGFOOD_20251118_184127"   # ví dụ, sửa lại cho đúng
BASE_CKPT_NAME = "mtl_effb0_best.pt"

# Prefix tên run incremental
RUN_PREFIX = "MTL_TGFOOD_INC_"   # ví dụ: MTL_TGFOOD_INC_20251118_210001

IMG_SIZE = 224
EPOCHS = 30              # incremental nên train ít hơn
BATCH_SIZE = 64
NUM_WORKERS = 8

SEED = 1337
LR = 1e-4                 # LR nhỏ hơn finetune cho an toàn
WEIGHT_DECAY = 1e-4

GAMMA = 1.5               # focal gamma
SMOOTH = 0.05             # label smoothing
DROPOUT = 0.4
PRETRAINED = False        # incremental: ta copy từ model cũ, ko cần load ImageNet
FREEZE_BACKBONE = False   # sẽ tự freeze bằng tay cho linh hoạt

PATIENCE = 8              # early stopping


# Root cho các run
RUNS_ROOT = ROOT_DIR / "runs"

# Tạo RUN_NAME = prefix + timestamp
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"{RUN_PREFIX}{RUN_ID}"          # vd: MTL_TGFOOD_INC_20251118_210001

# Thư mục run cụ thể: runs/MTL_TGFOOD_INC_YYYYMMDD_HHMMSS
RUN_DIR = RUNS_ROOT / RUN_NAME
CKPT_DIR = RUN_DIR / "checkpoints"
IMAGES_DIR = ROOT_DIR / "images"

# Tạo thư mục (nếu chưa có)
RUN_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = RUN_DIR / "history.csv"
HISTORY_JSON = RUN_DIR / "history.json"
METRICS_CSV = RUN_DIR / "metrics.csv"

# SAVE META GLOBAL (dùng chung cho các model)
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


# ============================
# 3. HÀM XÂY MODEL INCREMENTAL
# ============================

def build_incremental_model(device: torch.device):
    """
    - Load checkpoint 33 lớp
    - Build model_old (33 lớp) & model_new (34 lớp)
    - Copy backbone + head (trừ fc cuối)
    - Copy trọng số fc cho các class cũ dựa trên tên class
    - Trả về: model_new, new_class_names, class_counts
    """
    # 3.1. Dataloader mới (dataset đã thêm class -> 34 lớp)
    loaders, new_class_names, class_counts = create_dataloaders(
        DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    num_new = len(new_class_names)
    print(f"New classes ({num_new}):", new_class_names)

    # 3.2. Load checkpoint cũ 33 lớp
    base_run_dir = RUNS_ROOT / BASE_RUN_NAME
    base_ckpt_path = base_run_dir / "checkpoints" / BASE_CKPT_NAME
    print("Loading base checkpoint from:", base_ckpt_path)

    ckpt = torch.load(base_ckpt_path, map_location=device)
    old_class_names = ckpt["class_names"]
    num_old = len(old_class_names)
    print(f"Old classes ({num_old}):", old_class_names)

    if num_new <= num_old:
        raise ValueError(
            f"Dataset mới ({num_new}) không lớn hơn số class cũ ({num_old}). "
            f"Incremental phải có thêm ít nhất 1 class mới."
        )

    # 3.3. Build model_old & load trọng số
    model_old = build_model(
        num_classes=num_old,
        dropout=DROPOUT,
        pretrained=False,      # không cần load lại ImageNet
        freeze_backbone=False,
        device=device,
    )
    model_old.load_state_dict(ckpt["model_state"])
    model_old.eval()

    # 3.4. Build model_new với số class mới
    model_new = build_model(
        num_classes=num_new,
        dropout=DROPOUT,
        pretrained=PRETRAINED,      # vẫn để False
        freeze_backbone=False,
        device=device,
    )

    # 3.5. Copy backbone & head (trừ fc cuối)
    with torch.no_grad():
        # backbone features
        model_new.backbone.features.load_state_dict(
            model_old.backbone.features.state_dict()
        )

        # các layer classifier trước fc cuối (0..len-2)
        for i in range(len(model_old.backbone.classifier) - 1):
            model_new.backbone.classifier[i].load_state_dict(
                model_old.backbone.classifier[i].state_dict()
            )

        # 3.6. Copy trọng số cho từng class cũ theo tên
        old_fc = model_old.backbone.classifier[-1]  # Linear out_old
        new_fc = model_new.backbone.classifier[-1]  # Linear out_new

        # Map theo tên class, tránh lệch thứ tự
        for new_idx, cls_name in enumerate(new_class_names):
            if cls_name in old_class_names:
                old_idx = old_class_names.index(cls_name)
                new_fc.weight[new_idx] = old_fc.weight[old_idx]
                new_fc.bias[new_idx]   = old_fc.bias[old_idx]
                print(f"Copied weights for existing class: {cls_name}")
            else:
                # Class mới -> giữ random init
                print(f"New class (random init): {cls_name}")

    # 3.7. Freeze backbone + các layer head trừ fc cuối
    for param in model_new.backbone.features.parameters():
        param.requires_grad = False

    for i in range(len(model_new.backbone.classifier) - 1):
        for p in model_new.backbone.classifier[i].parameters():
            p.requires_grad = False

    trainable_params = [n for n, p in model_new.named_parameters() if p.requires_grad]
    print("Trainable params count:", len(trainable_params))
    print("Trainable params:", trainable_params)

    return model_new, loaders, new_class_names, class_counts


# ============================
# 4. TRAIN LOOP INCREMENTAL
# ============================

def main():
    print(f"🔥 Incremental training run: {RUN_NAME}")
    print(f"📂 Run directory   : {RUN_DIR}")
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 4.1. Build model_new từ checkpoint cũ + dataloader mới
    model, loaders, class_names, class_counts = build_incremental_model(device)
    num_classes = len(class_names)

    # 4.2. SAVE runs_meta (class_names mới)
    with open(RUNS_META_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    with open(RUNS_META_DIR / "mean_std.json", "w") as f:
        json.dump(
            {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            f,
            indent=2,
        )
    with open(RUNS_META_DIR / "img_size.json", "w") as f:
        json.dump({"img_size": IMG_SIZE}, f, indent=2)
    print("✅ Saved runs_meta/ (updated with new classes)")

    # 4.3. Criterion + optimizer + scheduler
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

    history: List[Dict[str, Any]] = []
    best_val_acc = 0.0
    best_ckpt_name = "mtl_effb0_inc_best.pt"
    epochs_no_improve = 0

    # Chuẩn bị CSV: history.csv & metrics.csv
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    with open(METRICS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "lr", "time_s"]
        )

    # Lưu config cơ bản của run để sau dễ tra
    run_config = {
        "run_name": RUN_NAME,
        "run_id": RUN_ID,
        "base_run_name": BASE_RUN_NAME,
        "data_dir": str(DATA_DIR),
        "img_size": IMG_SIZE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "seed": SEED,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "gamma": GAMMA,
        "smooth": SMOOTH,
        "dropout": DROPOUT,
        "pretrained": PRETRAINED,
        "freeze_backbone": True,     # thực tế đã freeze features + head
        "patience": PATIENCE,
        "num_classes": num_classes,
    }
    with open(RUN_DIR / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    train_loader = loaders["train"]
    val_loader   = loaders["val"]

    # ========================
    # 5. TRAINING LOOP
    # ========================
    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")
        epoch_start = time.time()

        # ---- Train ----
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        n_train = 0

        pbar = tqdm(
            train_loader,
            desc=f"[Train INC] {epoch}/{EPOCHS}",
            dynamic_ncols=True,
        )
        for imgs, labels in pbar:
            imgs = (
                imgs.to(device, non_blocking=True)
                .to(memory_format=torch.channels_last)
            )
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

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_val = 0

        with torch.no_grad():
            pbar_val = tqdm(
                val_loader,
                desc=f"[Valid INC] {epoch}/{EPOCHS}",
                dynamic_ncols=True,
            )
            for imgs, labels in pbar_val:
                imgs = (
                    imgs.to(device, non_blocking=True)
                    .to(memory_format=torch.channels_last)
                )
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

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f} | "
            f"lr={current_lr:.8f}, time={epoch_time:.2f}s"
        )

        # Lưu vào history.json (list dict)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr,
                "time_s": epoch_time,
            }
        )

        # Lưu vào history.csv (đơn giản)
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

        # Lưu vào metrics.csv (đầy đủ)
        with open(METRICS_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [epoch, train_loss, val_loss, train_acc, val_acc, current_lr, epoch_time]
            )

        # Lưu best checkpoint + xử lý early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            ckpt_path = CKPT_DIR / best_ckpt_name
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "class_names": class_names,   # 34 lớp mới
                },
                ckpt_path,
            )
            print(f"💾 New BEST (val_acc={best_val_acc:.4f}) → {ckpt_path}")
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= PATIENCE:
            print(f"⛔ Early stopping: val_acc không cải thiện {PATIENCE} epoch liên tiếp")
            break

    # Save history.json
    with open(HISTORY_JSON, "w") as f:
        json.dump(history, f, indent=2)
    print("✅ Incremental training done.")
    print("   Run dir      :", RUN_DIR)
    print("   history.csv  :", CSV_PATH)
    print("   metrics.csv  :", METRICS_CSV)
    print("   history.json :", HISTORY_JSON)


if __name__ == "__main__":
    main()
