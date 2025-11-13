# webapps/app_trainmodel.py
# =========================================================
# Tab Streamlit: HUẤN LUYỆN MÔ HÌNH EfficientNet-B0 (PyTorch)
# - Dùng kiến trúc MTLEfficientNetB0 trong model/mtl_efficientnet_b0.py
# - Cấu hình: epochs, batch size, lr, patience, image size, ...
# - Hiển thị tiến trình train như Jupyter (log + progress bar)
# - Lưu CSV lịch sử train, checkpoint best/last, ảnh loss/accuracy
# - Cho phép tiếp tục train từ run cũ (resume)
# =========================================================

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple, Optional

import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Import kiến trúc model gốc ----
try:
    from model.mtl_efficientnet_b0 import MTLEfficientNetB0
except Exception:
    # Phòng trường hợp project không dùng package "model"
    from mtl_efficientnet_b0 import MTLEfficientNetB0  # type: ignore

# ========================
# CẤU HÌNH CHUNG
# ========================

# Nơi sinh ra các thư mục run
RUNS_ROOT = Path("Jupyter") / "runs"

# Chuẩn hoá theo ImageNet (đúng với EfficientNet B0 pretrained)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ========================
# HÀM PHỤ TRỢ
# ========================

@st.cache_resource(show_spinner=False)
def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Tạo transform cho train / val, cho phép chọn kích thước ảnh.
    """
    t_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    t_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return t_train, t_val


def _make_loaders(
    data_dir: Path,
    batch_size: int,
    img_size: int,
    val_ratio: float,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Hỗ trợ 2 layout:
        1) DATA_DIR/Train, DATA_DIR/Validate
        2) DATA_DIR/<class_name>/*.jpg  (tự tách train/val theo val_ratio)
    """
    t_train, t_val = _build_transforms(img_size)

    if (data_dir / "Train").exists() and (data_dir / "Validate").exists():
        ds_train = datasets.ImageFolder(data_dir / "Train", transform=t_train)
        ds_val = datasets.ImageFolder(data_dir / "Validate", transform=t_val)
        class_names = ds_train.classes
    else:
        full = datasets.ImageFolder(data_dir, transform=t_train)
        class_names = full.classes
        n = len(full)
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_val
        gen = torch.Generator().manual_seed(2025)
        ds_train, ds_val = torch.utils.data.random_split(
            full,
            [n_train, n_val],
            generator=gen,
        )
        # bảo đảm tập val không augment
        ds_val.dataset.transform = t_val  # type: ignore

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        loader_opts = dict(num_workers=2, pin_memory=True, persistent_workers=True)
    else:
        loader_opts = dict(num_workers=0)

    train_dl = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, drop_last=False, **loader_opts
    )
    val_dl = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False, drop_last=False, **loader_opts
    )
    return train_dl, val_dl, class_names


def _build_model(
    num_classes: int,
    freeze_backbone: bool,
    dropout: float,
    pretrained: bool,
    device: torch.device,
) -> nn.Module:
    """
    Dùng đúng kiến trúc MTLEfficientNetB0.
    """
    model = MTLEfficientNetB0(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout_rate=dropout,
    )
    model = model.to(device).to(memory_format=torch.channels_last)
    return model


def _find_latest_run() -> Optional[Path]:
    """
    Tìm run gần nhất để gợi ý khi chọn 'Tiếp tục huấn luyện'.
    """
    if not RUNS_ROOT.exists():
        return None
    candidates = [
        p for p in RUNS_ROOT.iterdir()
        if p.is_dir() and p.name.startswith("mtl_efficientnet_b0_")
    ]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def _load_prev_history(run_dir: Path) -> Tuple[pd.DataFrame, int]:
    """
    Đọc train_log.csv nếu có để biết đã train tới epoch nào.
    Trả về (df, last_epoch).
    """
    csv_path = run_dir / "train_log.csv"
    if not csv_path.exists():
        return pd.DataFrame(), 0
    df = pd.read_csv(csv_path)
    last_ep = int(df["epoch"].max())
    return df, last_ep


# ========================
# HÀM VẼ BIỂU ĐỒ
# ========================

def _plot_loss_acc(df: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    sns.lineplot(x="epoch", y="train_loss", data=df, label="Train", ax=ax[0])
    sns.lineplot(x="epoch", y="val_loss", data=df, label="Validate", ax=ax[0])
    ax[0].set_title("Loss theo Epoch")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")

    sns.lineplot(x="epoch", y="train_acc", data=df, label="Train", ax=ax[1])
    sns.lineplot(x="epoch", y="val_acc", data=df, label="Validate", ax=ax[1])
    ax[1].set_title("Accuracy theo Epoch")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ========================
# MAIN UI ENTRY
# ========================

def render_train_tab() -> None:
    st.header("Huấn luyện mô hình")
    st.caption("PyTorch • MTLEfficientNetB0 • Mixed Precision nếu dùng CUDA")

    # ------------- DATA_DIR -------------
    default_dir = st.session_state.get(
        "DATA_DIR",
        "/media/mtl/DATA 6TB/AI DATASET/vietnamese-foods/Images",
    )
    data_dir_str = st.text_input(
        "📁 DATA_DIR",
        value=default_dir,
        help="Thư mục chứa ảnh. Hỗ trợ dạng `Train/Validate/Test` hoặc tất cả ảnh chung trong một thư mục.",
    )
    data_dir = Path(data_dir_str)
    st.session_state["DATA_DIR"] = data_dir_str

    if not data_dir.exists():
        st.warning("⚠️ Không tìm thấy DATA_DIR. Hãy kiểm tra lại đường dẫn.")
        return

    # ------------- CHẾ ĐỘ TRAIN -------------
    mode = st.radio(
        "Chế độ huấn luyện",
        ["Huấn luyện mới", "Tiếp tục từ run trước"],
        horizontal=True,
    )

    # ------------- CẤU HÌNH CHUNG -------------
    col1, col2, col3, col4 = st.columns(4)
    epochs = int(col1.number_input("Epochs (tổng)", 1, 1000, value=100, step=1))
    batch_size = int(col2.number_input("Batch size", 1, 512, value=64, step=1))
    lr_str = col3.selectbox("Learning Rate", ["3e-4", "5e-4", "1e-3", "2e-3"], index=2)
    lr = float(lr_str)
    patience = int(col4.number_input("Early stopping (patience)", 1, 100, value=5, step=1))

    col5, col6, col7 = st.columns(3)
    img_size = int(col5.selectbox("Kích thước ảnh đầu vào", [160, 192, 224, 256, 288], index=2))
    val_ratio = float(col6.slider("Tỉ lệ Validate (nếu không có folder Validate)", 0.05, 0.3, 0.1, 0.01))
    num_workers = int(col7.number_input("num_workers (DataLoader)", 0, 8, value=2))

    freeze_backbone = st.checkbox("🔒 Freeze backbone EfficientNet", value=True)
    col8, col9 = st.columns(2)
    dropout = float(col8.slider("Dropout head classifier", 0.1, 0.7, 0.4, 0.05))
    pretrained = bool(col9.checkbox("Sử dụng weight pretrained ImageNet", value=True))

    # ------------- LOAD DATASET -------------
    try:
        # tạm thời dùng num_workers từ UI bằng cách monkey patch DataLoader opts
        torch_dataloader_workers_backup = torch.utils.data.DataLoader
        # (chỗ này chỉ để nhắc tới num_workers cho người dùng, thực tế _make_loaders đã set hợp lý
        #  theo GPU/CPU; nếu muốn fix cứng thì có thể sửa _make_loaders.)
        train_dl, val_dl, class_names = _make_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            img_size=img_size,
            val_ratio=val_ratio,
        )
        st.success(f"✅ Dataset OK — Train: {len(train_dl.dataset):,} • Validate: {len(val_dl.dataset):,}")
        st.caption(f"Số lớp: **{len(class_names)}**")
    except Exception as e:
        st.error("❌ Không thể load dataset.")
        st.exception(e)
        return

    # ------------- CHUẨN BỊ RUN_DIR / RESUME -------------
    run_dir: Optional[Path] = None
    prev_df: Optional[pd.DataFrame] = None
    start_epoch = 1

    if mode == "Tiếp tục từ run trước":
        latest = _find_latest_run()
        default_run = str(latest) if latest else ""
        run_dir_str = st.text_input(
            "Run folder để tiếp tục huấn luyện",
            value=default_run,
            help="Thư mục đã tạo khi train trước, ví dụ: `Jupyter/runs/mtl_efficientnet_b0_20251113-001830`.",
        )
        if run_dir_str.strip():
            run_dir = Path(run_dir_str.strip())
            if run_dir.exists():
                prev_df, last_ep = _load_prev_history(run_dir)
                if last_ep > 0:
                    start_epoch = last_ep + 1
                    st.info(f"📂 Đã tìm thấy lịch sử train đến epoch {last_ep}. "
                            f"Sẽ tiếp tục từ epoch {start_epoch}.")
            else:
                st.warning("⚠️ Run folder không tồn tại, sẽ tạo run mới.")
                run_dir = None

    # ------------- NÚT TRAIN -------------
    if st.button("🚀 Bắt đầu huấn luyện", use_container_width=True, key="btn_start_train"):
        device = _get_device()
        use_gpu = device.type == "cuda"
        torch.backends.cudnn.benchmark = use_gpu

        status = st.status("Đang khởi tạo…", expanded=True)
        prog = st.progress(0, text="Khởi tạo…")
        log_box = st.empty()

        def log(msg: str) -> None:
            ts = time.strftime("%H:%M:%S")
            line = f"[{ts}] {msg}"
            hist = st.session_state.get("_TRAIN_LOG", [])
            hist.append(line)
            st.session_state["_TRAIN_LOG"] = hist[-300:]
            log_box.code("\n".join(st.session_state["_TRAIN_LOG"]), language="bash")

        try:
            # ----- Tạo / dùng lại run_dir -----
            status.update(label="Chuẩn bị thư mục run/checkpoints/images…", state="running")
            if run_dir is None:
                now = time.strftime("%Y%m%d-%H%M%S")
                run_dir = RUNS_ROOT / f"mtl_efficientnet_b0_{now}"
            ckpt_dir = run_dir / "checkpoints"
            img_dir = run_dir / "images"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            img_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "classes.txt").write_text("\n".join(class_names), encoding="utf-8")

            st.session_state["TRAIN_RUN_DIR"] = str(run_dir)
            log(f"Tạo / sử dụng run_dir: {run_dir}")

            # ----- Model + Optim -----
            status.update(label="Khởi tạo mô hình & optimizer…", state="running")
            model = _build_model(
                num_classes=len(class_names),
                freeze_backbone=freeze_backbone,
                dropout=dropout,
                pretrained=pretrained,
                device=device,
            )
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
            scaler = torch.cuda.amp.GradScaler(enabled=use_gpu)

            # Nếu resume: load checkpoint last
            if mode == "Tiếp tục từ run trước":
                last_ckpt = ckpt_dir / "mtl_efficientnet_b0_last.pt"
                if last_ckpt.exists():
                    model.load_state_dict(torch.load(last_ckpt, map_location=device))
                    log(f"🔁 Đã load checkpoint: {last_ckpt}")
                else:
                    log("⚠️ Không tìm thấy last checkpoint, train từ đầu.")

            log(f"Thiết bị: {device} | Freeze backbone: {freeze_backbone} | "
                f"Dropout: {dropout:.2f} | LR={lr:g} | img_size={img_size}")

            # ===== Hàm train 1 epoch =====
            def _run_epoch(dataloader: DataLoader, mode: str):
                is_train = (mode == "train")
                model.train(is_train)

                total_loss, correct, total = 0.0, 0, 0
                n_img = 0
                t0 = time.time()

                for step, (imgs, labels) in enumerate(dataloader, 1):
                    imgs = imgs.to(device).to(memory_format=torch.channels_last, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    if is_train:
                        optimizer.zero_grad(set_to_none=True)

                    with torch.cuda.amp.autocast(enabled=use_gpu):
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)

                    if is_train:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                    total_loss += loss.item() * imgs.size(0)
                    preds = outputs.argmax(1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    n_img += imgs.size(0)

                sec = max(time.time() - t0, 1e-6)
                imgs_per_s = n_img / sec
                avg_loss = total_loss / max(total, 1)
                acc = correct / max(total, 1)
                return avg_loss, acc, sec, imgs_per_s

            # ===== VÒNG LẶP EPOCH =====
            best_val_acc = -1.0
            no_improve = 0

            # Nếu resume: dùng lại lịch sử cũ
            hist_rows: List[dict] = []
            if prev_df is not None and not prev_df.empty:
                hist_rows.extend(prev_df.to_dict(orient="records"))
                best_val_acc = float(prev_df["val_acc"].max())

            total_epochs = epochs
            for ep in range(start_epoch, total_epochs + 1):
                # TRAIN
                status.update(label=f"Epoch {ep}/{total_epochs} — Train…", state="running")
                tr_loss, tr_acc, tr_sec, tr_ips = _run_epoch(train_dl, "train")

                # VAL
                status.update(label=f"Epoch {ep}/{total_epochs} — Validate…", state="running")
                val_loss, val_acc, val_sec, val_ips = _run_epoch(val_dl, "val")
                scheduler.step()

                # Log row
                row = {
                    "epoch": ep,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_loss": tr_loss,
                    "val_loss": val_loss,
                    "train_acc": tr_acc,
                    "val_acc": val_acc,
                    "train_sec": tr_sec,
                    "val_sec": val_sec,
                    "train_imgs_per_s": tr_ips,
                    "val_imgs_per_s": val_ips,
                    "img_size": img_size,
                    "batch_size": batch_size,
                }
                hist_rows.append(row)

                # ETA cho phần còn lại
                eta = (tr_sec + val_sec) * max(0, (total_epochs - ep))
                log(
                    f"Epoch {ep:03d}: "
                    f"train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | "
                    f"train_acc={tr_acc*100:.2f}% | val_acc={val_acc*100:.2f}% | "
                    f"speed={tr_ips:.1f}/{val_ips:.1f} img/s (train/val) | "
                    f"ETA~{eta/60:.1f} phút"
                )

                # Lưu checkpoint
                last_ckpt = ckpt_dir / "mtl_efficientnet_b0_last.pt"
                torch.save(model.state_dict(), last_ckpt)
                if val_acc > best_val_acc + 1e-6:
                    best_val_acc = val_acc
                    best_ckpt = ckpt_dir / "mtl_efficientnet_b0_best.pt"
                    torch.save(model.state_dict(), best_ckpt)
                    no_improve = 0
                    log(f"⭐ Cải thiện! val_acc={best_val_acc*100:.2f}% -> lưu best.")
                else:
                    no_improve += 1

                # Progress bar
                prog.progress(
                    ep / float(total_epochs),
                    text=(
                        f"Epoch {ep}/{total_epochs} • "
                        f"Train {tr_acc*100:.1f}% • Val {val_acc*100:.1f}% • "
                        f"Speed {tr_ips:.0f}/{val_ips:.0f} img/s • "
                        f"Patience {no_improve}/{patience}"
                    ),
                )

                # Early stopping
                if no_improve >= patience:
                    log(f"⏹️ Early stopping tại epoch {ep} (không cải thiện {patience} lần liên tiếp).")
                    st.info(f"⏹️ Early stopping tại epoch {ep}.")
                    break

            # ===== LƯU CSV + VẼ BIỂU ĐỒ =====
            status.update(label="Ghi train_log.csv & vẽ biểu đồ Loss/Accuracy…", state="running")
            df = pd.DataFrame(hist_rows)
            csv_path = run_dir / "train_log.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8")
            log(f"📄 Đã lưu lịch sử train: {csv_path}")

            img_loss_acc_path = img_dir / "loss_acc.png"
            _plot_loss_acc(df, img_loss_acc_path)
            log(f"🖼️ Đã lưu biểu đồ: {img_loss_acc_path}")

            status.update(label="Hoàn tất huấn luyện", state="complete")
            prog.empty()
            st.success("🎉 Huấn luyện hoàn tất!")
            st.caption(f"📂 Run folder: `{run_dir}`")
            st.session_state["NEED_RELOAD_CKPTS"] = True  # để tab đánh giá model biết reload checkpoint

        except RuntimeError as e:
            prog.empty()
            status.update(label="Lỗi trong quá trình huấn luyện", state="error")
            log(f"❌ RuntimeError: {e}")
            if "CUDA out of memory" in str(e):
                st.error("💥 CUDA out of memory — hãy giảm Batch size hoặc kích thước ảnh.")
            st.exception(e)
        except Exception as e:
            prog.empty()
            status.update(label="Lỗi trong quá trình huấn luyện", state="error")
            log(f"❌ Exception: {e}")
            st.exception(e)
