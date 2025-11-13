# app1.py — VN Foods Streamlit (EfficientNet-B0, chuẩn notebook)
import os, io, zipfile, warnings
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import streamlit as st

import torch
import torch.nn.functional as F
from torchvision import transforms

warnings.filterwarnings("ignore")

# ================== ĐƯỜNG DẪN & HYPER ==================
BASE = Path(__file__).resolve().parent
RUNS_DIR = (BASE / "Jupyter" / "runs") if (BASE / "Jupyter" / "runs").exists() else (BASE / "runs")
BEST_TOKEN = "best"   # chỉ tìm file có chữ 'best' trong tên

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMG_TYPES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# ============ IMPORT KIẾN TRÚC GIỐNG LÚC TRAIN ============
# đảm bảo có file: model/mtl_efficientnet_b0.py
from model.mtl_efficientnet_b0 import mtl_efficientnet_b0_model  # type: ignore


# ================== TIỆN ÍCH ==================
def list_best_ckpts(runs_dir: Path) -> List[Path]:
    """Tìm tất cả checkpoint có chữ 'best' trong runs/*/checkpoints/*, sort mới nhất trước."""
    if not runs_dir.exists():
        return []
    out = []
    for run in runs_dir.glob("*"):
        ck = run / "checkpoints"
        if ck.is_dir():
            for p in ck.glob("*"):
                if p.is_file() and any(p.name.lower().endswith(ext) for ext in (".pt",".pth",".mtl",".ckpt",".bin",".keras")):
                    if BEST_TOKEN in p.name.lower():
                        out.append(p)
    out.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return out


def build_transform(mode: str = "stretch"):
    """
    mode='stretch'   : Resize(H, W) (nhiều notebook dùng cách này)
    mode='keep_ratio': Resize(shorter=IMG_SIZE) + CenterCrop(IMG_SIZE)
    """
    if mode == "keep_ratio":
        return transforms.Compose([
            transforms.Resize(IMG_SIZE, antialias=True),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    # default: stretch
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def read_images(files) -> List[Tuple[str, Image.Image]]:
    out = []
    if not files: return out
    for f in files:
        try:
            img = Image.open(f).convert("RGB")
            out.append((f.name, img))
        except Exception:
            pass
    return out


def read_zip(file_bytes: bytes) -> List[Tuple[str, Image.Image]]:
    out = []
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
        for n in z.namelist():
            if Path(n).suffix.lower() in IMG_TYPES:
                with z.open(n) as f:
                    try:
                        img = Image.open(io.BytesIO(f.read())).convert("RGB")
                        out.append((Path(n).name, img))
                    except Exception:
                        pass
    return out


@torch.inference_mode()
def predict_images(model, device, imgs: List[Image.Image], tfm, topk: int = 3):
    batch = torch.stack([tfm(im) for im in imgs]).to(device).to(memory_format=torch.channels_last)
    with torch.amp.autocast(device_type='cuda' if device.type=='cuda' else 'cpu'):
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        conf, idx = torch.topk(probs, k=topk, dim=1)
    return conf.cpu().numpy(), idx.cpu().numpy()


# ================== SMART CHECKPOINT LOADER ==================
def _extract_state_and_classes(raw) -> tuple[dict, Optional[list]]:
    """Cố gắng bóc state_dict và class_names từ nhiều format khác nhau."""
    class_names = None
    if isinstance(raw, dict):
        for key in ("class_names", "classes", "labels"):
            if key in raw and isinstance(raw[key], (list, tuple)):
                class_names = list(raw[key])
                break
        for k in ("model_state", "state_dict", "model_state_dict", "model"):
            if k in raw and isinstance(raw[k], dict):
                return raw[k], class_names
        # có thể là state_dict thẳng
        return raw, class_names
    raise RuntimeError("Checkpoint format not supported")


def _strip_prefix(k: str, pref: str) -> str:
    return k[len(pref):] if k.startswith(pref) else k


def _try_map(sd_in: dict, transform: str) -> dict:
    """Các phép đổi key phổ biến để khớp kiến trúc."""
    if transform == "identity":
        return {k: v for k, v in sd_in.items()}
    if transform == "strip_module":
        return {_strip_prefix(k, "module."): v for k, v in sd_in.items()}
    if transform == "strip_1tok":
        return {(k.split(".", 1)[1] if "." in k else k): v for k, v in sd_in.items()}
    if transform == "strip_2tok":
        return {(".".join(k.split(".")[2:]) if k.count(".") >= 2 else k): v for k, v in sd_in.items()}
    if transform == "features_to_backbone":
        out = {}
        for k, v in sd_in.items():
            if k.startswith(("features.", "classifier.")):
                out["backbone." + k] = v
            else:
                out[k] = v
        return out
    return sd_in


def smart_load_weights(model, ckpt_path: Path, device):
    """Tải trọng số, tự map key cho tỉ lệ khớp cao nhất; trả về (class_names, hit_pct, missing, unexpected)."""
    raw = torch.load(ckpt_path, map_location=device)
    sd_raw, class_names = _extract_state_and_classes(raw)
    base = _try_map(sd_raw, "strip_module")

    transforms = ["identity", "features_to_backbone", "strip_1tok", "strip_2tok"]
    msd_keys = set(model.state_dict().keys())

    best_sd, best_hit = None, -1
    for t in transforms:
        cand = _try_map(base, t)
        hit = len(msd_keys.intersection(cand.keys()))
        if hit > best_hit:
            best_hit, best_sd = hit, cand

    missing, unexpected = model.load_state_dict(best_sd, strict=False)
    hit_pct = 100.0 * (len(msd_keys) - len(missing)) / max(1, len(msd_keys))
    return class_names, hit_pct, missing, unexpected


# ================== STREAMLIT UI ==================
st.set_page_config(page_title="VN Foods — EfficientNet-B0", layout="wide")
st.title("🍜 Dự đoán món ăn Việt — EfficientNet-B0 (chuẩn notebook)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.success(f"Thiết bị: **{device.type.upper()}**")

# Chọn checkpoint "best"
ckpts = list_best_ckpts(RUNS_DIR)
if not ckpts:
    st.error(f"Không tìm thấy file '*{BEST_TOKEN}*' trong '{RUNS_DIR}/**/checkpoints/'.")
    st.stop()

sel = st.sidebar.selectbox(
    "Checkpoint (best):",
    options=[str(p.relative_to(RUNS_DIR)) for p in ckpts],
    index=0,  # mặc định mới nhất
)
ckpt_path = RUNS_DIR / sel
st.sidebar.write(f"📁 `{ckpt_path}`")

# Resize mode giống notebook
resize_mode = st.sidebar.radio("Resize mode", ["stretch (Resize HxW)", "keep_ratio (Resize short + CenterCrop)"], index=0)
resize_mode_key = "stretch" if resize_mode.startswith("stretch") else "keep_ratio"
topk = st.sidebar.slider("Top-K", 1, 5, 3, 1)
cols = st.sidebar.slider("Số cột", 1, 6, 3, 1)
show_prob = st.sidebar.toggle("Hiện % xác suất", value=True)

# Load model + weights
# Fallback class names nếu ckpt không lưu
CLASS_NAMES = [
    "Bánh beo","Bánh bot loc","Bánh can","Bánh canh","Bánh chung","Bánh cuon",
    "Bánh duc","Bánh gio","Bánh khot","Bánh mi","Bánh pia","Bánh tet",
    "Bánh trang nuong","Bánh xeo","Bun bo Hue","Bun dau mam tom","Bun mam",
    "Bun rieu","Bun thit nuong","Ca kho to","Canh chua","Cao lau","Chao long",
    "Com tam","Goi cuon","Hu tieu","Mi quang","Nem chua","Pho","Xoi xeo",
    "banh_da_lon","banh_tieu","banh_trung_thu",
]

try:
    # tạo model tạm 33 lớp để load & lấy class_names từ ckpt nếu có
    tmp_model = mtl_efficientnet_b0_model(num_classes=len(CLASS_NAMES)).to(device)
    classes_from_ckpt, hit_pct_tmp, missing_tmp, unexpected_tmp = smart_load_weights(tmp_model, ckpt_path, device)
    if classes_from_ckpt and isinstance(classes_from_ckpt, list):
        CLASS_NAMES = list(classes_from_ckpt)

    # tạo model cuối đúng num_classes và load lại
    num_classes = len(CLASS_NAMES)
    model = mtl_efficientnet_b0_model(num_classes=num_classes).to(device)
    classes_from_ckpt, hit_pct, missing, unexpected = smart_load_weights(model, ckpt_path, device)
    model.eval()

    st.sidebar.info(f"Key match: {hit_pct:.1f}% • missing={len(missing)} • unexpected={len(unexpected)}")
    if hit_pct < 98:
        st.warning("⚠️ Trọng số khớp < 98%. Kiểm tra kiến trúc/ckpt (kết quả có thể sai).")
    else:
        st.success("Weights loaded ✓")
except Exception as e:
    st.error(f"Lỗi load checkpoint: {e}")
    st.stop()

# Upload ảnh
st.subheader("Tải ảnh đơn lẻ")
files = st.file_uploader(
    "Kéo-thả PNG/JPG/JPEG/BMP/TIFF/WEBP (chọn nhiều ảnh)",
    type=[t.replace(".","") for t in IMG_TYPES],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

st.subheader("Hoặc tải 1 file ZIP ảnh")
zipf = st.file_uploader("Kéo-thả ZIP", type=["zip"], label_visibility="collapsed")

images: List[Tuple[str, Image.Image]] = []
images += read_images(files)
if zipf is not None:
    images += read_zip(zipf.read())

if not images:
    st.info("👉 Kéo-thả ảnh hoặc ZIP vào ô trên để dự đoán.")
    st.stop()

# Infer
tfm = build_transform(resize_mode_key)
names = [n for n,_ in images]
imgs  = [im for _,im in images]
conf, idx = predict_images(model, device, imgs, tfm, topk=topk)

# Hiển thị
st.divider()
grid = st.columns(cols)
for i, (name, im) in enumerate(images):
    with grid[i % cols]:
        st.image(im, use_container_width=True)
        tops = [(CLASS_NAMES[idx[i,k]], float(conf[i,k])) for k in range(conf.shape[1])]
        if show_prob:
            st.write(" · ".join(f"**{lbl}** — {p*100:.1f}%" for lbl,p in tops))
        else:
            st.write(" · ".join(lbl for lbl,_ in tops))
        st.caption(name)
