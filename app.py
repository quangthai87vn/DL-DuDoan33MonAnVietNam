# app.py
# Streamlit inference cho miniVGG (PyTorch) với 30 món ăn VN

import io
import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import streamlit as st

# -----------------------------
# 1) CẤU HÌNH
# -----------------------------
# Đường dẫn mặc định tới checkpoint .pt (sửa lại nếu cần)
DEFAULT_CKPT = "checkpoints/classification_best.pt"

# Mapping 30 lớp THEO THỨ TỰ TRAIN (index 24 = Gỏi cuốn)
CLASS_NAMES = [
    "Bánh bèo", "Bánh bột lọc", "Bánh căn", "Bánh canh", "Bánh chưng",
    "Bánh cuốn", "Bánh đúc", "Bánh giò", "Bánh khọt", "Bánh mì",
    "Bánh pía", "Bánh tét", "Bánh tráng nướng", "Bánh xèo",
    "Bún bò Huế", "Bún đậu mắm tôm", "Bún mắm", "Bún riêu",
    "Bún thịt nướng", "Cá kho tộ", "Canh chua", "Cao lầu",
    "Cháo lòng", "Cơm tấm", "Gỏi cuốn", "Hủ tiếu",
    "Mì Quảng", "Nem chua", "Phở", "Xôi xéo"
]

# Tiền xử lý PHẢI KHỚP lúc train (224x224 cho miniVGG của bạn)
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

IMG_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"]


# -----------------------------
# 2) LOAD MODEL (cache)
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_model(ckpt_path: str):
    # để import được model.cnn miniVGG khi chạy từ mọi chỗ
    import sys
    root = Path(".").resolve()
    # nếu không tìm thấy folder model ở cwd, thử bò ngược vài cấp
    for _ in range(6):
        if (root / "model").exists():
            break
        root = root.parent
    sys.path.insert(0, str(root))

    from model.cnn import miniVGG  # <- đúng kiến trúc đã train

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = miniVGG().to(device).eval()

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
    model.load_state_dict(state_dict)
    return model, device


# -----------------------------
# 3) DỰ ĐOÁN
# -----------------------------
@torch.no_grad()
def predict_image(model: torch.nn.Module, device: str, image: Image.Image, topk: int = 3) -> Tuple[List[int], List[float]]:
    x = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)  # (1,C,H,W)
    logits = model(x)                                           # (1, num_classes)
    probs = F.softmax(logits, dim=1).squeeze(0)                 # (num_classes,)
    k = min(topk, probs.numel())
    vals, idxs = torch.topk(probs, k)
    return idxs.tolist(), vals.tolist()


# -----------------------------
# 4) UI
# -----------------------------
st.set_page_config(page_title="DỰ ĐOÁN MÓN ĂN VN", page_icon="🍜", layout="wide")
st.title("🍜 DỰ ĐOÁN MÓN ĂN VN – BÙI QUANG THÁI")

with st.sidebar:
    st.subheader("⚙️ Cấu hình")
    ckpt_path = st.text_input("Đường dẫn checkpoint (.pt)", DEFAULT_CKPT)
    topk = st.slider("Top-K hiển thị", 1, 5, 3)
    cols = st.slider("Số cột hiển thị", 1, 5, 3)
    show_prob = st.toggle("Hiện % xác suất", value=True)
    st.caption("Nếu lỗi ‘không thấy model’, chạy app từ thư mục gốc dự án hoặc chỉnh lại đường dẫn ở trên.")

# Tải model
if not Path(ckpt_path).exists():
    st.error(f"Không tìm thấy checkpoint: {ckpt_path}")
    st.stop()

model, device = load_model(ckpt_path)
st.success(f"Model đã sẵn sàng trên **{device.upper()}** • checkpoint: `{ckpt_path}`")

# Uploader: hỗ trợ nhiều ảnh
files = st.file_uploader("Chọn 1 hoặc nhiều ảnh", type=IMG_TYPES, accept_multiple_files=True)

# Option: kéo-thả cả thư mục đã nén .zip
zip_file = st.file_uploader("Hoặc chọn 1 file .zip chứa ảnh", type=["zip"], accept_multiple_files=False)

# Gom tất cả ảnh cần dự đoán
images_to_run = []

if files:
    for f in files:
        try:
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
            images_to_run.append((f.name, img))
        except Exception as e:
            st.warning(f"Ảnh lỗi `{f.name}`: {e}")

# Giải nén zip (nếu có)
if zip_file is not None:
    import zipfile, tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        zf = zipfile.ZipFile(io.BytesIO(zip_file.read()))
        zf.extractall(tmpdir)
        for p in Path(tmpdir).rglob("*"):
            if p.suffix.lower().strip(".") in IMG_TYPES:
                try:
                    img = Image.open(p).convert("RGB")
                    images_to_run.append((str(p.name), img))
                except Exception as e:
                    st.warning(f"Ảnh lỗi `{p}`: {e}")

# Nếu không có ảnh → demo hướng dẫn
if not images_to_run:
    st.info("👉 Hãy kéo-thả ảnh (hoặc .zip) vào khung trên để dự đoán.")
    st.stop()

# Hiển thị theo grid
n = len(images_to_run)
rows = (n + cols - 1) // cols
idx = 0
for _ in range(rows):
    c = st.columns(cols)
    for j in range(cols):
        if idx >= n:
            break
        name, pil_img = images_to_run[idx]
        with c[j]:
            # dự đoán
            try:
                top_idx, top_prob = predict_image(model, device, pil_img, topk=topk)
                # render
                # st.image(pil_img, use_container_width=True)
                try:
                    st.image(pil_img, use_container_width=True)   # Streamlit mới (>= ~1.25)
                except TypeError:
                    st.image(pil_img, use_column_width=True)      # Streamlit cũ
                if show_prob:
                    lines = [f"**{k}. {CLASS_NAMES[k] if k < len(CLASS_NAMES) else 'class_'+str(k)}** — {p*100:.1f}%"
                             for k, p in zip(top_idx, top_prob)]
                else:
                    lines = [f"**{k}. {CLASS_NAMES[k] if k < len(CLASS_NAMES) else 'class_'+str(k)}**"
                             for k in top_idx]
                st.markdown(f"**{name}**")
                st.markdown(" • ".join(lines))





            except Exception as e:
                st.error(f"Lỗi dự đoán `{name}`: {e}")
        idx += 1

