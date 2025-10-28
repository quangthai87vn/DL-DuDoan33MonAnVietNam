# app.py — Streamlit: chọn checkpoint trong thư mục Models/ rồi dự đoán ảnh
import io


from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import streamlit as st

# === các kiến trúc bạn có ===
from model.mtl_cnn import mtl_cnn_v1
from model.mobilenet_v4 import CustomMobileNetV4
from model.efficientnet_b0 import CustomEfficientNetB0

MODELS_DIR = Path("Models")
IMG_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"]

CLASS_NAMES = [
    "Bánh bèo","Bánh bột lọc","Bánh căn","Bánh canh","Bánh chưng",
    "Bánh cuốn","Bánh đúc","Bánh giò","Bánh khọt","Bánh mì",
    "Bánh pía","Bánh tét","Bánh tráng nướng","Bánh xèo",
    "Bún bò Huế","Bún đậu mắm tôm","Bún mắm","Bún riêu",
    "Bún thịt nướng","Cá kho tộ","Canh chua","Cao lầu",
    "Cháo lòng","Cơm tấm","Gỏi cuốn","Hủ tiếu",
    "Mì Quảng","Nem chua","Phở","Xôi xéo",
    "banh_da_lon","banh_tieu","banh_trung_thu"
]
NUM_CLASSES = len(CLASS_NAMES)

# bạn train không normalize → giữ nguyên
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def list_checkpoints(models_dir: Path) -> list[Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    return sorted([p for p in models_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in (".mtl", ".pt")])

def guess_arch(ckpt_name: str) -> str:
    n = ckpt_name.lower()
    if "mobilenetv4" in n or "mobilev4" in n:
        return "mobilenetv4"
    if "efficientnet" in n or "_b0" in n:
        return "efficientnet_b0"
    return "mtl_cnn"

def _read_state_dict(ckpt_path: Path, device: str):
    obj = torch.load(str(ckpt_path), map_location=device)
    if isinstance(obj, dict) and "net" in obj:
        state = obj["net"]
    elif isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    else:
        state = obj
    new_state = {}
    for k, v in state.items():
        new_state[k[7:]] = v if k.startswith("module.") else v  # strip "module."
    # nếu k không bắt đầu "module.", dòng trên vẫn giữ nguyên v
    return { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }

def build_model_by_arch(arch: str, num_classes: int) -> nn.Module:
    if arch == "mtl_cnn":
        return mtl_cnn_v1(num_classes=num_classes)
    if arch == "mobilenetv4":
        return CustomMobileNetV4(num_classes=num_classes, pretrained=False)
    if arch == "efficientnet_b0":
        return CustomEfficientNetB0(num_classes=num_classes, pretrained=False)
    raise ValueError(f"Unknown arch: {arch}")

@st.cache_resource(show_spinner=False)
def load_model_cached(ckpt_path: str, arch: str, num_classes: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model_by_arch(arch, num_classes).to(device).eval()
    state = _read_state_dict(Path(ckpt_path), device)
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        missing, unexpected = model.load_state_dict(state, strict=False)
        warn = []
        if missing:    warn.append(f"missing: {len(missing)}")
        if unexpected: warn.append(f"unexpected: {len(unexpected)}")
        if warn: st.warning("Checkpoint không khớp hoàn toàn → dùng strict=False. " + " | ".join(warn))
    return model, device




@torch.no_grad()
def predict_image(model: nn.Module, device: str, image: Image.Image, topk: int = 3):
    x = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1).squeeze(0)
    k = min(topk, probs.numel())
    vals, idxs = torch.topk(probs, k)
    return idxs.tolist(), vals.tolist()

# ================= UI =================
st.set_page_config(page_title="DỰ ĐOÁN MÓN ĂN VN", page_icon="🍜", layout="wide")
st.title("🍜 DỰ ĐOÁN MÓN ĂN VN – BÙI QUANG THÁI - 24752551")

with st.sidebar:
    st.subheader("⚙️ Cấu hình")
    ckpt_files = list_checkpoints(MODELS_DIR)
    if not ckpt_files:
        st.error(f"Không tìm thấy checkpoint trong `{MODELS_DIR}/`.")
        st.stop()

    # HIỂN THỊ TÊN FILE NGẮN, KHÔNG CÓ 'Models\'
    name_to_path = {p.name: str(p) for p in ckpt_files}
    sel_name = st.selectbox("Chọn file model ", list(name_to_path.keys()), index=0)
    ckpt_path = name_to_path[sel_name]

    detected_arch = guess_arch(sel_name)
    st.caption(f"Kiến trúc suy luận: **{detected_arch}** (đổi tên file nếu muốn nhận diện khác)")
    topk = st.slider("Top-K hiển thị", 1, 5, 3)
    cols = st.slider("Số cột hiển thị", 1, 5, 3)
    show_prob = st.toggle("Hiện % xác suất", value=True)

# load model
model, device = load_model_cached(ckpt_path, detected_arch, NUM_CLASSES)
st.success(f"Model `{detected_arch}` đã sẵn sàng trên **{device.upper()}** • checkpoint: `{sel_name}`")

# upload ảnh
files = st.file_uploader("Chọn 1 hoặc nhiều ảnh", type=IMG_TYPES, accept_multiple_files=True)


zip_file = st.file_uploader("Hoặc chọn 1 file .zip chứa ảnh", type=["zip"], accept_multiple_files=False)

images_to_run: list[tuple[str, Image.Image]] = []
if files:
    for f in files:
        try:
            images_to_run.append((f.name, Image.open(io.BytesIO(f.read())).convert("RGB")))
        except Exception as e:
            st.warning(f"Ảnh lỗi `{f.name}`: {e}")



if zip_file is not None:
    import zipfile, tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        zf = zipfile.ZipFile(io.BytesIO(zip_file.read()))
        zf.extractall(tmpdir)
        for p in Path(tmpdir).rglob("*"):
            if p.suffix.lower().strip(".") in IMG_TYPES:
                try:
                    images_to_run.append((p.name, Image.open(p).convert("RGB")))
                except Exception as e:
                    st.warning(f"Ảnh lỗi `{p}`: {e}")

# chỉ dừng nếu SAU HAI KHỐI TRÊN vẫn chưa có ảnh
if not images_to_run:
    st.info("👉 Kéo-thả ảnh (hoặc .zip) vào khung trên để dự đoán.")
    st.stop()

# hiển thị kết quả — SỬA THỤT LỀ & TĂNG CHỈ SỐ idx ĐÚNG VÒNG LẶP
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


            try:
                ids, probs = predict_image(model, device, pil_img, topk=topk)
                st.image(pil_img, use_container_width=True)
                if show_prob:
                    lines = [f"**{CLASS_NAMES[k] if k < NUM_CLASSES else 'class_'+str(k)}** — {p*100:.1f}%"
                             for k, p in zip(ids, probs)]
                else:
                    lines = [f"**{CLASS_NAMES[k] if k < NUM_CLASSES else 'class_'+str(k)}**" for k in ids]
                st.markdown(f"**{name}**")
                st.markdown(" • ".join(lines))





            except Exception as e:


                st.error(f"Lỗi dự đoán `{name}`: {e}")
        idx += 1


