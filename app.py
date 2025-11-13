# app.py — Streamlit inference cho nhiều model, đọc checkpoint từ thư mục Models/
import io

from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import streamlit as st

# ====== các kiến trúc bạn đang dùng (đảm bảo các file này có sẵn) ======
from model.mtl_cnn import mtl_cnn_v1
from model.mobilenet_v4 import CustomMobileNetV4
from model.efficientnet_b0 import CustomEfficientNetB0


from model.mtl_efficientnet_b0 import mtl_efficientnet_b0_model

# đặt ở đầu file sau imports
from inspect import signature

def show_img_responsive(img):
    # Tự dò chữ ký hàm st.image để chọn tham số đúng với version
    params = signature(st.image).parameters
    if "use_container_width" in params:
        st.image(img, use_container_width=True)
    elif "use_column_width" in params:
        st.image(img, use_column_width=True)
    else:
        st.image(img)



# ====== cấu hình chung ======
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

# Bạn train không normalize -> giữ nguyên
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ====== page & CSS ======
st.set_page_config(page_title="DỰ ĐOÁN MÓN ĂN VN", page_icon="🍜", layout="wide")

# --- CSS responsive & hide file list ---
st.markdown("""
<style>
/* Ẩn danh sách file đã chọn của st.file_uploader */
[data-testid="stFileUploader"] .uploadedFile { display: none !important; }
[data-testid="stFileUploader"] .uploadedFileList { display: none !important; }
/* Giấu nhãn label nếu muốn gọn hơn (tuỳ chọn) */
/* [data-testid="stFileUploader"] label { display:none !important; } */

/* Layout 2 uploader: desktop = 2 cột, mobile = xếp dọc */
.upload-row { display: flex; gap: 16px; }
.upload-col { flex: 1 1 0; }

/* Màn hình hẹp (≤ 768px) -> xếp dọc */
@media (max-width: 768px) {
  .upload-row { flex-direction: column; }
}
</style>
""", unsafe_allow_html=True)




st.title("🍜 DỰ ĐOÁN MÓN ĂN VN – BÙI QUANG THÁI - 24752551")

# ====== tiện ích ======
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

    if "mtl_efficientnet_b0" in n:
        return "mtl_efficientnet_b0"
    return "mtl_cnn"

def _read_state_dict(ckpt_path: Path, device: str):
    obj = torch.load(str(ckpt_path), map_location=device)
    if isinstance(obj, dict) and "net" in obj:
        state = obj["net"]
    elif isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    else:
        state = obj
    # bỏ 'module.' nếu dùng DataParallel
    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v
    return new_state

def build_model_by_arch(arch: str, num_classes: int) -> nn.Module:
    if arch == "mtl_cnn":
        return mtl_cnn_v1(num_classes=num_classes)
    if arch == "mobilenetv4":
        return CustomMobileNetV4(num_classes=num_classes, pretrained=False)
    if arch == "efficientnet_b0":
        return CustomEfficientNetB0(num_classes=num_classes, pretrained=False)
    if arch == "mtl_efficientnet_b0":
        return mtl_efficientnet_b0_model(num_classes=num_classes, pretrained=True, freeze_backbone=False)






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

# ====== sidebar ======
with st.sidebar:
    st.subheader("⚙️ Cấu hình")
    ckpt_files = list_checkpoints(MODELS_DIR)
    if not ckpt_files:
        st.error(f"Không tìm thấy checkpoint trong `{MODELS_DIR}/`.")
        st.stop()

    # chỉ hiển thị tên file, không kèm 'Models\\'
    name_to_path = {p.name: str(p) for p in ckpt_files}
    sel_name = st.selectbox("Chọn file model (.mtl/.pt)", list(name_to_path.keys()), index=0)
    ckpt_path = name_to_path[sel_name]

    detected_arch = guess_arch(sel_name)
    st.caption(f"Kiến trúc suy luận: **{detected_arch}** (đổi tên file nếu muốn nhận diện khác)")
    topk = st.slider("Top-K hiển thị", 1, 5, 3)
    cols = st.slider("Số cột hiển thị", 1, 5, 3)
    show_prob = st.toggle("Hiện % xác suất", value=True)

# ====== load model ======
model, device = load_model_cached(ckpt_path, detected_arch, NUM_CLASSES)
st.success(f"Model `{detected_arch}` đã sẵn sàng trên **{device.upper()}** • checkpoint: `{sel_name}`")

# ====== Upload UI (responsive + không show list file) ======
def build_upload_ui() -> list[tuple[str, Image.Image]]:
    """
    Trả về danh sách [(tên, PIL Image)] đã đọc từ 2 uploader:
    - Trái: chọn nhiều ảnh
    - Phải: 1 file .zip chứa ảnh
    Responsive: 2 cột desktop, 1 cột mobile.
    Không hiển thị danh sách file — chỉ render ảnh ngay sau khi đọc.
    """
    import zipfile, tempfile

    images: list[tuple[str, Image.Image]] = []

    # Khối wrapper để CSS nhận diện
    with st.container():
        st.markdown('<div class="upload-row">', unsafe_allow_html=True)

        # --- uploader trái: nhiều ảnh ---
        st.markdown('<div class="upload-col">', unsafe_allow_html=True)
        up_imgs = st.file_uploader(
            "Chọn 1 hoặc nhiều ảnh",
            type=IMG_TYPES,
            accept_multiple_files=True,
            label_visibility="collapsed",   # ẩn label
            key="uploader_images"
        )
        if up_imgs:
            for f in up_imgs:
                try:
                    images.append((f.name, Image.open(io.BytesIO(f.read())).convert("RGB")))
                except Exception as e:
                    st.warning(f"Ảnh lỗi `{f.name}`: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

        # --- uploader phải: 1 file zip ---
        st.markdown('<div class="upload-col">', unsafe_allow_html=True)
        up_zip = st.file_uploader(
            "Hoặc chọn 1 file .zip chứa ảnh",
            type=["zip"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            key="uploader_zip"
        )
        if up_zip is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    zf = zipfile.ZipFile(io.BytesIO(up_zip.read()))
                    zf.extractall(tmpdir)
                    for p in Path(tmpdir).rglob("*"):
                        if p.suffix.lower().strip(".") in IMG_TYPES:
                            try:
                                images.append((p.name, Image.open(p).convert("RGB")))
                            except Exception as e:
                                st.warning(f"Ảnh lỗi `{p}`: {e}")
                except Exception as e:
                    st.error(f"Không đọc được file .zip: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    return images



images_to_run = build_upload_ui()

if not images_to_run:
    st.info("👉 Kéo-thả ảnh (hoặc .zip) vào hai ô bên trên để dự đoán.")
    st.stop()




# ====== hiển thị kết quả ======
n = len(images_to_run)
rows = (n + cols - 1) // cols
idx = 0
for _ in range(rows):
    c = st.columns(cols)
    for j in range(cols):
        if idx >= n: break
        name, pil_img = images_to_run[idx]
        with c[j]:


            try:
                ids, probs = predict_image(model, device, pil_img, topk=topk)
                #st.image(pil_img, use_container_width=True)
                show_img_responsive(pil_img)
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


