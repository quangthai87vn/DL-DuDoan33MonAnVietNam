# Webapps/app.py
import sys
from pathlib import Path
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models

# ====== resolve project root (giữ đúng cấu trúc hiện hữu) ======
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# === import local model (giữ nguyên như app cũ) ===
from model.mtl_efficientnet_b0 import mtl_efficientnet_b0_model  # type: ignore

# === import các module UI ===
import app_datamining as dm
import app_validatemodel as vm

import app_predict as pr
import app_trainmodel as tr   # <--- NEW
import app_augment as ag
from app_trainmodel import render_train_tab  
TITLE = "Nhận diện 33 món ăn - Bùi Quang Thái - 24752551"

# ---------------- utils: runs & ckpt ----------------
@st.cache_data(show_spinner=False)
def detect_runs_dir() -> Path:
    # ưu tiên Jupyter/runs
    for p in (ROOT / "Jupyter" / "runs", ROOT / "runs"):
        if p.exists():
            return p
    return ROOT / "Jupyter" / "runs"

@st.cache_data(show_spinner=False)
def list_best_ckpts(runs_dir: Path):
    outs = []
    if runs_dir.exists():
        for run in runs_dir.glob("*"):
            ck = run / "checkpoints"
            if ck.is_dir():
                for p in ck.glob("*"):
                    n = p.name.lower()
                    if "best" in n and any(n.endswith(ext) for ext in (".pt", ".pth", ".ckpt", ".bin", ".mtl", ".keras")):
                        outs.append(p)
    outs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return outs

def _extract_state_and_classes(raw):
    names = None
    if isinstance(raw, dict):
        for k in ("class_names", "classes", "labels"):
            if k in raw and isinstance(raw[k], (list, tuple)):
                names = list(raw[k]); break
        for k in ("state_dict", "model_state", "model_state_dict", "model"):
            if k in raw and isinstance(raw[k], dict):
                return raw[k], names
        return raw, names
    return raw, None

def _strip_prefix(k: str, pref: str) -> str:
    return k[len(pref):] if k.startswith(pref) else k

def _map_keys(sd: dict, how: str) -> dict:
    if how == "identity": return {k: v for k, v in sd.items()}
    if how == "strip_module": return {_strip_prefix(k, "module."): v for k, v in sd.items()}
    if how == "strip_1tok": return {(k.split(".", 1)[1] if "." in k else k): v for k, v in sd.items()}
    if how == "strip_2tok": return {(".".join(k.split(".")[2:]) if k.count(".") >= 2 else k): v for k, v in sd.items()}
    if how == "features_to_backbone":
        out = {}
        for k, v in sd.items():
            out[("backbone." + k) if k.startswith(("features.", "classifier.")) else k] = v
        return out
    return sd

def smart_load_weights(model, ckpt_path: Path, device):
    import torch as T
    raw = T.load(ckpt_path, map_location=device)
    sd_raw, names = _extract_state_and_classes(raw)
    base = _map_keys(sd_raw, "strip_module")
    msd = set(model.state_dict().keys())
    best_hit = -1; best_sd = None
    for how in ["identity", "features_to_backbone", "strip_1tok", "strip_2tok"]:
        cand = _map_keys(base, how)
        hit = len(msd.intersection(cand.keys()))
        if hit > best_hit:
            best_hit = hit; best_sd = cand
    missing, unexpected = model.load_state_dict(best_sd, strict=False)
    hit = 100.0 * (len(msd) - len(missing)) / max(1, len(msd))
    return names, hit, missing, unexpected

# ---------------- session scaffolding ----------------
def ensure_session_scaffolding():
    if "GLOBAL_MODEL_CACHE" not in st.session_state:
        st.session_state["GLOBAL_MODEL_CACHE"] = {}
    if "GLOBAL_CLASSES_CACHE" not in st.session_state:
        st.session_state["GLOBAL_CLASSES_CACHE"] = {}
    if "RUNS_DIR" not in st.session_state:
        st.session_state["RUNS_DIR"] = str(detect_runs_dir())
    if "GLOBAL_SELECTED_CKPT" not in st.session_state:
        st.session_state["GLOBAL_SELECTED_CKPT"] = None
    if "DATA_DIR" not in st.session_state:
        # để các module khác dùng; không show ở sidebar theo yêu cầu cũ
        st.session_state["DATA_DIR"] = "/media/mtl/DATA 6TB/AI DATASET/vietnamese-foods/Images"

def ensure_model_loaded(ckpt_path: Path, device: torch.device):
    key = str(ckpt_path)
    if key in st.session_state["GLOBAL_MODEL_CACHE"]:
        return st.session_state["GLOBAL_MODEL_CACHE"][key], st.session_state["GLOBAL_CLASSES_CACHE"][key]

    # tạm build model 33 lớp để đọc names nếu có
    tmp = mtl_efficientnet_b0_model(num_classes=33).to(device)
    names_from_ckpt, *_ = smart_load_weights(tmp, ckpt_path, device)
    classes = names_from_ckpt or [
        "Bánh beo","Bánh bot loc","Bánh can","Bánh canh","Bánh chung","Bánh cuon","Bánh duc",
        "Bánh gio","Bánh khot","Bánh mi","Bánh pia","Bánh tet","Bánh trang nuong","Bánh xeo",
        "Bun bo Hue","Bun dau mam tom","Bun mam","Bun rieu","Bun thit nuong","Ca kho to","Canh chua",
        "Cao lau","Chao long","Com tam","Goi cuon","Hu tieu","Mi quang","Nem chua","Pho","Xoi xeo",
        "banh_da_lon","banh_tieu","banh_trung_thu",
    ]
    model = mtl_efficientnet_b0_model(num_classes=len(classes)).to(device)
    smart_load_weights(model, ckpt_path, device)
    model.eval()

    st.session_state["GLOBAL_MODEL_CACHE"][key] = model
    st.session_state["GLOBAL_CLASSES_CACHE"][key] = classes
    return model, classes

# ================= App =================
st.set_page_config(page_title=TITLE, page_icon="🍜", layout="wide")
ensure_session_scaffolding()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- SIDEBAR -----
st.sidebar.title("Tuỳ chọn")

runs_dir = Path(st.session_state["RUNS_DIR"])
ckpts = list_best_ckpts(runs_dir)
if not ckpts:
    st.sidebar.warning(f"Không thấy checkpoint trong '{runs_dir}/**/checkpoints/*best*'")
    st.session_state["GLOBAL_SELECTED_CKPT"] = None
else:
    # hiển thị đường dẫn tương đối gọn
    base = runs_dir.parent if runs_dir.name == "runs" else runs_dir
    show = [str(p.relative_to(base)) for p in ckpts]
    idx = st.sidebar.selectbox("🏁 Checkpoint (best)", show, index=0, key="SB_CKPT_IDX")
    chosen = ckpts[show.index(idx)]
    st.session_state["GLOBAL_SELECTED_CKPT"] = str(chosen)
    model, classes = ensure_model_loaded(chosen, device)
    st.sidebar.success(f"Loaded · classes={len(classes)} · device={device.type.upper()}")

st.sidebar.markdown("---")
st.sidebar.caption("Thiết bị: **CUDA**" if device.type == "cuda" else "Thiết bị: **CPU**")

# ----- HEADER -----
st.title(TITLE)
with st.expander("📘 Giới thiệu & Hướng xử lý", expanded=False):
    st.markdown(
        "- App gồm 5 bước theo pipeline: **Khai phá dữ liệu → Tăng cường dữ liệu → Huấn luyện mô hình → Đánh giá → Dự đoán**.\n"
        "- Checkpoint (best) chỉ nạp **một lần** và dùng chung giữa các tab."
    )

# ----- Điều hướng -----
menu = st.sidebar.radio(
    "Chức năng",
    ["📊 Khai phá dữ liệu", "🌀 Tăng cường dữ liệu","🧠 Huấn luyện mô hình", "📈 Đánh giá mô hình", "🖼️ Dự đoán ảnh"],
    index=0
)


#augment_mode = st.sidebar.checkbox("🌀 Tăng cường dữ liệu")

# Nếu module train vừa set cờ cần reload danh sách ckpt → rerun để sidebar cập nhật ngay
if st.session_state.get("NEED_RELOAD_CKPTS"):
    st.session_state["NEED_RELOAD_CKPTS"] = False
    st.rerun()

# ----- Router -----
if menu == "📊 Khai phá dữ liệu":
    dm.run()
elif menu == "🌀 Tăng cường dữ liệu":
    import app_augment
    ag.run()
elif menu == "🧠 Huấn luyện mô hình":
    render_train_tab() 
elif menu == "📈 Đánh giá mô hình":
    vm.run()

else:
    pr.run()
