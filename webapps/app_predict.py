# app_predict.py — Inference UI (upload or random from Test)
import sys, io, zipfile, random, warnings
from pathlib import Path
from typing import List, Tuple
warnings.filterwarnings("ignore")

import streamlit as st
from PIL import Image
import torch, torch.nn.functional as F
from torchvision import transforms

IMG_TYPES = {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def tfm(mode: str):
    if mode=="keep_ratio":
        return transforms.Compose([
            transforms.Resize(IMG_SIZE, antialias=True),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def _read_files(files) -> List[Tuple[str, Image.Image]]:
    out=[]
    for f in files or []:
        try: out.append((f.name, Image.open(f).convert("RGB")))
        except: pass
    return out

def _read_zip(buf: bytes) -> List[Tuple[str, Image.Image]]:
    out=[]
    with zipfile.ZipFile(io.BytesIO(buf)) as z:
        for n in z.namelist():
            if Path(n).suffix.lower() in IMG_TYPES:
                with z.open(n) as f:
                    try: out.append((Path(n).name, Image.open(io.BytesIO(f.read())).convert("RGB")))
                    except: pass
    return out

def _pick_from_test(test_dir: Path, n: int) -> List[Tuple[str, Image.Image]]:
    imgs=[p for p in test_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_TYPES]
    if not imgs: return []
    pick=random.sample(imgs, min(n, len(imgs)))
    out=[]
    for p in pick:
        try: out.append((p.name, Image.open(p).convert("RGB")))
        except: pass
    return out

@torch.inference_mode()
def _infer(model, device, images: List[Image.Image], transform, topk: int):
    import torch as T
    batch=T.stack([transform(im) for im in images]).to(device).to(memory_format=T.channels_last)
    with T.amp.autocast(device_type='cuda' if device.type=='cuda' else 'cpu'):
        logits=model(batch)
        probs=F.softmax(logits, dim=1)
        conf, idx = T.topk(probs, k=topk, dim=1)
    return conf.cpu().numpy(), idx.cpu().numpy()

def run():
    st.header("🧠 Dự đoán ảnh món ăn Việt — EfficientNet-B0 (chuẩn notebook)")

    ckpt = st.session_state.get("GLOBAL_SELECTED_CKPT")
    if not ckpt or ckpt not in st.session_state["GLOBAL_MODEL_CACHE"]:
        st.warning("Chưa chọn **Checkpoint (best)** ở mục **Tuỳ chọn**."); return
    model = st.session_state["GLOBAL_MODEL_CACHE"][ckpt]
    classes = st.session_state["GLOBAL_CLASSES_CACHE"][ckpt] or []
    device = next(model.parameters()).device
    st.caption(f"Checkpoint: `{Path(ckpt).name}` • classes={len(classes) or '??'} • device={device.type.upper()}")

    resize_mode = st.radio("Resize mode", ["stretch","keep_ratio"], index=0, horizontal=True, key="pd_resize")
    topk   = st.slider("Top-K", 1, 5, 3, key="pd_topk")
    ncols  = st.slider("Số cột hiển thị", 1, 6, 4, key="pd_cols")
    show_p = st.toggle("Hiện % xác suất", value=True, key="pd_prob")

    src = st.radio("Nguồn ảnh", ["Upload ảnh/ZIP","Ngẫu nhiên từ Test"], horizontal=True, key="pd_src")
    images: List[Tuple[str, Image.Image]] = []

    if src=="Upload ảnh/ZIP":
        files = st.file_uploader("Tải ảnh", type=[t[1:] for t in IMG_TYPES], accept_multiple_files=True, key="pd_files")
        zf    = st.file_uploader("Hoặc 1 file ZIP", type=["zip"], key="pd_zip")
        images += _read_files(files)
        if zf: images += _read_zip(zf.read())
    else:
        data_root = Path(st.text_input("Thư mục Test (<DATA>/Test)",
                        value=Path(st.session_state.get("DATA_DIR", "/media/mtl/DATA 6TB/AI DATASET/vietnamese-foods/Images"))/"Test",
                        key="pd_testdir"))
        n = st.number_input("Số ảnh ngẫu nhiên", 1, 64, 12, key="pd_n")
        if st.button("🎲 Lấy ảnh", key="pd_btn_rand"):
            st.session_state["PD_CACHE_IMGS"] = _pick_from_test(data_root, int(n))
            st.success(f"Đã lấy {len(st.session_state.get('PD_CACHE_IMGS', []))} ảnh.")
        images = st.session_state.get("PD_CACHE_IMGS", [])

    if not images: return

    conf, idx = _infer(model, device, [im for _,im in images], tfm(resize_mode), topk)
    cols=st.columns(ncols)
    for i,(name,im) in enumerate(images):
        with cols[i % ncols]:
            st.image(im, use_container_width=True)
            pairs=[(classes[idx[i,k]] if classes else str(idx[i,k]), float(conf[i,k]))
                   for k in range(conf.shape[1])]
            st.write(" · ".join((f"**{lbl}** — {p*100:.1f}%" if show_p else f"**{lbl}**") for lbl,p in pairs))
            st.caption(name)
