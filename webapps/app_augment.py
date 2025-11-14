# app_augment.py
import io
from pathlib import Path

import streamlit as st
from PIL import Image
from torchvision import transforms
import torch


# ==== HẰNG SỐ CƠ BẢN ====
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ==== HÀM TẠO TRANSFORMS TỪ UI ====
def build_train_tfms(cfg, img_size: int):
    ops = []

    # 1. Luôn đảm bảo ảnh là RGB
    ops.append(transforms.Lambda(lambda im: im.convert("RGB")))

    # 2. Resize / RandomResizedCrop
    if cfg["use_rrc"]:
        ops.append(
            transforms.RandomResizedCrop(
                img_size,
                scale=(cfg["rrc_scale_min"], cfg["rrc_scale_max"]),
                ratio=(cfg["rrc_ratio_min"], cfg["rrc_ratio_max"]),
            )
        )
    else:
        ops.append(transforms.Resize((img_size, img_size)))

    # 3. Flip ngang
    if cfg["use_hflip"]:
        ops.append(transforms.RandomHorizontalFlip(p=cfg["hflip_p"]))

    # 4. Xoay
    if cfg["use_rotate"]:
        ops.append(transforms.RandomRotation(degrees=cfg["rotate_deg"]))

    # 5. Color jitter (độ sáng, tương phản, saturation, hue)
    if cfg["use_colorjitter"]:
        ops.append(
            transforms.ColorJitter(
                brightness=cfg["cj_brightness"],
                contrast=cfg["cj_contrast"],
                saturation=cfg["cj_saturation"],
                hue=cfg["cj_hue"],
            )
        )

    # 6. Affine nhẹ (dịch, zoom, shear) – optional
    if cfg["use_affine"]:
        ops.append(
            transforms.RandomAffine(
                degrees=cfg["affine_deg"],
                translate=(cfg["affine_translate"], cfg["affine_translate"]),
                scale=(1.0 - cfg["affine_scale"], 1.0 + cfg["affine_scale"]),
                shear=cfg["affine_shear"],
            )
        )

    # 7. Gaussian Blur (áp dụng ngẫu nhiên)
    if cfg["use_blur"]:
        ops.append(
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=cfg["blur_kernel"],
                        sigma=(cfg["blur_sigma_min"], cfg["blur_sigma_max"]),
                    )
                ],
                p=cfg["blur_p"],
            )
        )

    # 8. Chuyển sang Tensor + Normalize (luôn phải có)
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))

    # 9. RandomErasing – CHỈ ĐƯỢC đặt sau ToTensor/Normalize
    if cfg["use_erasing"]:
        ops.append(
            transforms.RandomErasing(
                p=cfg["erase_p"],
                scale=(cfg["erase_scale_min"], cfg["erase_scale_max"]),
                ratio=(cfg["erase_ratio_min"], cfg["erase_ratio_max"]),
                value="random",
            )
        )

    return transforms.Compose(ops)


# ==== HÀM HIỂN THỊ CODE PYTHON TƯƠNG ỨNG ====
def render_code(cfg, img_size: int):
    lines = [
        "from torchvision import transforms",
        "",
        f"IMG_SIZE = {img_size}",
        "IMAGENET_MEAN = (0.485, 0.456, 0.406)",
        "IMAGENET_STD  = (0.229, 0.224, 0.225)",
        "",
        "train_tfms = transforms.Compose([",
        "    transforms.Lambda(lambda im: im.convert('RGB')),",
    ]

    if cfg["use_rrc"]:
        lines.append(
            f"    transforms.RandomResizedCrop(IMG_SIZE, "
            f"scale=({cfg['rrc_scale_min']}, {cfg['rrc_scale_max']}), "
            f"ratio=({cfg['rrc_ratio_min']}, {cfg['rrc_ratio_max']})),"
        )
    else:
        lines.append("    transforms.Resize((IMG_SIZE, IMG_SIZE)),")

    if cfg["use_hflip"]:
        lines.append(f"    transforms.RandomHorizontalFlip(p={cfg['hflip_p']}),")

    if cfg["use_rotate"]:
        lines.append(f"    transforms.RandomRotation(degrees={cfg['rotate_deg']}),")

    if cfg["use_colorjitter"]:
        lines.append(
            "    transforms.ColorJitter("
            f"brightness={cfg['cj_brightness']}, "
            f"contrast={cfg['cj_contrast']}, "
            f"saturation={cfg['cj_saturation']}, "
            f"hue={cfg['cj_hue']}),"
        )

    if cfg["use_affine"]:
        lines.append(
            "    transforms.RandomAffine("
            f"degrees={cfg['affine_deg']}, "
            f"translate=({cfg['affine_translate']}, {cfg['affine_translate']}), "
            f"scale=(1.0 - {cfg['affine_scale']}, 1.0 + {cfg['affine_scale']}), "
            f"shear={cfg['affine_shear']}),"
        )

    if cfg["use_blur"]:
        lines.append(
            "    transforms.RandomApply(["
            "transforms.GaussianBlur("
            f"kernel_size={cfg['blur_kernel']}, "
            f"sigma=({cfg['blur_sigma_min']}, {cfg['blur_sigma_max']}))"
            f"], p={cfg['blur_p']}),"
        )

    lines += [
        "    transforms.ToTensor(),",
        "    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),",
    ]

    if cfg["use_erasing"]:
        lines.append(
            "    transforms.RandomErasing("
            f"p={cfg['erase_p']}, "
            f"scale=({cfg['erase_scale_min']}, {cfg['erase_scale_max']}), "
            f"ratio=({cfg['erase_ratio_min']}, {cfg['erase_ratio_max']}), "
            "value='random'),"
        )

    lines.append("])")
    return "\n".join(lines)


# ==== APP STREAMLIT CHÍNH ====
def run():
    #st.set_page_config(page_title="Cấu hình Data Augmentation", layout="wide")

    st.title("🧪 Cấu hình Data Augmentation cho món ăn VN")

    # --- Sidebar: tham số chung ---
    st.sidebar.header("⚙️ Cấu hình chung")
    img_size = st.sidebar.slider("IMG_SIZE", 128, 512, 224, step=16)

    st.sidebar.markdown("---")
    st.sidebar.header("📦 Các phép biến đổi")

    # RandomResizedCrop / Resize
    use_rrc = st.sidebar.checkbox("RandomResizedCrop thay cho Resize", True)
    if use_rrc:
        rrc_scale = st.sidebar.slider("RRC scale", 0.3, 1.0, (0.85, 1.0))
        rrc_ratio = st.sidebar.slider("RRC ratio", 0.5, 2.0, (0.9, 1.1))
    else:
        rrc_scale = (0.85, 1.0)
        rrc_ratio = (0.9, 1.1)

    # Flip
    use_hflip = st.sidebar.checkbox("RandomHorizontalFlip", True)
    hflip_p = st.sidebar.slider("Xác suất lật ngang", 0.0, 1.0, 0.5, 0.05) if use_hflip else 0.5

    # Rotate
    use_rotate = st.sidebar.checkbox("RandomRotation", True)
    rotate_deg = st.sidebar.slider("Góc xoay tối đa (±)", 0, 45, 10, 1) if use_rotate else 0

    # ColorJitter
    use_colorjitter = st.sidebar.checkbox("ColorJitter (brightness/contrast/…)", True)
    if use_colorjitter:
        cj_brightness = st.sidebar.slider("Brightness", 0.0, 0.8, 0.25, 0.05)
        cj_contrast   = st.sidebar.slider("Contrast",   0.0, 0.8, 0.25, 0.05)
        cj_saturation = st.sidebar.slider("Saturation", 0.0, 0.8, 0.20, 0.05)
        cj_hue        = st.sidebar.slider("Hue",        0.0, 0.2, 0.05, 0.01)
    else:
        cj_brightness = cj_contrast = cj_saturation = cj_hue = 0.0

    # Affine
    use_affine = st.sidebar.checkbox("RandomAffine (dịch/zoom/shear nhẹ)", True)
    if use_affine:
        affine_deg = st.sidebar.slider("Affine degrees", 0, 30, 5)
        affine_translate = st.sidebar.slider("Translate (tỉ lệ)", 0.0, 0.4, 0.05, 0.01)
        affine_scale = st.sidebar.slider("Scale jitter", 0.0, 0.3, 0.05, 0.01)
        affine_shear = st.sidebar.slider("Shear (độ)", 0, 30, 5)
    else:
        affine_deg = affine_translate = affine_scale = affine_shear = 0.0

    # Gaussian Blur
    use_blur = st.sidebar.checkbox("GaussianBlur ngẫu nhiên", True)
    if use_blur:
        blur_p = st.sidebar.slider("P(Blur)", 0.0, 1.0, 0.2, 0.05)
        blur_kernel = st.sidebar.selectbox("Kernel size", [3, 5], index=0)
        blur_sigma = st.sidebar.slider("Sigma range", 0.1, 2.0, (0.1, 1.0), 0.1)
    else:
        blur_p = 0.0
        blur_kernel = 3
        blur_sigma = (0.1, 1.0)

    # RandomErasing
    use_erasing = st.sidebar.checkbox("RandomErasing (sau Normalize)", True)
    if use_erasing:
        erase_p = st.sidebar.slider("P(Erasing)", 0.0, 1.0, 0.25, 0.05)
        erase_scale = st.sidebar.slider("Scale (min, max)", 0.01, 0.3, (0.02, 0.15), 0.01)
        erase_ratio = st.sidebar.slider("Ratio (min, max)", 0.1, 3.5, (0.3, 3.3), 0.1)
    else:
        erase_p = 0.0
        erase_scale = (0.02, 0.15)
        erase_ratio = (0.3, 3.3)

    # Gộp cấu hình
    cfg = dict(
        use_rrc=use_rrc,
        rrc_scale_min=rrc_scale[0],
        rrc_scale_max=rrc_scale[1],
        rrc_ratio_min=rrc_ratio[0],
        rrc_ratio_max=rrc_ratio[1],
        use_hflip=use_hflip,
        hflip_p=hflip_p,
        use_rotate=use_rotate,
        rotate_deg=rotate_deg,
        use_colorjitter=use_colorjitter,
        cj_brightness=cj_brightness,
        cj_contrast=cj_contrast,
        cj_saturation=cj_saturation,
        cj_hue=cj_hue,
        use_affine=use_affine,
        affine_deg=affine_deg,
        affine_translate=affine_translate,
        affine_scale=affine_scale,
        affine_shear=affine_shear,
        use_blur=use_blur,
        blur_p=blur_p,
        blur_kernel=blur_kernel,
        blur_sigma_min=blur_sigma[0],
        blur_sigma_max=blur_sigma[1],
        use_erasing=use_erasing,
        erase_p=erase_p,
        erase_scale_min=erase_scale[0],
        erase_scale_max=erase_scale[1],
        erase_ratio_min=erase_ratio[0],
        erase_ratio_max=erase_ratio[1],
    )

    train_tfms = build_train_tfms(cfg, img_size)

    # ==== MAIN AREA ====
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🧾 Code transforms sinh ra")
        code = render_code(cfg, img_size)
        st.code(code, language="python")

    with col2:
        st.subheader("👀 Xem thử augment trên 1 ảnh")
        uploaded = st.file_uploader("Chọn 1 ảnh demo", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            img = Image.open(uploaded)
            st.write("Ảnh gốc:")
            st.image(img, use_column_width=True)

            st.write("Ảnh sau augment (1 lần apply):")
            aug_img = train_tfms(img)
            # chuyển lại về PIL để hiển thị
            inv = transforms.Normalize(
                mean=[-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
                std=[1.0 / s for s in IMAGENET_STD],
            )
            aug_np = inv(aug_img).clamp(0, 1)
            aug_pil = transforms.ToPILImage()(aug_np)
            st.image(aug_pil, use_column_width=True)
        else:
            st.info("Upload 1 file ảnh để xem hiệu ứng augment.")

    st.markdown("---")
    st.markdown(
        "✅ **Ghi chú quan trọng:** `RandomErasing` được đặt **sau** `ToTensor()` và "
        "`Normalize(IMAGENET_MEAN, IMAGENET_STD)`, tránh lỗi "
        "`Image object has no attribute 'shape'` như bạn gặp trước đó."
    )


if __name__ == "__main__":
    run()
