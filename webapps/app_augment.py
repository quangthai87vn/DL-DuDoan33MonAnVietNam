import streamlit as st
import os
from pathlib import Path
from PIL import Image, ImageEnhance
import random

st.title("🔄 Tăng cường dữ liệu hình ảnh (Data Augmentation)")
st.caption("Giúp mô hình học tốt hơn – giảm overfitting – tăng độ đa dạng mẫu.")

AUG_SAVE_DIR = Path("augmented")
AUG_SAVE_DIR.mkdir(exist_ok=True)

# =====================
# Các hàm augment
# =====================
def aug_flip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def aug_rotate(img):
    angle = random.randint(-25, 25)
    return img.rotate(angle, expand=True)

def aug_brightness(img):
    factor = random.uniform(0.5, 1.6)
    return ImageEnhance.Brightness(img).enhance(factor)

def aug_contrast(img):
    factor = random.uniform(0.5, 1.6)
    return ImageEnhance.Contrast(img).enhance(factor)

def aug_color(img):
    factor = random.uniform(0.5, 1.6)
    return ImageEnhance.Color(img).enhance(factor)

AUG_FUNCS = {
    "Lật ngang": aug_flip,
    "Xoay ngẫu nhiên (-25 → 25°)": aug_rotate,
    "Tăng/Giảm độ sáng": aug_brightness,
    "Tăng/Giảm độ tương phản": aug_contrast,
    "Tăng/Giảm độ bão hoà màu": aug_color
}


# =====================
# UI MAIN
# =====================

uploaded = st.file_uploader("📤 Tải ảnh lên để augment", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Ảnh gốc", use_column_width=True)

    st.subheader("🛠 Chọn kỹ thuật tăng cường dữ liệu")
    selected = st.multiselect(
        "Chọn 1 hoặc nhiều thao tác:",
        list(AUG_FUNCS.keys())
    )

    num_samples = st.slider("Số lượng ảnh augment tạo ra:", 5, 50, 10)

    if st.button("🚀 Tạo dữ liệu augment"):
        if not selected:
            st.warning("Hãy chọn ít nhất **1 kỹ thuật** augment!")
        else:
            st.success(f"Tạo {num_samples} mẫu dữ liệu…")

            cols = st.columns(5)
            generated_paths = []

            for i in range(num_samples):
                aug_img = img.copy()
                choosed = random.sample(selected, k=random.randint(1, len(selected)))

                for aug_name in choosed:
                    aug_img = AUG_FUNCS[aug_name](aug_img)

                save_path = AUG_SAVE_DIR / f"aug_{i}.jpg"
                aug_img.save(save_path)
                generated_paths.append(save_path)

                with cols[i % 5]:
                    st.image(aug_img, caption=f"aug_{i}", use_column_width=True)

            st.success(f"Đã lưu {len(generated_paths)} ảnh vào thư mục: **{AUG_SAVE_DIR}**")

# =====================
# Giải thích chuyên môn
# =====================
with st.expander("📘 Giải thích ý nghĩa từng kỹ thuật tăng cường"):
    st.markdown("""
### 1️⃣ Lật ngang  
Giúp mô hình nhìn món ăn từ góc trái/phải → tránh lệ thuộc vào bố cục cố định.

### 2️⃣ Xoay ảnh  
Giúp mô hình nhận dạng dù ảnh bị nghiêng – đặc biệt quan trọng cho món ăn chụp linh hoạt.

### 3️⃣ Tăng/Giảm độ sáng  
Đảm bảo mô hình vẫn nhận dạng tốt trong môi trường thiếu sáng / dư sáng.

### 4️⃣ Điều chỉnh độ tương phản  
Làm nổi bật chi tiết – giúp mô hình phân biệt màu sắc rõ hơn.

### 5️⃣ Điều chỉnh độ bão hoà màu  
Giúp mô hình không bị phụ thuộc màu sắc gốc của món ăn.  
Ví dụ: Bánh bèo có thể ngả vàng hoặc trắng, mô hình vẫn nhận được.
    """)
