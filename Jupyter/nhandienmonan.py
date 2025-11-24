import os
import sys
from collections import Counter
import json   # <-- thêm dòng này
import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

# =========================
#  CẤU HÌNH CHUNG
# =========================

# Thư mục hiện tại của file nhandienmonan.py
BASE_DIR = os.path.dirname(__file__)

# Đường dẫn mặc định tới file class names
DEFAULT_CLASS_JSON = os.path.join(
    BASE_DIR,
    "runs_meta",
    "yolo_cls33_class_names.json"
)

def load_class_names(json_path: str = None):
    """
    Load danh sách tên lớp từ file JSON.
    File JSON dạng: ["Banh beo", "Banh bot loc", ...]
    """
    if json_path is None:
        json_path = DEFAULT_CLASS_JSON

    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Không tìm thấy file class names: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    if not isinstance(class_names, list):
        raise ValueError("File class names JSON phải là list các chuỗi.")

    return class_names


# Load class names 1 lần khi import module
CLASS_NAMES = load_class_names()
NUM_CLASSES = len(CLASS_NAMES)



def get_device():
    """Trả về 'cuda' nếu có GPU, không thì dùng 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# Transform cho EfficientNet-B0 (chuẩn ImageNet)
def get_classifier_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])












from pathlib import Path

def _extract_state_and_classes(raw):
    """Tách state_dict + class_names (nếu có) từ checkpoint."""
    names = None
    if isinstance(raw, dict):
        # lấy class_names nếu checkpoint có kèm
        for k in ("class_names", "classes", "labels"):
            if k in raw and isinstance(raw[k], (list, tuple)):
                names = list(raw[k])
                break
        # lấy state_dict thực sự
        for k in ("state_dict", "model_state", "model_state_dict", "model"):
            if k in raw and isinstance(raw[k], dict):
                return raw[k], names
        return raw, names
    return raw, None


def _strip_prefix(k: str, pref: str) -> str:
    return k[len(pref):] if k.startswith(pref) else k


def _map_keys(sd: dict, how: str) -> dict:
    """Thử nhiều kiểu map key khác nhau để khớp với model hiện tại."""
    if how == "identity":
        return {k: v for k, v in sd.items()}
    if how == "strip_module":
        return {_strip_prefix(k, "module."): v for k, v in sd.items()}
    if how == "strip_1tok":
        # bỏ token đầu tiên trước dấu chấm
        return {(k.split(".", 1)[1] if "." in k else k): v for k, v in sd.items()}
    if how == "strip_2tok":
        # bỏ 2 token đầu
        return {
            (".".join(k.split(".")[2:]) if k.count(".") >= 2 else k): v
            for k, v in sd.items()
        }
    if how == "features_to_backbone":
        # map features.* / classifier.* -> backbone.features.*, backbone.classifier.*
        out = {}
        for k, v in sd.items():
            if k.startswith(("features.", "classifier.")):
                out["backbone." + k] = v
            else:
                out[k] = v
        return out
    return sd


def smart_load_weights(model: torch.nn.Module, ckpt_path: Path, device: torch.device):
    """
    Load state_dict 'thông minh', tự map key, strict=False để dùng được nhiều kiểu checkpoint.
    Trả về (class_names, hit%, missing_keys, unexpected_keys) để in debug.
    """
    import torch as T

    raw = T.load(ckpt_path, map_location=device)
    sd_raw, names = _extract_state_and_classes(raw)

    # base: luôn strip module. trước
    base = _map_keys(sd_raw, "strip_module")

    model_keys = set(model.state_dict().keys())
    best_hit = -1
    best_sd = None

    for how in ["identity", "features_to_backbone", "strip_1tok", "strip_2tok"]:
        cand = _map_keys(base, how)
        hit = len(model_keys.intersection(cand.keys()))
        if hit > best_hit:
            best_hit = hit
            best_sd = cand

    missing, unexpected = model.load_state_dict(best_sd, strict=False)
    hit_pct = 100.0 * (len(model_keys) - len(missing)) / max(1, len(model_keys))
    return names, hit_pct, missing, unexpected




















# =========================
#  LOAD MODEL
# =========================

def load_yolo_model(weights_path: str, device: str = None):
    """
    Load mô hình YOLO để detect box các món ăn.
    """
    if device is None:
        device = get_device()

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Không tìm thấy YOLO weights: {weights_path}")

    model = YOLO(weights_path)
    # ultralytics tự handle device, nhưng ta vẫn truyền cho predict
    return model, device

'''
def load_classifier_model(weights_path: str, device: str = None):
    """
    Load mô hình EfficientNet-B0 đã train 33 món.
    - Đọc checkpoint
    - Suy ra số lớp từ layer classifier cuối
    - Khởi tạo MTLEfficientNetB0 đúng out_dim để không bị size mismatch.
    """
    if device is None:
        device = get_device()   # "cuda" hoặc "cpu"

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Không tìm thấy classifier weights: {weights_path}")

    # Thư mục models: Jupyter/models/efficientnet_b0.py
    base_dir = os.path.dirname(__file__)          # .../Jupyter
    models_dir = os.path.join(base_dir, "models") # .../Jupyter/models
    if models_dir not in sys.path:
        sys.path.append(models_dir)

    from efficientnet_b0 import MTLEfficientNetB0  # type: ignore

    torch_device = torch.device(device)

    # --- 1. Đọc checkpoint & lấy state_dict thực sự ---
    ckpt = torch.load(weights_path, map_location=torch_device)

    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # --- 2. Suy ra số lớp out_dim từ classifier cuối trong checkpoint ---
    out_dim = None
    for k, v in state_dict.items():
        if k.endswith("classifier.5.weight"):
            out_dim = v.shape[0]   # số hàng = số lớp
            break
    if out_dim is None:
        out_dim = NUM_CLASSES  # fallback (nếu không tìm thấy thì lấy theo JSON)

    print(f"➡️  Classifier checkpoint out_dim = {out_dim}")

    # --- 3. Khởi tạo model với đúng số lớp và load weight ---
    model = MTLEfficientNetB0(num_classes=out_dim).to(torch_device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"⚠️ Missing keys khi load classifier: {len(missing)}")
    if unexpected:
        print(f"⚠️ Unexpected keys khi load classifier: {len(unexpected)}")

    model.eval()
    transform = get_classifier_transform()
    return model, transform, device

'''
def load_classifier_model(weights_path: str, device: str = None):
    """
    Load mô hình EfficientNet-B0 đã train 34 (hoặc 33) món.
    - Import MTLEfficientNetB0 từ Jupyter/models/efficientnet_b0.py
    - Đọc checkpoint
    - Bỏ prefix `_orig_mod.` (do torch.compile)
    - Suy ra số lớp từ classifier cuối
    - Load state_dict (strict=False) vào MTLEfficientNetB0
    """
    if device is None:
        device = get_device()   # "cuda" hoặc "cpu"

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Không tìm thấy classifier weights: {weights_path}")

    # === trỏ đúng tới thư mục models (Jupyter/models) ===
    from pathlib import Path
    this_file = Path(__file__).resolve()          # .../Jupyter/nhandienmonan.py
    base_dir  = this_file.parent                  # .../Jupyter
    models_dir = base_dir / "models"              # .../Jupyter/models

    if str(models_dir) not in sys.path:
        sys.path.insert(0, str(models_dir))

    # file model gốc: Jupyter/models/efficientnet_b0.py
    from efficientnet_b0 import MTLEfficientNetB0  # type: ignore

    torch_device = torch.device(device)

    # --- 1. Đọc checkpoint & lấy state_dict thực sự ---
    raw = torch.load(weights_path, map_location=torch_device)

    class_names_from_ckpt = None
    if isinstance(raw, dict):
        # lấy class_names nếu có
        for k in ("class_names", "classes", "labels"):
            if k in raw and isinstance(raw[k], (list, tuple)):
                class_names_from_ckpt = list(raw[k])
                break
        # lấy state_dict chính
        for k in ("model_state", "model_state_dict", "state_dict", "model"):
            if k in raw and isinstance(raw[k], dict):
                state_dict = raw[k]
                break
        else:
            state_dict = raw
    else:
        state_dict = raw

    # --- 2. BỎ tiền tố `_orig_mod.` trong tất cả key (do torch.compile) ---
    fixed_sd = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("_orig_mod."):
            new_k = new_k[len("_orig_mod."):]  # "_orig_mod.backbone..." -> "backbone..."
        fixed_sd[new_k] = v
    state_dict = fixed_sd

    # --- 3. Suy ra số lớp out_dim từ classifier cuối ---
    out_dim = None
    for k, v in state_dict.items():
        if k.endswith("classifier.5.weight"):
            out_dim = v.shape[0]   # số hàng = số lớp
            break
    if out_dim is None:
        out_dim = len(CLASS_NAMES)   # fallback

    print(f"➡️  Classifier checkpoint out_dim = {out_dim}")

    # --- 4. Khởi tạo model đúng số lớp & load weight ---
    model = MTLEfficientNetB0(num_classes=out_dim).to(torch_device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"⚠️ Missing keys khi load classifier: {len(missing)}")
    print(f"⚠️ Unexpected keys khi load classifier: {len(unexpected)}")



    '''
    if class_names_from_ckpt:
        print(f"   class_names trong ckpt: {len(class_names_from_ckpt)} lớp")

    model.eval()
    transform = get_classifier_transform()
    return model, transform, device
    '''

    if class_names_from_ckpt:
        print(f"   class_names trong ckpt: {len(class_names_from_ckpt)} lớp")
        # ⚠ cập nhật lại CLASS_NAMES toàn module cho đúng thứ tự ckpt
        import sys as _sys
        _this_mod = _sys.modules[__name__]
        _this_mod.CLASS_NAMES = class_names_from_ckpt
        _this_mod.NUM_CLASSES = len(class_names_from_ckpt)

    model.eval()
    transform = get_classifier_transform()
    return model, transform, device







# =========================
#  XỬ LÝ ẢNH & DETECT
# =========================

def load_image(image_path: str) -> Image.Image:
    """
    Load ảnh bàn ăn từ đường dẫn.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
    img = Image.open(image_path).convert("RGB")
    return img


def detect_dishes(
    yolo_model,
    image_path: str,
    device: str = "cuda",
    det_conf_thres: float = 0.25
):
    """
    Chạy YOLO để detect các box món ăn trên bàn.
    Return:
        boxes_xyxy: np.ndarray shape (N, 4)
        scores: np.ndarray shape (N,)
    """
    results = yolo_model(
        source=image_path,
        conf=det_conf_thres,
        device=device,
        verbose=False
    )

    r = results[0]
    if r.boxes is None or r.boxes.xyxy is None:
        return np.zeros((0, 4), dtype=int), np.array([])

    boxes_xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
    scores = r.boxes.conf.cpu().numpy()

    return boxes_xyxy, scores


def crop_dish_patches(
    image: Image.Image,
    boxes_xyxy: np.ndarray,
    output_dir: str = None,
    base_name: str = "crop"
):
    """
    Cắt ảnh theo từng box món ăn.
    - image: PIL Image gốc
    - boxes_xyxy: np.array N x 4 (x1, y1, x2, y2)
    - output_dir: nếu không None thì lưu các crop xuống thư mục
    Return:
        crops: list[PIL.Image]
        crop_paths: list[str hoặc None]
    """
    crops = []
    crop_paths = []

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        crop = image.crop((x1, y1, x2, y2))
        crops.append(crop)

        if output_dir is not None:
            filename = f"{base_name}_crop_{i:03d}.jpg"
            path = os.path.join(output_dir, filename)
            crop.save(path)
            crop_paths.append(path)
        else:
            crop_paths.append(None)

    return crops, crop_paths


# =========================
#  CLASSIFY 33 MÓN
# =========================

@torch.no_grad()
def classify_dish_patches(
    crops,
    clf_model,
    transform,
    device: str = "cuda"
):
    """
    Nhận diện món ăn cho từng ảnh crop.
    Return:
        results: list[dict] mỗi phần tử:
            {
                "label_idx": int,
                "label_name": str,
                "score": float
            }
    """
    results = []

    for img in crops:
        tensor = transform(img).unsqueeze(0).to(device)
        logits = clf_model(tensor)
        probs = torch.softmax(logits, dim=1)
        score, idx = probs.max(dim=1)
        idx = idx.item()
        score = score.item()
        label_name = CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else f"class_{idx}"
        results.append({
            "label_idx": idx,
            "label_name": label_name,
            "score": score
        })

    return results


def count_dishes_by_label(class_results, cls_conf_thres: float = 0.8):
    """
    Đếm số lượng món ăn đã nhận diện vượt ngưỡng.
    Return:
        Counter {label_name: count}
    """
    labels = [
        r["label_name"]
        for r in class_results
        if r["score"] >= cls_conf_thres
    ]
    return Counter(labels)


# =========================
#  VẼ BOX & HIỂN THỊ
# =========================

def draw_boxes_with_labels(
    image: Image.Image,
    boxes_xyxy: np.ndarray,
    class_results,
    cls_conf_thres: float = 0.8,
    show_unknown: bool = False
) -> Image.Image:
    """
    Vẽ box + label lên ảnh:
    - Nếu score < cls_conf_thres: label "Unknown" (có thể ẩn nếu show_unknown=False)
    """
    img = np.array(image).copy()
    h, w, _ = img.shape

    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = box
        cls_res = class_results[i] if i < len(class_results) else None

        if cls_res is None:
            label_text = "Unknown"
            score = 0.0
        else:
            score = cls_res["score"]
            if score >= cls_conf_thres:
                label_text = f'{cls_res["label_name"]} {score * 100:.1f}%'
            else:
                label_text = "Unknown"

        # Nếu không muốn vẽ box Unknown thì skip
        if label_text == "Unknown" and not show_unknown:
            continue

        color = (0, 0, 255)    # đỏ

        # Vẽ rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Vẽ background cho text
        (tw, th), baseline = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1
        )
        y_text = max(0, y1 - th - 4)
        cv2.rectangle(
            img,
            (x1, y_text),
            (x1 + tw + 4, y_text + th + baseline + 4),
            color,
            -1
        )

        # Vẽ text
        cv2.putText(
            img,
            label_text,
            (x1 + 2, y_text + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return Image.fromarray(img)


def save_image(image: Image.Image, output_path: str):
    """
    Lưu ảnh xuống đường dẫn (tạo thư mục nếu chưa có).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    return output_path


# =========================
#  PIPELINE TỔNG HỢP
# =========================

def run_full_pipeline(
    image_path: str,
    yolo_weights: str,
    clf_weights: str,
    output_boxes_dir: str,
    output_crops_dir: str,
    det_conf_thres: float = 0.25,
    cls_conf_thres: float = 0.80,
    show_unknown: bool = False
):
    """
    Chạy full:
    - Load models
    - Detect box món ăn
    - Cắt từng món
    - Classify 33 món
    - Vẽ box + label lên ảnh
    - Lưu ảnh kết quả + crops
    Return:
        dict chứa toàn bộ intermediate để show trên notebook / streamlit
    """
    # Load ảnh
    image = load_image(image_path)

    # Load models
    yolo_model, yolo_device = load_yolo_model(yolo_weights)
    clf_model, transform, clf_device = load_classifier_model(clf_weights)

    # Detect box
    boxes_xyxy, det_scores = detect_dishes(
        yolo_model,
        image_path,
        device=yolo_device,
        det_conf_thres=det_conf_thres
    )

    # Nếu không detect được gì thì return sớm
    if len(boxes_xyxy) == 0:
        return {
            "image_path": image_path,
            "original_image": image,
            "boxes_xyxy": boxes_xyxy,
            "det_scores": det_scores,
            "crops": [],
            "crop_paths": [],
            "class_results": [],
            "counts": Counter(),
            "boxed_image": image,
            "boxed_image_path": None
        }

    # Cắt ảnh theo box
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    crops, crop_paths = crop_dish_patches(
        image,
        boxes_xyxy,
        output_dir=output_crops_dir,
        base_name=base_name
    )

    # Classify
    class_results = classify_dish_patches(
        crops,
        clf_model,
        transform,
        device=clf_device
    )

    # Đếm số lượng món theo ngưỡng cls_conf_thres
    counts = count_dishes_by_label(class_results, cls_conf_thres=cls_conf_thres)

    # Vẽ box + label
    boxed_image = draw_boxes_with_labels(
        image,
        boxes_xyxy,
        class_results,
        cls_conf_thres=cls_conf_thres,
        show_unknown=show_unknown
    )

    # Lưu ảnh đã vẽ box
    os.makedirs(output_boxes_dir, exist_ok=True)
    boxed_image_path = os.path.join(
        output_boxes_dir,
        f"{base_name}_boxed.jpg"
    )
    boxed_image.save(boxed_image_path)

    return {
        "image_path": image_path,
        "original_image": image,
        "boxes_xyxy": boxes_xyxy,
        "det_scores": det_scores,
        "crops": crops,
        "crop_paths": crop_paths,
        "class_results": class_results,
        "counts": counts,
        "boxed_image": boxed_image,
        "boxed_image_path": boxed_image_path
    }
