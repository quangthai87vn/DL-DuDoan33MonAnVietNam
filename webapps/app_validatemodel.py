# app_validatemodel.py — Model Evaluation (lazy render + pretty UI + inline expander help)
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report
)
from sklearn.preprocessing import label_binarize

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sns.set_style("whitegrid")


def _infer_all_with_progress(model, dl_test, device):
    """Suy luận toàn bộ test loader + progressbar theo từng batch."""
    total = len(dl_test)
    prog = st.progress(0, text="Đang suy luận…")
    logits_all, targets_all = [], []

    model.eval()
    for i, (imgs, labels) in enumerate(dl_test):
        imgs = imgs.to(device).to(memory_format=torch.channels_last)
        labels = labels.to(device)

        with torch.inference_mode(), torch.amp.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu"
        ):
            logits = model(imgs)

        logits_all.append(logits.detach().cpu())
        targets_all.append(labels.detach().cpu())

        prog.progress(
            (i + 1) / total,
            text=f"Đang suy luận… Batch {i+1}/{total}"
        )

    prog.empty()
    return torch.cat(logits_all, 0), torch.cat(targets_all, 0)


# ======== IMG & TRANSFORMS ========
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMG_SIZE = 224

def make_transform(mode: str = "stretch"):
    if mode == "keep_ratio":
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

# ======== SMALL UI HELPERS ========
def show_plot(title: str, fig, explainer_md: str, key: str = ""):
    """Render a titled plot + explanation expander with consistent spacing."""
    st.markdown(" ")
    st.markdown(f"### {title}")
    st.pyplot(fig, use_container_width=True)
    with st.expander("ℹ️ Giải thích nhanh", expanded=False):
        st.markdown(explainer_md)
    st.divider()

# ======== PLOTS ========
def _plot_confusion(cm, classes, err_thresh_pct: float = 10.0, dpi: int = 600):
    cm = np.asarray(cm, dtype=np.int32)
    per_row = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_pct = cm / per_row

    fig, ax = plt.subplots(figsize=(12, 12), dpi=dpi)
    cmap = sns.color_palette("YlGnBu", as_cmap=True)
    sns.heatmap(cm_pct * 100, cmap=cmap, vmin=0, vmax=100, cbar=True,
                xticklabels=classes, yticklabels=classes, ax=ax)

    ax.set_title("Confusion Matrix - Ma trận nhầm lẫn (% theo hàng)")
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Nhãn thật sự")

    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            p = cm_pct[i, j] * 100
            if i == j:
                ax.text(j + 0.5, i + 0.5, f"{p:.0f}%",
                        color="white", ha="center", va="center",
                        fontsize=8, fontweight="bold")
            else:
                if p >= err_thresh_pct:
                    ax.text(j + 0.5, i + 0.5, f"{cm[i, j]}\n({p:.0f}%)",
                            color="#F30000", ha="center", va="center",
                            fontsize=7)
    plt.tight_layout()
    return fig

def _plot_roc(y_true, prob, classes, top_worst: int = 5, dpi: int = 300):
    C = prob.shape[1]
    y_bin = label_binarize(y_true, classes=list(range(C)))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(C):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(C)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(C):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= C
    macro_auc = auc(all_fpr, mean_tpr)

    worst = sorted(range(C), key=lambda i: roc_auc[i])[:min(top_worst, C)]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    for i in worst:
        ax.plot(fpr[i], tpr[i], lw=2, label=f"{classes[i]} (AUC={roc_auc[i]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot(all_fpr, mean_tpr, "k--", lw=2, label=f"Macro-avg (AUC={macro_auc:.3f})")
    ax.set_title(f"ROC – Top {len(worst)} lớp AUC thấp")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=.3)
    plt.tight_layout()
    return fig

def _plot_pr(y_true, prob, classes, top_worst: int = 5, dpi: int = 300):
    C = prob.shape[1]
    y_bin = label_binarize(y_true, classes=list(range(C)))
    prc, rec, ap = {}, {}, {}
    for i in range(C):
        prc[i], rec[i], _ = precision_recall_curve(y_bin[:, i], prob[:, i])
        ap[i] = average_precision_score(y_bin[:, i], prob[:, i])

    worst = sorted(range(C), key=lambda i: ap[i])[:min(top_worst, C)]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    for i in worst:
        ax.plot(rec[i], prc[i], lw=2, label=f"{classes[i]} (AP={ap[i]:.3f})")
    ax.set_title(f"Precision–Recall – Top {len(worst)} lớp AP thấp")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    ax.grid(alpha=.3)
    plt.tight_layout()
    return fig

def _plot_topk(y_true, prob, k_max: int = 5, dpi: int = 300):
    order = np.argsort(-prob, axis=1)
    accs = []
    for k in range(1, k_max + 1):
        topk = order[:, :k]
        acc_k = np.mean([y in row for y, row in zip(y_true, topk)])
        accs.append(acc_k)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
    ax.plot(range(1, k_max + 1), np.array(accs) * 100, marker="o")
    ax.set_title("Top-K Accuracy")
    ax.set_xlabel("K")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(alpha=.3)
    ax.set_xticks(range(1, k_max + 1))
    plt.tight_layout()
    return fig

def _plot_perclass(y_true, y_pred, classes, dpi: int = 300):
    C = len(classes)

    # BỎ QUA các mẫu có label >= C (ví dụ dataset 34 lớp nhưng model 33 lớp)
    mask = y_true < C
    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]

    # Tính support & correct trên phần đã lọc
    support = np.bincount(y_true_f, minlength=C)[:C]
    correct = np.bincount(y_true_f[y_true_f == y_pred_f], minlength=C)[:C]

    acc_cls = np.divide(correct, np.maximum(support, 1))

    df = pd.DataFrame({
        "class": classes,
        "support": support,
        "acc": acc_cls
    }).sort_values("acc")

    fig, ax1 = plt.subplots(figsize=(10, 8), dpi=dpi)
    sns.barplot(data=df, x="acc", y="class", ax=ax1, color="#42a5f5")
    ax1.set_xlabel("Accuracy theo lớp")
    ax1.set_ylabel("")
    ax1.set_xlim(0, 1)
    for i, v in enumerate(df["acc"]):
        ax1.text(v + 0.01, i, f"{v*100:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    return fig



def _plot_conf_pairs(y_true, y_pred, classes, top_n: int = 10, dpi: int = 300):
    C = len(classes)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(C)))
    pairs = []
    for i in range(C):
        for j in range(C):
            if i != j and cm[i, j] > 0:
                pairs.append((classes[i], classes[j], int(cm[i, j])))

    pairs.sort(key=lambda x: -x[2])
    pairs = pairs[:top_n]
    if not pairs:
        fig, ax = plt.subplots(figsize=(6, 3), dpi=dpi)
        ax.text(0.5, 0.5, "Không có lỗi dự đoán.", ha="center", va="center")
        ax.axis("off")
        return fig

    df = pd.DataFrame(pairs, columns=["true", "pred", "count"])
    fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)
    sns.barplot(
        data=df,
        x="count",
        y=df.apply(lambda r: f"{r['true']} → {r['pred']}", axis=1),
        ax=ax,
        color="#ef5350",
    )
    ax.set_title("Top confusions (True → Pred)")
    ax.set_xlabel("Số lần nhầm")
    ax.set_ylabel("")
    for i, v in enumerate(df["count"]):
        ax.text(v + 0.5, i, f"{v}", va="center")
    plt.tight_layout()
    return fig

def _plot_prf_bars(y_true, y_pred, classes, dpi: int = 300):
    # labels = 0..C-1 theo model, để khớp với target_names
    labels = list(range(len(classes)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    rows = []
    for name in classes:
        if name in report:
            r = report[name]
            rows.append({
                "class": name,
                "precision": r.get("precision", 0.0),
                "recall":    r.get("recall", 0.0),
                "f1-score":  r.get("f1-score", 0.0),
            })

    if not rows:
        fig, ax = plt.subplots(figsize=(6, 3), dpi=dpi)
        ax.text(0.5, 0.5, "Không có dữ liệu để vẽ.", ha="center", va="center")
        ax.axis("off")
        return fig

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.35), 6), dpi=dpi)

    width = 0.25
    x = np.arange(len(df))
    ax.bar(x - width, df["precision"], width, label="precision")
    ax.bar(x,          df["recall"],    width, label="recall")
    ax.bar(x + width,  df["f1-score"],  width, label="f1-score")

    ax.set_ylabel("Score")
    ax.set_title("Biểu đồ cột Precision / Recall / F1 theo từng lớp")
    ax.set_xticks(x)
    ax.set_xticklabels(df["class"], rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig


def _plot_f1_sorted(y_true, y_pred, classes, ascending: bool = False, dpi: int = 300):
    # labels = 0..C-1 để khớp với target_names
    labels = list(range(len(classes)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    rows = [
        {"class": name, "f1": report.get(name, {}).get("f1-score", 0.0)}
        for name in classes
    ]
    df = pd.DataFrame(rows).sort_values("f1", ascending=ascending).reset_index(drop=True)

    fig_w = max(10, len(df) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, 5), dpi=dpi)

    ax.plot(range(len(df)), df["f1"], marker="o", linewidth=2)
    ax.set_title("F1-score (đường) sắp xếp theo thứ tự từng class")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["class"], rotation=45, ha="right")

    for i, v in enumerate(df["f1"]):
        if v <= 0.9:
            ax.text(i, v - 0.03, f"{v:.2f}", ha="center", va="top", fontsize=8, color="#d32f2f")

    plt.tight_layout()
    return fig



def plot_learning_curves(df: pd.DataFrame):
    """
    Vẽ 2 biểu đồ:
    - Loss theo epoch (train_loss, val_loss)
    - Accuracy theo epoch (train_acc, val_acc)
    df đọc từ metrics.csv, phải có các cột:
        epoch, train_loss, val_loss, train_acc, val_acc
    """
    df = df.sort_values("epoch").reset_index(drop=True)

    best_idx = df["val_acc"].idxmax()
    best_epoch = int(df.loc[best_idx, "epoch"])
    best_val_acc = float(df.loc[best_idx, "val_acc"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ---- Loss ----
    axes[0].plot(df["epoch"], df["train_loss"], label="train_loss", marker="o")
    axes[0].plot(df["epoch"], df["val_loss"], label="val_loss", marker="o")
    axes[0].axvline(best_epoch, ls="--", color="gray", alpha=0.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss theo epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ---- Accuracy ----
    axes[1].plot(df["epoch"], df["train_acc"], label="train_acc", marker="o")
    axes[1].plot(df["epoch"], df["val_acc"], label="val_acc", marker="o")
    axes[1].axvline(best_epoch, ls="--", color="gray", alpha=0.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy theo epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"Learning curves — best val_acc @ epoch {best_epoch}: {best_val_acc:.4f}",
        fontsize=12,
        fontweight="bold"
    )
    plt.tight_layout()
    return fig


# ======== MAIN TAB RUN ========
def run():
    st.header("📈 Đánh giá mô hình")

    ckpt_str = st.session_state.get("GLOBAL_SELECTED_CKPT")
    if not ckpt_str or ckpt_str not in st.session_state["GLOBAL_MODEL_CACHE"]:
        st.warning("Chưa load được model từ sidebar.")
        return

    model = st.session_state["GLOBAL_MODEL_CACHE"][ckpt_str]
    classes = st.session_state["GLOBAL_CLASSES_CACHE"][ckpt_str] or []
    device = next(model.parameters()).device

    ckpt_path = Path(ckpt_str)
    run_dir = ckpt_path.parent.parent

    data_dir = Path(st.session_state.get(
        "DATA_DIR",
        "/media/mtl/DATA 6TB/AI DATASET/vietnamese-foods/Images"
    ))

    # Test dir & resize mode (vẫn cho chỉnh, nhưng sẽ auto evaluate)
    default_test_dir = data_dir / "Test"
    test_dir = Path(st.text_input("📁 Test Dir", value=str(default_test_dir), key="vm_test"))
    resize_mode = st.radio(
        "Resize mode",
        ["stretch", "keep_ratio"],
        index=0,
        horizontal=True,
        key="vm_resize"
    )
    bs = st.slider("Batch size đánh giá", 8, 128, 64, step=8, key="vm_bs")

    st.caption(
        f"Checkpoint: `{ckpt_path.name}` • classes={len(classes) or '??'} • "
        f"device={device.type.upper()}"
    )

    # ========== PHÂN TÍCH QUÁ TRÌNH TRAIN TỪ metrics.csv ==========
    st.subheader("📊 Learning curves (metrics.csv)")

    metrics_path = run_dir / "metrics.csv"
    st.caption(f"📄 metrics.csv: `{metrics_path}`")

    if not metrics_path.exists():
        st.warning("⚠️ Không tìm thấy file metrics.csv trong thư mục run này.")
    else:
        df_metrics = pd.read_csv(metrics_path)
        required = {"epoch", "train_loss", "val_loss", "train_acc", "val_acc"}
        if not required.issubset(df_metrics.columns):
            st.error(f"metrics.csv thiếu cột: {required - set(df_metrics.columns)}")
        else:
            with st.spinner("Đang vẽ learning curves..."):
                fig_lc = plot_learning_curves(df_metrics)
            st.pyplot(fig_lc, use_container_width=True)

            st.markdown("#### 📋 Một vài epoch cuối")
            st.dataframe(
                df_metrics.tail(5).reset_index(drop=True),
                use_container_width=True,
            )

    # ========== AUTO EVAL TRÊN TEST KHI ĐỔI CHECKPOINT ==========
    cache = st.session_state.get("VM_CACHE")
    cache_ckpt = st.session_state.get("VM_CACHE_CKPT")

    need_eval = (
        cache is None
        or cache_ckpt != ckpt_str
    )

    if need_eval:
        # chạy đánh giá luôn cho checkpoint này
        if not test_dir.exists():
            st.error(f"Thư mục Test không tồn tại: {test_dir}")
            return

        tfm = make_transform(resize_mode)
        ds_test = datasets.ImageFolder(test_dir, transform=tfm)
        if not classes:
            classes = ds_test.classes
            st.session_state["GLOBAL_CLASSES_CACHE"][ckpt_str] = classes

        dl_test = DataLoader(
            ds_test,
            batch_size=bs,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        with st.spinner("🧪 Đang đánh giá mô hình trên tập Test..."):
            logits, targets = _infer_all_with_progress(model, dl_test, device)

        probs = torch.softmax(logits, dim=1).numpy()
        y_true = targets.numpy()
        y_pred = probs.argmax(axis=1)

        st.session_state["VM_CACHE"] = dict(
            y_true=y_true,
            y_pred=y_pred,
            probs=probs,
            classes=classes,
        )
        st.session_state["VM_CACHE_CKPT"] = ckpt_str

        st.success(
            f"Test Accuracy (auto): {(y_pred == y_true).mean() * 100:.2f}% "
            f"| Số ảnh: {len(y_true):,}"
        )
    else:
        # đã có cache cho checkpoint này
        st.info(
            "Đã sử dụng kết quả đánh giá cache cho checkpoint hiện tại. "
            "Đổi checkpoint ở sidebar để đánh giá model khác."
        )

    cache = st.session_state.get("VM_CACHE")
    if not cache:
        st.info("Chưa có kết quả đánh giá để vẽ biểu đồ.")
        return

    y_true = cache["y_true"]
    y_pred = cache["y_pred"]
    probs = cache["probs"]
    classes = cache["classes"]

    if "VM_FLAGS" not in st.session_state:
        st.session_state["VM_FLAGS"] = set()

    # ===== NÚT VẼ BIỂU ĐỒ =====
    if st.button("🔀 Vẽ Confusion Matrix", key="vm_btn_cm"):
        st.session_state["VM_FLAGS"].add("cm")
    if "cm" in st.session_state["VM_FLAGS"]:
        err_thresh = st.slider("Annotate lỗi ≥ (%)", 0, 50, 10, key="vm_err_thresh")
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
        fig = _plot_confusion(cm, classes, err_thresh_pct=float(err_thresh), dpi=600)
        explain = (
            "- **Confusion Matrix**: mô tả tỉ lệ dự đoán đúng/sai theo từng lớp (mỗi hàng chuẩn hóa 100%).  \n"
            "  • Ô **đường chéo** là tỉ lệ đúng của từng lớp.  \n"
            "  • Ô **màu đỏ có số** là các lỗi nổi bật (≥ ngưỡng bạn chọn).  \n"
            "  → Dùng để **soi các cặp lớp dễ nhầm** và kiểm tra chất lượng phân biệt giữa các lớp."
        )
        show_plot("🔀 Confusion Matrix (% theo hàng)", fig, explain, key="sec_cm")

    if st.button("📊 Vẽ Precision / Recall / F1 theo lớp", key="vm_btn_prf_bars"):
        st.session_state["VM_FLAGS"].add("prf_bars")
    if "prf_bars" in st.session_state["VM_FLAGS"]:
        fig = _plot_prf_bars(y_true, y_pred, classes)
        explain = (
            "- **Precision**: trong số ảnh model dự đoán là lớp *X*, bao nhiêu % là đúng lớp *X*.  \n"
            "- **Recall**: trong số ảnh *X* thực tế, model thu hồi được bao nhiêu %.  \n"
            "- **F1-score**: trung bình hài hòa giữa precision & recall.  \n"
            "→ Dùng để **so sánh chất lượng từng lớp**, phát hiện lớp **precision thấp** (dễ dương tính giả) "
            "hoặc **recall thấp** (hay bỏ sót)."
        )
        show_plot("📊 Precision / Recall / F1 theo lớp", fig, explain, key="sec_prf_bars")

    if st.button("📈 Vẽ F1-score (đường, sắp xếp)", key="vm_btn_f1line"):
        st.session_state["VM_FLAGS"].add("f1line")
    if "f1line" in st.session_state["VM_FLAGS"]:
        asc = st.checkbox("Sắp xếp tăng dần (mặc định giảm dần)", value=False, key="vm_f1_asc")
        fig = _plot_f1_sorted(y_true, y_pred, classes, ascending=asc)
        explain = (
            "- **F1-line theo lớp (đã sắp xếp)**: nhìn nhanh lớp nào **yếu nhất/khỏe nhất**.  \n"
            "→ Ưu tiên xử lý các lớp F1 thấp: **bổ sung dữ liệu**, **tăng augmentation**, **tách lớp/đổi label** nếu cần."
        )
        show_plot("📈 F1-score (đường) — sắp xếp theo lớp", fig, explain, key="sec_f1line")

    if st.button("🧮 Vẽ ROC (Top worst)", key="vm_btn_roc"):
        st.session_state["VM_FLAGS"].add("roc")
    if "roc" in st.session_state["VM_FLAGS"]:
        top_k = st.slider("Top-k AUC thấp", 3, 10, 5, key="vm_topk_roc")
        fig = _plot_roc(y_true, probs, classes, top_worst=int(top_k))
        explain = (
            "- **ROC/AUC**: đo năng lực **phân tách** lớp *X* với phần còn lại ở nhiều ngưỡng.  \n"
            "  • **AUC** càng gần 1 càng tốt.  \n"
            "→ Dùng để **soi các lớp phân tách kém** (AUC thấp), gợi ý điều chỉnh dữ liệu/tiền xử lý."
        )
        show_plot("🧮 ROC – Top lớp AUC thấp", fig, explain, key="sec_roc")

    if st.button("🧷 Vẽ Precision–Recall (Top worst)", key="vm_btn_pr"):
        st.session_state["VM_FLAGS"].add("pr")
    if "pr" in st.session_state["VM_FLAGS"]:
        top_k_pr = st.slider("Top-k AP thấp", 3, 10, 5, key="vm_topk_pr")
        fig = _plot_pr(y_true, probs, classes, top_worst=int(top_k_pr))
        explain = (
            "- **Precision–Recall / AP**: hữu ích khi dữ liệu **mất cân bằng**.  \n"
            "  • **AP** là diện tích dưới đường P–R (càng cao càng tốt).  \n"
            "→ Dùng để **đánh giá độ ổn định precision/recall** theo ngưỡng."
        )
        show_plot("🧷 Precision–Recall – Top lớp AP thấp", fig, explain, key="sec_pr")

    if st.button("🎯 Vẽ Top-K Accuracy", key="vm_btn_topk"):
        st.session_state["VM_FLAGS"].add("topk")
    if "topk" in st.session_state["VM_FLAGS"]:
        kmax = st.slider("K tối đa", 3, 10, 5, key="vm_kmax")
        fig = _plot_topk(y_true, probs, k_max=int(kmax))
        explain = (
            "- **Top-K Accuracy**: ảnh đúng nếu **nhãn đúng nằm trong K dự đoán cao nhất**.  \n"
            "→ Dùng khi ứng dụng hiển thị **K gợi ý** (ví dụ top-3 món ăn) thay vì một nhãn duy nhất."
        )
        show_plot("🎯 Top-K Accuracy", fig, explain, key="sec_topk")

    if st.button("🧩 Accuracy theo lớp & Support", key="vm_btn_perclass"):
        st.session_state["VM_FLAGS"].add("perclass")
    if "perclass" in st.session_state["VM_FLAGS"]:
        fig = _plot_perclass(y_true, y_pred, classes)
        explain = (
            "- **Accuracy theo lớp** kèm **Support (số mẫu/lớp)**.  \n"
            "→ Phát hiện **lớp ít dữ liệu** (support thấp) hoặc **lớp nhiều nhưng accuracy thấp** "
            "(cần xem lại chất lượng dữ liệu/nhãn)."
        )
        show_plot("🧩 Accuracy theo lớp & Support", fig, explain, key="sec_perclass")

    if st.button("🚨 Top confusions (True → Pred)", key="vm_btn_pairs"):
        st.session_state["VM_FLAGS"].add("pairs")
    if "pairs" in st.session_state["VM_FLAGS"]:
        top_conf = st.slider("Số cặp hiển thị", 5, 20, 10, key="vm_top_conf")
        fig = _plot_conf_pairs(y_true, y_pred, classes, top_n=int(top_conf))
        explain = (
            "- **Top confusions**: các cặp *True → Pred* bị nhầm nhiều nhất.  \n"
            "→ Dùng để **điều tra nguyên nhân nhầm lẫn** (ảnh giống nhau, ánh sáng/góc chụp, label chồng lấn) "
            "và **đề xuất tách/ghép lớp, tăng dữ liệu**."
        )
        show_plot("🚨 Top Confusions (True → Pred)", fig, explain, key="sec_pairs")
