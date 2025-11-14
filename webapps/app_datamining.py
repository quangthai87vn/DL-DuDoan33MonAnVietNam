# app_datamining.py — Data Mining UI (mỗi button 1 dòng, có progress + expander giải thích)
import sys
from pathlib import Path
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from matplotlib import cm
import math

sns.set_style("whitegrid")

IMG_TYPES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# ---------- helper: scan folder ----------
def scan_with_progress(data_dir: Path) -> pd.DataFrame:
    """Quét toàn bộ thư mục Train/Validate/Test + hiển thị progress."""
    rows, splits = [], ["Train", "Validate", "Test"]
    total_classes = sum(len(list((data_dir / s).glob("*/"))) for s in splits if (data_dir / s).exists())
    done = 0
    prog = st.progress(0, text="🔍 Đang quét dữ liệu...")

    for split in splits:
        d = data_dir / split
        if not d.exists():
            continue
        for cls in sorted([p for p in d.iterdir() if p.is_dir()]):
            n = len([p for p in cls.rglob("*") if p.is_file() and p.suffix.lower() in IMG_TYPES])
            rows.append({"split": split, "class": cls.name, "count": n})
            done += 1
            prog.progress(min(done / max(total_classes, 1), 1.0),
                          text=f"Đang quét: {split}/{cls.name} ({done}/{total_classes})")
    prog.empty()
    return pd.DataFrame(rows)


# ---------- plot functions ----------
def pie_overall(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw=dict(aspect="equal"))
    tot = df.groupby("split")["count"].sum().reindex(["Train", "Validate", "Test"])
    S = int(tot.sum())
    colors = ["#FFA726", "#66BB6A", "#26C6DA"]
    ax.pie(
        tot.values,
        labels=[f"{s}_set" for s in tot.index],
        autopct=lambda p: f"{p:.1f}%\n({int(p * S / 100)})",
        startangle=120,
        colors=colors,
        explode=(.05, .05, .05),
        shadow=True,
        wedgeprops=dict(width=.4, edgecolor="w"),
        textprops={"fontsize": 10},
    )
    ax.set_title("Phân bố số lượng ảnh Train / Validate / Test", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def pie_each(df: pd.DataFrame):
    splits = [s for s in ["Train", "Validate", "Test"] if s in df["split"].unique()]
    n = len(splits)
    fig, axes = plt.subplots(n, 1, figsize=(8, 4 * n))
    if n == 1:
        axes = [axes]
    cmap = cm.get_cmap("tab20c")
    for ax, s in zip(axes, splits):
        sub = df[df["split"] == s].sort_values("class")
        S = sub["count"].sum()
        ax.pie(sub["count"], labels=sub["class"],
               autopct=lambda p: f"{p:.1f}%\n({int(p * S / 100)})",
               startangle=90, colors=cmap.colors[:len(sub)], pctdistance=.8,
               textprops={"fontsize": 8})
        ax.set_title(f"Phân phối dữ liệu tập {s.lower()} ({len(sub)} lớp)")
    plt.tight_layout()
    return fig




def stacked_per_class(df: pd.DataFrame):
    # Pivot ra bảng class x (Train/Val/Test)
    pv = df.pivot_table(
        index="class",
        columns="split",
        values="count",
        fill_value=0
    )[["Train", "Validate", "Test"]]

    # 👉 Thêm cột tổng và sắp xếp giảm dần theo tổng số ảnh
    pv["__total__"] = pv.sum(axis=1)
    pv = pv.sort_values("__total__", ascending=False)
    pv = pv.drop(columns="__total__")

    # Vẽ stacked bar như cũ
    fig, ax = plt.subplots(figsize=(12, 6))
    pv.plot(kind="bar", stacked=True, ax=ax,
            color=["#FFA726", "#66BB6A", "#26C6DA"])

    ax.set_title("Phân phối theo lớp (stacked) — sắp xếp theo tổng số ảnh (giảm dần)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=60)

    plt.tight_layout()
    return fig




def total_per_class(df: pd.DataFrame):
    agg = df.groupby("class")["count"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=agg.index, y=agg.values, ax=ax, color="#42a5f5")
    ax.set_title("Tổng số ảnh theo lớp (gộp Train/Val/Test)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=60)
    for i, v in enumerate(agg.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    return fig


def counts_table_2panel(df: pd.DataFrame):
    """
    Vẽ bảng số lượng ảnh theo từng lớp (Train/Val/Test) chia làm 2 panel như ảnh mẫu.
    - Header có màu: Train=green, Val=orange, Test=blue.
    - Mỗi panel ~ n/2 lớp để dễ nhìn.
    """
    pv = (df.pivot_table(index="class", columns="split", values="count", fill_value=0)
            .reindex(columns=["Train", "Validate", "Test"]))
    classes = pv.index.tolist()
    n = len(classes)
    mid = math.ceil(n / 2)
    left_df, right_df = pv.iloc[:mid], pv.iloc[mid:]

    # style options
    head_colors = {"Train": "#66BB6A", "Validate": "#FFA726", "Test": "#42A5F5"}
    edge = "#555"

    def draw_one(ax, sub_df, title):
        ax.axis("off")
        # make bigger cell text
        table = ax.table(cellText=sub_df.values,
                         rowLabels=sub_df.index,
                         colLabels=sub_df.columns,
                         loc="center",
                         cellLoc="center",
                         colLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.25)

        # header colors
        for j, col in enumerate(sub_df.columns):
            cell = table[0, j]
            cell.set_facecolor(head_colors.get(col, "#e0e0e0"))
            cell.set_edgecolor(edge)
            cell.get_text().set_color("black")
            cell.get_text().set_fontweight("bold")

        # edge colors
        for (i, j), cell in table.get_celld().items():
            cell.set_edgecolor(edge)
            # nhẹ background cho body
            if i > 0:
                cell.set_facecolor("#FAFAFA")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 20], width_ratios=[1, 1])
    # big title
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.set_title("Thống kê số lượng ảnh theo từng món ăn (Train / Validate / Test)",
                       fontsize=16, fontweight="bold", pad=10)

    # two panels
    ax_left = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])
    draw_one(ax_left, left_df, f"Phần 1 ({len(left_df)} món)")
    draw_one(ax_right, right_df, f"Phần 2 ({len(right_df)} món)")

    plt.tight_layout()
    return fig


# ---------- unified UI helper ----------
def show_plot(title: str, fig, explain_md: str):
    st.markdown(f"### {title}")
    st.pyplot(fig, use_container_width=True)
    with st.expander("ℹ️ Giải thích nhanh", expanded=False):
        st.markdown(explain_md)
    st.divider()


# ---------- main ----------
def run():
    st.header("📊 Khai phá dữ liệu")

    data_dir = Path(st.text_input(
        "📁 DATA_DIR",
        value="/media/mtl/DATA 6TB/AI DATASET/vietnamese-foods/Images",
        key="dm_data"
    ))
    if not data_dir.exists():
        st.warning("⚠️ Không tìm thấy thư mục dữ liệu!")
        return

    # ----------- Quét dữ liệu -----------
    if st.button("🔍 Quét dữ liệu", key="dm_scan"):
        with st.spinner("Đang quét dữ liệu..."):
            df = scan_with_progress(data_dir)
        st.session_state["DM_DF"] = df
        st.success(f"✅ Đã quét xong! Tổng {len(df):,} dòng thống kê.")

    df = st.session_state.get("DM_DF")
    if df is None or df.empty:
        st.info("Bấm **🔍 Quét dữ liệu** để lấy thống kê.")
        return

    if "DM_FLAGS" not in st.session_state:
        st.session_state["DM_FLAGS"] = set()

    # ----------- Các nút vẽ (mỗi button 1 dòng) -----------
    if st.button("🧭 Biểu đồ tổng Train / Validate / Test", key="dm_btn_overall"):
        st.session_state["DM_FLAGS"].add("overall")
    if "overall" in st.session_state["DM_FLAGS"]:
        with st.spinner("Đang vẽ biểu đồ tổng..."):
            fig = pie_overall(df)
        show_plot("🧭 Biểu đồ tổng Train / Validate / Test", fig,
                  "- Hiển thị **tỷ lệ phần trăm** ảnh của từng tập (Train, Validate, Test).  \n"
                  "→ Dùng để **kiểm tra sự cân bằng tổng thể** giữa các tập dữ liệu.")

    if st.button("🍩 Pie từng tập (Train / Val / Test)", key="dm_btn_each"):
        st.session_state["DM_FLAGS"].add("each")
    if "each" in st.session_state["DM_FLAGS"]:
        with st.spinner("Đang vẽ biểu đồ pie từng tập..."):
            fig = pie_each(df)
        show_plot("🍩 Phân phối dữ liệu trong từng tập", fig,
                  "- Mỗi biểu đồ thể hiện **tỷ lệ ảnh từng lớp** trong một tập dữ liệu cụ thể.  \n"
                  "→ Giúp **so sánh mức độ cân bằng** giữa các lớp trong Train, Validate và Test.")

    if st.button("📊 Biểu đồ stacked theo lớp", key="dm_btn_stacked"):
        st.session_state["DM_FLAGS"].add("stacked")
    if "stacked" in st.session_state["DM_FLAGS"]:
        with st.spinner("Đang vẽ stacked bar..."):
            fig = stacked_per_class(df)
        show_plot("📊 Phân phối theo lớp (stacked)", fig,
                  "- Mỗi cột biểu diễn **một lớp**, chia thành ba phần: Train, Validate, Test.  \n"
                  "→ Giúp phát hiện **lớp nào thiếu ảnh ở tập Validate/Test** hoặc mất cân đối giữa các tập.")

    if st.button("🏷️ Tổng số ảnh theo từng lớp", key="dm_btn_total"):
        st.session_state["DM_FLAGS"].add("total")
    if "total" in st.session_state["DM_FLAGS"]:
        with st.spinner("Đang vẽ biểu đồ tổng số ảnh..."):
            fig = total_per_class(df)
        show_plot("🏷️ Tổng số ảnh theo từng lớp", fig,
                  "- Biểu đồ thanh thể hiện **tổng số ảnh** của mỗi lớp (gộp cả Train/Val/Test).  \n"
                  "→ Dùng để **đánh giá mức độ cân bằng dữ liệu** giữa các lớp, phát hiện lớp quá ít ảnh.")

    # 🔥 NÚT MỚI: BẢNG 2 PANEL NHƯ ẢNH MẪU
    if st.button("🧾 Bảng số lượng ảnh theo lớp (2 panel)", key="dm_btn_table2"):
        st.session_state["DM_FLAGS"].add("table2")
    if "table2" in st.session_state["DM_FLAGS"]:
        with st.spinner("Đang dựng bảng…"):
            fig = counts_table_2panel(df)
        show_plot("🧾 Bảng số lượng ảnh theo lớp (Train/Val/Test) — 2 panel", fig,
                  "- Bảng tách làm **2 phần** cho dễ quan sát khi số lớp nhiều.  \n"
                  "- Header có màu: **Train (xanh lá)**, **Validate (cam)**, **Test (xanh dương)**.  \n"
                  "→ Nhanh chóng soi lớp nào **thiếu ảnh** ở từng tập.")

    # ----------- DataFrame preview + export -----------
    with st.expander("📄 Bảng thống kê chi tiết & Tải CSV", expanded=False):
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "⬇️ Tải CSV",
            df.to_csv(index=False).encode("utf-8-sig"),
            "dataset_distribution.csv",
            "text/csv",
            key="dm_dl"
        )


if __name__ == "__main__":
    run()
