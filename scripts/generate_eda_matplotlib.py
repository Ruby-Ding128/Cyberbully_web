import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "outputs" / "eda_analysis" / "eda_metrics.json"
FIGURE_DIR = ROOT / "outputs" / "eda_analysis" / "matplotlib_figures"

NAVY = "#17324D"
TEAL = "#168C8C"
BLUE = "#2878A5"
ORANGE = "#E88C30"
GRID = "#D8E1E8"

LABEL_ZH = {
    "age": "年龄欺凌",
    "ethnicity": "族裔欺凌",
    "gender": "性别欺凌",
    "religion": "宗教欺凌",
    "target": "总体毒性",
    "severe_toxicity": "严重毒性",
    "obscene": "粗俗/淫秽",
    "identity_attack": "身份攻击",
    "insult": "侮辱",
    "threat": "威胁",
    "sexual_explicit": "露骨色情",
}


def configure_chinese_font():
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    font_path = next((path for path in candidates if path.exists()), None)
    if font_path:
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["font.sans-serif"] = [font_name]
    plt.rcParams["axes.unicode_minus"] = False


def style_axis(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#516170")


def save(fig, name):
    fig.savefig(FIGURE_DIR / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    configure_chinese_font()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    # 数据来源占比
    source = metrics["source_counts"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = ["原毒性数据" if x["source"] == "data.csv" else "四类欺凌推文" for x in source]
    values = [x["count"] for x in source]
    colors = [BLUE, TEAL]
    wedges, _, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct=lambda p: f"{p:.1f}%",
        pctdistance=0.72,
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 2},
    )
    for text in autotexts:
        text.set_color("white")
        text.set_fontweight("bold")
    ax.text(0, 0.05, f"{sum(values):,}", ha="center", va="center", fontsize=17, fontweight="bold", color=NAVY)
    ax.text(0, -0.12, "总记录", ha="center", va="center", fontsize=10, color="#64748B")
    ax.set_title("合并数据来源构成", fontsize=15, fontweight="bold", color=NAVY, pad=14)
    save(fig, "01_source_share.png")

    # 四类标签分布
    binary = metrics["binary_summary"]
    binary = sorted(binary, key=lambda x: x["count"])
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    y = np.arange(len(binary))
    counts = [x["count"] for x in binary]
    bars = ax.barh(y, counts, color=["#55A7A3", "#3D9996", "#258B89", TEAL], height=0.62)
    ax.set_yticks(y, [LABEL_ZH[x["label"]] for x in binary])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("正例数量", color="#516170")
    ax.set_title("四个新增类别数量接近，整体较均衡", fontsize=15, fontweight="bold", color=NAVY, pad=14)
    for bar, count in zip(bars, counts):
        ax.text(count + 5, bar.get_y() + bar.get_height() / 2, f"{count:,}", va="center", color=NAVY, fontweight="bold")
    ax.set_xlim(min(counts) - 80, max(counts) + 80)
    style_axis(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.8)
    ax.grid(axis="y", visible=False)
    save(fig, "02_binary_labels.png")

    # 文本长度分布
    length = metrics["length_distribution"]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    names = [x["length_range"] for x in length]
    values = [x["count"] for x in length]
    bars = ax.bar(names, values, color=TEAL, width=0.68)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("字符长度区间", color="#516170")
    ax.set_ylabel("记录数", color="#516170")
    ax.set_title("文本主要集中在 101–500 字符", fontsize=15, fontweight="bold", color=NAVY, pad=14)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 600, f"{value:,}", ha="center", fontsize=9, color=NAVY)
    style_axis(ax)
    save(fig, "03_text_length.png")

    # 毒性标签阈值覆盖
    toxic = metrics["toxic_summary"]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    names = [LABEL_ZH[x["label"]] for x in toxic]
    positive = np.array([x["positive_count"] for x in toxic])
    high = np.array([x["ge_0_5_count"] for x in toxic])
    x = np.arange(len(names))
    width = 0.36
    ax.bar(x - width / 2, positive, width, label="评分 > 0", color="#80C5C0")
    ax.bar(x + width / 2, high, width, label="评分 ≥ 0.5", color=TEAL)
    ax.set_xticks(x, names, rotation=18, ha="right")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_ylabel("记录数", color="#516170")
    ax.set_title("data.csv 各毒性标签覆盖情况", fontsize=15, fontweight="bold", color=NAVY, pad=14)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    style_axis(ax)
    save(fig, "04_toxic_coverage.png")

    # 相关性热力图
    corr_rows = metrics["correlation"]
    keys = [x["label"] for x in corr_rows]
    matrix = np.array([[row[key] for key in keys] for row in corr_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8.6, 7.0))
    im = ax.imshow(matrix, cmap="GnBu", vmin=0, vmax=1)
    tick_labels = [LABEL_ZH[key] for key in keys]
    ax.set_xticks(np.arange(len(keys)), tick_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(keys)), tick_labels)
    for i in range(len(keys)):
        for j in range(len(keys)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white" if matrix[i, j] > 0.58 else NAVY, fontsize=9)
    ax.set_title("data.csv 毒性标签相关性热力图", fontsize=15, fontweight="bold", color=NAVY, pad=14)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("Pearson 相关系数", color="#516170")
    ax.spines[:].set_visible(False)
    save(fig, "05_correlation_heatmap.png")

    print(FIGURE_DIR)
    for path in sorted(FIGURE_DIR.glob("*.png")):
        print(path.name, path.stat().st_size)


if __name__ == "__main__":
    main()
