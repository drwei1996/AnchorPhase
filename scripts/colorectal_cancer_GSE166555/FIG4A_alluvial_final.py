# -*- coding: utf-8 -*-
"""
FIG4A alluvial plot: Normal vs Tumor rhythmic-gene membership
作者：ChatGPT（按你的要求重写）
说明：
1. 本图基于 union 5,434 genes（在 Normal 或 Tumor 至少一边 phase-associated）
2. 因此只有 3 条流：
   - R  -> R   = 2,097  （shared rhythmic）
   - R  -> AR  = 1,611  （Normal-only rhythmic）
   - AR -> R   = 1,726  （Tumor-only rhythmic）
   - 不存在 AR -> AR
3. 同时输出：
   - 有字版
   - 无字版
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.ticker import FuncFormatter


# =========================
# 1. 数据区（你这张图固定就是这些数）
# =========================
UNION_TOTAL = 5434

# 左边（Normal）
NORMAL_R = 3708
NORMAL_AR = 1726

# 右边（Tumor）
TUMOR_R = 3823
TUMOR_AR = 1611

# 三条流
FLOW_R_TO_R = 2097      # shared rhythmic
FLOW_R_TO_AR = 1611     # normal-only rhythmic
FLOW_AR_TO_R = 1726     # tumor-only rhythmic

assert NORMAL_R + NORMAL_AR == UNION_TOTAL
assert TUMOR_R + TUMOR_AR == UNION_TOTAL
assert FLOW_R_TO_R + FLOW_R_TO_AR == NORMAL_R
assert FLOW_R_TO_R + FLOW_AR_TO_R == TUMOR_R
assert FLOW_AR_TO_R == NORMAL_AR
assert FLOW_R_TO_AR == TUMOR_AR


# =========================
# 2. 颜色与样式
# =========================
BAR_R = "#ddb49b"        # R 柱子（浅橙）
BAR_AR = "#b8c6dc"       # AR 柱子（浅蓝）

FLOW_R = "#e8c8b7"       # 所有从 R 出发的流带都用浅橙
FLOW_AR = "#c8d6e8"      # 所有从 AR 出发的流带都用浅蓝

BAR_EDGE = "#555b66"     # 柱子描边
SEP_COLOR = "white"      # R / AR 分隔线
TEXT_COLOR = "#111111"

TITLE = "Normal vs Tumor"


# =========================
# 3. 工具函数
# =========================
def comma_fmt(x, pos):
    return f"{int(x):,}"


def add_flow(ax, x0, x1, y0_low, y0_high, y1_low, y1_high,
             color, alpha=0.72, curvature=0.35, zorder=1):
    """
    画一条 alluvial / ribbon 流带
    左侧竖边在 x0，右侧竖边在 x1
    左侧上下边界 [y0_low, y0_high]
    右侧上下边界 [y1_low, y1_high]
    """
    cx0 = x0 + curvature * (x1 - x0)
    cx1 = x1 - curvature * (x1 - x0)

    verts = [
        # 上边：左上 -> 右上
        (x0, y0_high),
        (cx0, y0_high),
        (cx1, y1_high),
        (x1, y1_high),

        # 右侧下移
        (x1, y1_low),

        # 下边：右下 -> 左下
        (cx1, y1_low),
        (cx0, y0_low),
        (x0, y0_low),

        # 闭合
        (x0, y0_high)
    ]

    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,

        Path.LINETO,

        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,

        Path.CLOSEPOLY
    ]

    path = Path(verts, codes)
    patch = PathPatch(
        path,
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
        zorder=zorder
    )
    ax.add_patch(patch)


def draw_bar(ax, x_center, width, total_height, r_height, ar_height,
             show_text=True, left_or_right="left"):
    """
    画分段柱子：
    底部 = R
    顶部 = AR
    """
    x_left = x_center - width / 2

    # 底部 R
    rect_r = Rectangle(
        (x_left, 0), width, r_height,
        facecolor=BAR_R, edgecolor=BAR_EDGE, linewidth=2.2, zorder=5
    )
    ax.add_patch(rect_r)

    # 顶部 AR
    rect_ar = Rectangle(
        (x_left, r_height), width, ar_height,
        facecolor=BAR_AR, edgecolor=BAR_EDGE, linewidth=2.2, zorder=5
    )
    ax.add_patch(rect_ar)

    # 中间白色分隔线
    ax.plot(
        [x_left, x_left + width],
        [r_height, r_height],
        color=SEP_COLOR, lw=3.2, zorder=6
    )

    if show_text:
        # R 文字
        ax.text(
            x_center, r_height / 2,
            "R",
            ha="center", va="center",
            fontsize=22, fontweight="bold",
            color=TEXT_COLOR, zorder=7
        )

        # AR 文字
        ax.text(
            x_center, r_height + ar_height / 2,
            "AR",
            ha="center", va="center",
            fontsize=22, fontweight="bold",
            color=TEXT_COLOR, zorder=7
        )


def draw_figure(with_labels=True, out_prefix="FIG4A"):
    """
    生成图
    with_labels=True  : 有字版
    with_labels=False : 无字版
    """
    fig, ax = plt.subplots(figsize=(8.6, 9.3), dpi=300)

    # -------------------------
    # 布局参数
    # -------------------------
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(0, UNION_TOTAL + 120)

    # 柱子更粗
    bar_width = 0.26

    # 左右柱中心
    x_left_bar = 0.0
    x_right_bar = 1.0

    # 流带连接位置：从左柱右边缘，到右柱左边缘
    x0 = x_left_bar + bar_width / 2
    x1 = x_right_bar - bar_width / 2

    # -------------------------
    # 流带坐标（关键）
    # 左边 Normal：
    #   R  = [0, 3708]
    #   AR = [3708, 5434]
    #
    # 右边 Tumor：
    #   R  = [0, 3823]
    #   AR = [3823, 5434]
    # -------------------------

    # 1) R -> R = 2097
    add_flow(
        ax, x0, x1,
        y0_low=0, y0_high=FLOW_R_TO_R,
        y1_low=0, y1_high=FLOW_R_TO_R,
        color=FLOW_R, alpha=0.72, curvature=0.34, zorder=1
    )

    # 2) R -> AR = 1611
    add_flow(
        ax, x0, x1,
        y0_low=FLOW_R_TO_R,
        y0_high=FLOW_R_TO_R + FLOW_R_TO_AR,      # = 3708
        y1_low=TUMOR_R,                          # 3823
        y1_high=TUMOR_R + FLOW_R_TO_AR,          # = 5434
        color=FLOW_R, alpha=0.72, curvature=0.34, zorder=2
    )

    # 3) AR -> R = 1726
    add_flow(
        ax, x0, x1,
        y0_low=NORMAL_R,                         # 3708
        y0_high=NORMAL_R + FLOW_AR_TO_R,        # = 5434
        y1_low=FLOW_R_TO_R,                     # 2097
        y1_high=FLOW_R_TO_R + FLOW_AR_TO_R,     # = 3823
        color=FLOW_AR, alpha=0.72, curvature=0.34, zorder=3
    )

    # -------------------------
    # 画柱子（放在流带之上）
    # -------------------------
    draw_bar(
        ax,
        x_center=x_left_bar,
        width=bar_width,
        total_height=UNION_TOTAL,
        r_height=NORMAL_R,
        ar_height=NORMAL_AR,
        show_text=with_labels,
        left_or_right="left"
    )

    draw_bar(
        ax,
        x_center=x_right_bar,
        width=bar_width,
        total_height=UNION_TOTAL,
        r_height=TUMOR_R,
        ar_height=TUMOR_AR,
        show_text=with_labels,
        left_or_right="right"
    )

    # -------------------------
    # 坐标轴与文字
    # -------------------------
    ax.grid(False)  # 不要网格线

    # 只保留左和下边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.6)
    ax.spines["bottom"].set_linewidth(1.6)

    if with_labels:
        ax.set_title(TITLE, fontsize=24, fontweight="bold", pad=16)
        ax.set_ylabel("Gene count", fontsize=20)

        ax.set_xticks([x_left_bar, x_right_bar])
        ax.set_xticklabels(["Normal", "Tumor"], fontsize=18)

        ax.yaxis.set_major_formatter(FuncFormatter(comma_fmt))
        ax.tick_params(axis="y", labelsize=16, width=1.5, length=6, pad=6)
        ax.tick_params(axis="x", width=1.5, length=6, pad=8)

    else:
        # 无字版：去掉标题、坐标名、刻度标签
        ax.set_title("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    # 输出
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    fig.savefig(f"{out_prefix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.svg", bbox_inches="tight")
    plt.close(fig)


# =========================
# 4. 主程序
# =========================
if __name__ == "__main__":
    outdir = "FIG4A_FINAL_FIXED"

    # 有字版
    draw_figure(
        with_labels=True,
        out_prefix=os.path.join(outdir, "FIG4A_with_labels")
    )

    # 无字版
    draw_figure(
        with_labels=False,
        out_prefix=os.path.join(outdir, "FIG4A_no_labels")
    )

    print("Done.")
    print(f"Outputs saved in: {outdir}")