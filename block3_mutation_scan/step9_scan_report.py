#!/usr/bin/env python3
"""
================================================================
第三模块 / 第九步：突变扫描分析与报告生成
================================================================
功能：根据丙氨酸扫描（第七步）和饱和突变（第八步）的结果，
     生成可视化热图和汇总 HTML 报告。

输出：
  - alanine_scan_heatmap.png          丙氨酸扫描热图（可重生成）
  - saturation_heatmap_per_pos.png    每热点位点一栏的饱和突变热图
  - saturation_mean_heatmap.png       跨所有肽的均值 ΔΔG 热图
  - mutation_scan_report.html         两轮扫描综合 HTML 报告
  - （top_mutations.csv 已由第八步生成）

用法：
    conda activate pyrose
    python block3_mutation_scan/step9_scan_report.py \
        --scan_dir block3_mutation_scan/scan_results

运行环境：pyrose（需要 pandas + matplotlib，无需 PyRosetta）
================================================================
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ALL_AAS = list("ADEFGHIKLMNPQRSTVWY")  # 19 种氨基酸，与第八步一致


def make_saturation_heatmap_per_pos(df: pd.DataFrame, out_png: str):
    """每个热点位点绘制一栏热图：行=氨基酸，列=肽名，颜色=ΔΔG。"""
    hotspot_positions = sorted(df["position"].unique())
    pep_names = sorted(df["peptide"].unique())
    n_pos = len(hotspot_positions)

    fig, axes = plt.subplots(1, n_pos,
                             figsize=(4.5 * n_pos, max(8, len(ALL_AAS) * 0.5)))
    if n_pos == 1:
        axes = [axes]

    norm = mcolors.TwoSlopeNorm(vmin=-3, vcenter=0, vmax=5)

    for ax, pos in zip(axes, hotspot_positions):
        sub = df[df["position"] == pos].copy()
        mat = np.full((len(ALL_AAS), len(pep_names)), np.nan)

        for j, pep in enumerate(pep_names):
            for i, aa in enumerate(ALL_AAS):
                row = sub[(sub["peptide"] == pep) & (sub["mut_aa"] == aa)]
                if not row.empty:
                    mat[i, j] = row["ddG"].values[0]

        im = ax.imshow(mat, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_yticks(range(len(ALL_AAS)))
        ax.set_yticklabels(list(ALL_AAS), fontsize=7)
        ax.set_xticks(range(len(pep_names)))
        ax.set_xticklabels(pep_names, fontsize=5, rotation=90)

        # 黑色方框标注野生型残基
        for j, pep in enumerate(pep_names):
            wt_row = sub[sub["peptide"] == pep]["wt_aa"]
            if not wt_row.empty:
                wt = wt_row.values[0]
                if wt in ALL_AAS:
                    i = ALL_AAS.index(wt)
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               fill=False, edgecolor="black", lw=1.5))

        # 标注绝对值 >1.0 的数值
        for i in range(len(ALL_AAS)):
            for j in range(len(pep_names)):
                if not np.isnan(mat[i, j]) and abs(mat[i, j]) > 1.0:
                    c = "white" if abs(mat[i, j]) > 2 else "black"
                    ax.text(j, i, f"{mat[i,j]:.1f}", ha="center", va="center",
                            fontsize=4, color=c)

        ax.set_title(f"pos{pos}\n蓝色=改善 红色=破坏", fontsize=8)
        plt.colorbar(im, ax=ax, label="ΔΔG", shrink=0.6)

    plt.suptitle("饱和突变 ΔΔG（两体，REF2015）\n黑框=野生型残基", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  饱和突变热图（每位点）：{out_png}", flush=True)


def make_saturation_mean_heatmap(df: pd.DataFrame, out_png: str):
    """跨所有肽的均值 ΔΔG 热图：行=氨基酸，列=热点位点。"""
    hotspot_positions = sorted(df["position"].unique())
    mean_mat = np.full((len(ALL_AAS), len(hotspot_positions)), np.nan)

    for j, pos in enumerate(hotspot_positions):
        sub = df[df["position"] == pos]
        for i, aa in enumerate(ALL_AAS):
            vals = sub[sub["mut_aa"] == aa]["ddG"]
            if not vals.empty:
                mean_mat[i, j] = round(vals.mean(), 3)

    fig, ax = plt.subplots(figsize=(len(hotspot_positions) * 1.5 + 2, 9))
    norm = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=4)
    im = ax.imshow(mean_mat, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_yticks(range(len(ALL_AAS)))
    ax.set_yticklabels(list(ALL_AAS), fontsize=9)
    ax.set_xticks(range(len(hotspot_positions)))
    ax.set_xticklabels([f"p{p}" for p in hotspot_positions], fontsize=10)
    ax.set_title("跨所有肽的均值 ΔΔG\n（负值 = 有益替换）", fontsize=11)

    for i in range(len(ALL_AAS)):
        for j in range(len(hotspot_positions)):
            if not np.isnan(mean_mat[i, j]):
                c = "white" if abs(mean_mat[i, j]) > 1.5 else "black"
                ax.text(j, i, f"{mean_mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color=c)

    plt.colorbar(im, ax=ax, label="均值 ΔΔG (kcal/mol)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  饱和突变均值热图：{out_png}", flush=True)


def make_html_report(ala_csv: str, top_csv: str,
                     png_paths: list, out_html: str):
    """生成两轮突变扫描综合 HTML 报告。"""
    charts_html = ""
    for png in png_paths:
        if os.path.exists(png):
            rel = os.path.basename(png)
            charts_html += (f'<div style="margin:10px">'
                            f'<img src="{rel}" style="max-width:100%;'
                            f'border:1px solid #ddd">'
                            f'</div>\n')

    # Top 突变表格
    top_table = ""
    if top_csv and os.path.exists(top_csv):
        df_top = pd.read_csv(top_csv)
        if not df_top.empty:
            # 找均值 ΔΔG 最低的 20 行
            ddg_col = [c for c in df_top.columns if "ΔΔG" in c or "ddG" in c.lower()][0]
            df_best = df_top.sort_values(ddg_col).head(20)
            rows = ""
            for _, row in df_best.iterrows():
                # 负值（有益突变）标绿
                ddg_val = row.get(ddg_col, 0)
                try:
                    ddg_float = float(ddg_val)
                    bg = "#e8f5e9" if ddg_float < -0.5 else "#fff"
                except Exception:
                    bg = "#fff"
                cells = "".join(f"<td>{row.get(c, '')}</td>" for c in df_best.columns)
                rows += f'<tr style="background:{bg}">{cells}</tr>\n'
            headers = "".join(f"<th>{c}</th>" for c in df_best.columns)
            top_table = f"""
<h2>Top 20 有益突变</h2>
<table border="1" cellpadding="6" cellspacing="0"
       style="border-collapse:collapse;font-size:12px">
  <tr style="background:#1a237e;color:white">{headers}</tr>
  {rows}
</table>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>GPC3 突变扫描报告</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 20px; color: #333; }}
  h1 {{ color: #1a237e; border-bottom: 2px solid #1a237e; }}
  h2 {{ color: #283593; margin-top: 30px; }}
  .legend {{ background: #f5f5f5; padding: 12px; border-radius: 4px; margin: 16px 0; }}
</style>
</head>
<body>
<h1>GPC3 环肽突变扫描报告</h1>

<div class="legend">
  <strong>第一轮（丙氨酸扫描）</strong>：识别热点位点。<br>
  ΔΔG &gt; 0 = 该位点对结合重要（Ala 减弱相互作用）。<br>
  <strong>第二轮（饱和突变）</strong>：热点位点 × 19 种氨基酸。<br>
  ΔΔG &lt; 0 = 突变优于野生型，增强结合。
</div>

{top_table}

<h2>可视化</h2>
{charts_html}

</body>
</html>
"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML 报告：{out_html}", flush=True)


def main():
    print("=" * 65)
    print("[第三模块 / 第九步] 突变扫描分析与报告生成")
    print("=" * 65)

    parser = argparse.ArgumentParser(
        description="第九步：生成热图和 HTML 突变扫描报告"
    )
    parser.add_argument("--scan_dir", required=True,
                        help="含 alanine_scan_ddg.csv 和 saturation_ddg.csv 的目录")
    parser.add_argument("--output_dir", default=None,
                        help="输出目录（默认：与 scan_dir 相同）")
    args = parser.parse_args()

    out_dir = args.output_dir or args.scan_dir
    os.makedirs(out_dir, exist_ok=True)

    ala_csv = os.path.join(args.scan_dir, "alanine_scan_ddg.csv")
    sat_csv = os.path.join(args.scan_dir, "saturation_ddg.csv")
    top_csv = os.path.join(args.scan_dir, "top_mutations.csv")

    png_paths = []

    # ── 丙氨酸扫描热图 ───────────────────────────────────────────────
    if os.path.exists(ala_csv):
        df_ala = pd.read_csv(ala_csv, index_col=0)
        print(f"  丙氨酸扫描：{ala_csv}（{len(df_ala)} 条肽）")
        ala_png = os.path.join(out_dir, "alanine_scan_heatmap.png")

        ddg_cols = [c for c in df_ala.columns if "_ddG" in c]
        pos_ints = [int(c.replace("pos", "").replace("_ddG", "")) for c in ddg_cols]
        n_peps, n_pos = len(df_ala), len(ddg_cols)
        mat = df_ala[ddg_cols].values

        fig, axes = plt.subplots(1, 2,
                                 figsize=(max(14, n_pos), max(6, n_peps * 0.4)),
                                 gridspec_kw={"width_ratios": [3.5, 1]})
        norm = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=6)
        im = axes[0].imshow(mat, cmap="RdBu_r", norm=norm, aspect="auto")
        axes[0].set_xticks(range(n_pos))
        axes[0].set_xticklabels([f"p{p}" for p in pos_ints], fontsize=8)
        axes[0].set_yticks(range(n_peps))
        axes[0].set_yticklabels(df_ala.index.tolist(), fontsize=6)
        axes[0].set_title("虚拟丙氨酸扫描 ΔΔG\n红色=热点 | 蓝色=Ala 改善", fontsize=10)
        for i in range(n_peps):
            for j in range(n_pos):
                if not np.isnan(mat[i, j]):
                    c = "white" if abs(mat[i, j]) > 2 else "black"
                    axes[0].text(j, i, f"{mat[i,j]:.1f}", ha="center", va="center",
                                 fontsize=5, color=c)
        plt.colorbar(im, ax=axes[0], label="ΔΔG (kcal/mol)")

        mean_ddg = np.nanmean(mat, axis=0)
        bar_colors = ["#d73027" if v > 2 else "#fc8d59" if v > 1
                      else "#4575b4" if v < -0.5 else "#fee090" for v in mean_ddg]
        axes[1].barh(range(n_pos), mean_ddg[::-1], color=bar_colors[::-1],
                     edgecolor="black", linewidth=0.5)
        axes[1].axvline(0, color="black", lw=0.8)
        axes[1].axvline(2.0, color="red", lw=1.2, linestyle="--", label="热点(2.0)")
        axes[1].set_yticks(range(n_pos))
        axes[1].set_yticklabels([f"p{p}" for p in pos_ints[::-1]], fontsize=7)
        axes[1].set_xlabel("均值 ΔΔG")
        axes[1].set_title("位点均值 ΔΔG", fontsize=9)
        axes[1].legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(ala_png, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  丙氨酸热图：{ala_png}")
        png_paths.append(ala_png)
    else:
        print(f"  警告：{ala_csv} 未找到，跳过丙氨酸热图")

    # ── 饱和突变热图 ─────────────────────────────────────────────────
    if os.path.exists(sat_csv):
        df_sat = pd.read_csv(sat_csv)
        print(f"  饱和突变：{sat_csv}（{len(df_sat)} 行）")

        sat_per_pos_png = os.path.join(out_dir, "saturation_heatmap_per_pos.png")
        make_saturation_heatmap_per_pos(df_sat, sat_per_pos_png)
        png_paths.append(sat_per_pos_png)

        sat_mean_png = os.path.join(out_dir, "saturation_mean_heatmap.png")
        make_saturation_mean_heatmap(df_sat, sat_mean_png)
        png_paths.append(sat_mean_png)
    else:
        print(f"  警告：{sat_csv} 未找到，跳过饱和突变热图")

    # ── HTML 报告 ─────────────────────────────────────────────────────
    report_html = os.path.join(out_dir, "mutation_scan_report.html")
    make_html_report(ala_csv, top_csv, png_paths, report_html)

    print(f"\n[第九步完成]")
    print(f"  输出目录：{out_dir}")
    print(f"  HTML 报告：{report_html}")
    print(f"\n  流程完成！请查看：")
    print(f"    block2_evaluation/evaluation_report.html")
    print(f"    block3_mutation_scan/scan_results/mutation_scan_report.html")


if __name__ == "__main__":
    main()
