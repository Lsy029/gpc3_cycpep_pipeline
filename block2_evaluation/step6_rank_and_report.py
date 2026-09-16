#!/usr/bin/env python3
"""
================================================================
第二模块 / 第六步：候选肽排名与评估报告生成
================================================================
功能：使用多项界面质量指标的综合得分对候选环肽进行排名，
     并生成含表格和柱状图的 HTML 评估报告。

综合得分（越低越好）：
  0.40 × rank(ddG)         — 结合自由能（主要权重）
  0.20 × rank(dSASA)       — 埋藏界面面积
  0.20 × rank(sc_score)    — 形状互补性
  0.20 × rank(packstat)    — 堆积密度

输入：evaluation_results.csv（来自第五步）
输出：ranked_candidates.csv
     evaluation_report.html（交互表格 + 柱状图）

用法：
    conda activate pyrose
    python block2_evaluation/step6_rank_and_report.py \
        --results_csv block2_evaluation/evaluation_results.csv \
        --output_dir block2_evaluation

运行环境：pyrose 或 SE3nv2（需要 pandas + matplotlib）
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

# 各指标权重
WEIGHTS = {
    "ddG":      0.40,
    "dSASA":    0.20,
    "sc_score": 0.20,
    "packstat": 0.20,
}

# ascending=True 表示该指标越低越好
RANK_ASCENDING = {
    "ddG":      True,    # ddG 越低 = 结合越强
    "dSASA":    False,   # dSASA 越高 = 埋藏面积越大 = 越好
    "sc_score": False,   # SC 越高越好
    "packstat": False,   # packstat 越高越好
}


def compute_composite_rank(df: pd.DataFrame) -> pd.DataFrame:
    """计算各指标排名及综合得分，返回排序后的 DataFrame。"""
    valid = df[df["error"].isna() | (df["error"] == "")].copy()

    for metric, ascending in RANK_ASCENDING.items():
        if metric not in valid.columns:
            continue
        valid[f"rank_{metric}"] = valid[metric].rank(ascending=ascending)

    # 加权平均归一化排名
    n = len(valid)
    composite = np.zeros(len(valid))
    for metric, weight in WEIGHTS.items():
        rc = f"rank_{metric}"
        if rc in valid.columns:
            composite += weight * (valid[rc] / n)
    valid["composite_score"] = np.round(composite, 4)
    valid = valid.sort_values("composite_score").reset_index(drop=True)
    valid["final_rank"] = range(1, len(valid) + 1)

    return valid


def make_bar_charts(df: pd.DataFrame, out_dir: str) -> dict:
    """为各关键指标生成柱状图 PNG，返回 {指标: 图片路径} 字典。"""
    png_paths = {}
    # (指标名, Y轴标签, 优化方向说明)
    metrics_to_plot = [
        ("ddG",          "界面 ΔΔG (REU)",         "越低越好"),
        ("dSASA",        "埋藏 SASA (Å²)",          "越高越好"),
        ("sc_score",     "形状互补性",               "越高越好"),
        ("packstat",     "堆积密度",                 "越高越好"),
        ("n_contacts",   "CA-CA 接触数 (≤8Å)",       "越高越好"),
        ("n_hbonds",     "界面氢键数",               "越高越好"),
    ]

    for metric, ylabel, note in metrics_to_plot:
        if metric not in df.columns:
            continue
        sub = df[["name", metric]].dropna().sort_values(metric)
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(max(8, len(sub) * 0.6), 4))
        # 前 25%（蓝色）、后 25%（橙色）、中间（浅蓝）
        colors = ["#2166ac" if v <= sub[metric].quantile(0.25)
                  else "#f4a582" if v >= sub[metric].quantile(0.75)
                  else "#92c5de"
                  for v in sub[metric]]
        ax.bar(range(len(sub)), sub[metric].values, color=colors,
               edgecolor="black", lw=0.5)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["name"].tolist(), rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"{ylabel}（{note}）", fontsize=10)
        ax.axhline(sub[metric].mean(), color="gray", lw=1, linestyle="--", alpha=0.7,
                   label=f"均值={sub[metric].mean():.2f}")
        ax.legend(fontsize=7)
        plt.tight_layout()

        png_path = os.path.join(out_dir, f"chart_{metric}.png")
        plt.savefig(png_path, dpi=120, bbox_inches="tight")
        plt.close()
        png_paths[metric] = png_path

    return png_paths


def make_html_report(df: pd.DataFrame, png_paths: dict, out_path: str):
    """生成含排名表格和嵌入图表的 HTML 评估报告。"""
    cols_display = ["final_rank", "name", "composite_score", "ddG", "dSASA",
                    "sc_score", "packstat", "n_contacts", "n_hbonds",
                    "n_unsatisfied_hb", "hydrophobic_dG"]
    cols_display = [c for c in cols_display if c in df.columns]

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v) if v is not None else ""

    rows_html = ""
    for _, row in df.iterrows():
        rank = int(row.get("final_rank", 0))
        bg = "#e8f5e9" if rank <= 3 else "#fff"  # 前三名标绿
        cells = "".join(f"<td>{fmt(row.get(c,''))}</td>" for c in cols_display)
        rows_html += f'<tr style="background:{bg}">{cells}</tr>\n'

    headers_zh = {
        "final_rank": "排名", "name": "名称", "composite_score": "综合得分",
        "ddG": "ddG", "dSASA": "dSASA", "sc_score": "形状互补性",
        "packstat": "堆积密度", "n_contacts": "CA接触", "n_hbonds": "氢键",
        "n_unsatisfied_hb": "未满足氢键", "hydrophobic_dG": "疏水得分"
    }
    headers = "".join(f"<th>{headers_zh.get(c, c)}</th>" for c in cols_display)

    charts_html = ""
    for metric, png_path in png_paths.items():
        rel_path = os.path.basename(png_path)
        charts_html += (f'<div class="chart">'
                        f'<img src="{rel_path}" style="max-width:100%;'
                        f'border:1px solid #ddd;border-radius:4px">'
                        f'</div>\n')

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>GPC3 环肽评估报告</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 20px; color: #333; }}
  h1 {{ color: #1a237e; border-bottom: 2px solid #1a237e; padding-bottom: 8px; }}
  h2 {{ color: #283593; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
  th {{ background: #1a237e; color: white; padding: 8px 12px; text-align: left; }}
  td {{ padding: 6px 12px; border-bottom: 1px solid #e0e0e0; }}
  tr:hover {{ background: #f5f5f5 !important; }}
  .charts {{ display: flex; flex-wrap: wrap; gap: 20px; margin-top: 20px; }}
  .chart {{ flex: 1 1 45%; }}
  .metric-legend {{ background: #f5f5f5; padding: 12px; border-radius: 4px; margin: 16px 0; }}
  .metric-legend ul {{ margin: 0; padding-left: 20px; }}
</style>
</head>
<body>
<h1>GPC3 环肽评估报告</h1>

<div class="metric-legend">
  <strong>综合得分</strong>（越低越好）：
  <ul>
    <li>权重 40%：界面 ΔΔG（REU）— 结合自由能</li>
    <li>权重 20%：埋藏 SASA（Å²）— 界面面积</li>
    <li>权重 20%：形状互补性（0-1）— 表面贴合度</li>
    <li>权重 20%：堆积密度（0-1）— 界面紧密程度</li>
  </ul>
</div>

<h2>候选肽排名</h2>
<table>
  <thead><tr>{headers}</tr></thead>
  <tbody>{rows_html}</tbody>
</table>

<h2>指标图表</h2>
<div class="charts">
{charts_html}
</div>

</body>
</html>
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    print("=" * 65)
    print("[第二模块 / 第六步] 候选肽排名与报告生成")
    print("  综合得分：40% ddG + 20% dSASA + 20% SC + 20% packstat")
    print("=" * 65)

    parser = argparse.ArgumentParser(
        description="第六步：候选肽排名并生成 HTML 评估报告"
    )
    parser.add_argument("--results_csv", required=True,
                        help="来自第五步的 evaluation_results.csv")
    parser.add_argument("--output_dir", default="block2_evaluation",
                        help="输出目录")
    args = parser.parse_args()

    if not os.path.exists(args.results_csv):
        print(f"错误：{args.results_csv} 未找到")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.results_csv)
    print(f"  已加载 {len(df)} 个候选肽")

    if "error" not in df.columns:
        df["error"] = ""

    # 转换数值列
    numeric_cols = ["total_score", "ddG", "dSASA", "sc_score", "packstat",
                    "n_contacts", "n_hbonds", "n_unsatisfied_hb", "hydrophobic_dG"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 排名
    ranked_df = compute_composite_rank(df)
    print(f"\n  前 5 名候选肽：")
    display_cols = ["final_rank", "name", "composite_score", "ddG", "dSASA", "sc_score"]
    display_cols = [c for c in display_cols if c in ranked_df.columns]
    print(ranked_df[display_cols].head(5).to_string(index=False))

    # 保存排名 CSV
    ranked_csv = os.path.join(args.output_dir, "ranked_candidates.csv")
    ranked_df.to_csv(ranked_csv, index=False)
    print(f"\n  排名 CSV：{ranked_csv}")

    # 生成柱状图
    png_paths = make_bar_charts(ranked_df, args.output_dir)
    print(f"  图表：{list(png_paths.values())}")

    # 生成 HTML 报告
    report_html = os.path.join(args.output_dir, "evaluation_report.html")
    make_html_report(ranked_df, png_paths, report_html)

    print(f"\n[第六步完成]")
    print(f"  排名 CSV：  {ranked_csv}")
    print(f"  HTML 报告：{report_html}")
    print(f"\n  最优候选肽：{ranked_df.iloc[0]['name'] if len(ranked_df) > 0 else '无'}")
    print(f"\n  下一步：python block3_mutation_scan/step7_alanine_scan.py "
          f"--docking_dir block2_evaluation/docking_results")


if __name__ == "__main__":
    main()
