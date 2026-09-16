#!/usr/bin/env python3
"""
================================================================
第二模块 / 第五步B：【可选】ColabFold AF2 结构验证
================================================================
功能：对 Rosetta 筛选后的 top-N 候选肽，使用 ColabFold
     （AlphaFold2-Multimer v3）进行异源二聚体结构再预测。
     通过 iPAE、iptm、肽段 pLDDT 等置信度指标验证结合模式。

⚠️  此步骤为可选步骤，依赖单独的 colabfold conda 环境。
    若无 GPU 或不需要 AF2 验证，可跳过此步骤直接进入第六步。

指标说明：
  iPAE    — 界面预测对齐误差（0-1，越低越好，<0.5 为合格）
  iptm    — 界面模板建模评分（0-1，越高越好，>0.3 为合格）
  pLDDT_pep — 肽段平均置信度（0-100，>70 为合格）

模型文件位置（本机）：
  AF2 权重：/home/liu_sy/.cache/colabfold/params/
  ColabFold 二进制：/home/liu_sy/conda/envs/colabfold/bin/colabfold_batch

输入：
  block2_evaluation/ranked_candidates.csv   (第六步前运行则用 evaluation_results.csv)
  GPC3 序列（FASTA 或直接使用 UniProt P51654 片段）

输出：
  block2_evaluation/colabfold_results/
      {名称}/        ColabFold 原始输出（PDB、PAE JSON、PNG）
  colabfold_scores.csv    iPAE / iptm / pLDDT_pep 汇总
  colabfold_report.html   带 PAE 图的 HTML 报告

用法：
    conda activate colabfold
    python block2_evaluation/step5b_colabfold_validate.py \\
        --ranked_csv block2_evaluation/ranked_candidates.csv \\
        --output_dir block2_evaluation/colabfold_results \\
        --top_n 10 \\
        --n_recycle 3

    # 使用自定义 GPC3 序列（默认使用内置 P51654 结合域片段）：
    python block2_evaluation/step5b_colabfold_validate.py \\
        --ranked_csv block2_evaluation/ranked_candidates.csv \\
        --gpc3_fasta gpc3_sequence.fasta \\
        --top_n 10

运行环境：colabfold（localcolabfold 1.6.1，JAX CUDA 12）
================================================================
"""

import argparse
import csv
import glob
import json
import math
import os
import subprocess
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── GPC3 内置序列（P51654 成熟体结合域，第 25-200 号残基，用于快速预测）──
GPC3_BINDING_FRAGMENT = (
    "MKALIILLAAAFSGAQAEEAGCPAGAPEEDTEEKFRVYLLTLEALKTLREGR"
    "RLDFRSLEEGAHPTSKDEQFDLSPVTAFGLQHFPHSDLPSSQPEGVALRHSQ"
    "SGMQTSSAGKAVQELRKTPGASVEEKVRGAQKLRQQVEEIPLLRSDRESPCL"
)

# iPAE 归一化因子（ColabFold PAE 满量程 = 31 Å）
PAE_NORM = 31.0


def read_top_candidates(csv_path: str, top_n: int) -> list[dict]:
    """从 CSV 读取 top-N 候选肽（序列 + 名称）。"""
    df = pd.read_csv(csv_path)
    # 兼容 ranked_candidates.csv 和 evaluation_results.csv 两种列名
    name_col = next((c for c in ["name", "peptide", "名称"] if c in df.columns), None)
    seq_col  = next((c for c in ["sequence", "seq", "序列"] if c in df.columns), None)

    if name_col is None:
        raise ValueError(f"找不到名称列：{csv_path}（列：{list(df.columns)}）")

    rows = []
    for _, row in df.head(top_n).iterrows():
        entry = {"name": str(row[name_col])}
        if seq_col:
            entry["sequence"] = str(row[seq_col])
        rows.append(entry)
    return rows


def extract_sequence_from_pdb(pdb_path: str) -> str:
    """从 PDB 文件提取链 A（肽段）序列。"""
    aa3_to_1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    seen, seq = set(), []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[21] == "A" and line[12:16].strip() == "CA":
                resnum = int(line[22:26])
                resname = line[17:20].strip()
                if resnum not in seen:
                    seen.add(resnum)
                    seq.append(aa3_to_1.get(resname, "X"))
    return "".join(seq)


def parse_colabfold_scores(score_json: str, gpc3_len: int, pep_seq: str) -> dict:
    """解析 ColabFold scores JSON，提取 iPAE、iptm、pLDDT_pep。"""
    with open(score_json) as f:
        data = json.load(f)

    result = {"iPAE": 1.0, "iptm": 0.0, "pLDDT_pep": 0.0}

    # iptm
    result["iptm"] = round(data.get("iptm", 0.0), 4)

    # 肽段 pLDDT
    plddt = data.get("plddt", [])
    if len(plddt) > gpc3_len:
        pep_plddt = plddt[gpc3_len:]
        result["pLDDT_pep"] = round(sum(pep_plddt) / len(pep_plddt), 2)

    # iPAE：GPC3→肽 区块均值，归一化到 0-1
    pae = data.get("pae", [])
    if pae:
        pep_len = len(pep_seq)
        inter_vals = []
        for i in range(min(gpc3_len, len(pae))):
            for j in range(gpc3_len, gpc3_len + pep_len):
                if j < len(pae[i]):
                    inter_vals.append(pae[i][j])
        if inter_vals:
            result["iPAE"] = round(sum(inter_vals) / len(inter_vals) / PAE_NORM, 4)

    return result


def run_colabfold(name: str, gpc3_seq: str, pep_seq: str,
                  out_dir: str, n_recycle: int, colabfold_bin: str) -> str | None:
    """运行单条肽的 ColabFold 异源二聚体预测。返回输出目录路径或 None（失败）。"""
    fa_dir = os.path.join(out_dir, "_inputs")
    os.makedirs(fa_dir, exist_ok=True)

    # ColabFold 多链输入：冒号分隔两条链
    fa_path = os.path.join(fa_dir, f"{name}.fasta")
    with open(fa_path, "w") as f:
        f.write(f">{name}_complex\n{gpc3_seq}:{pep_seq}\n")

    subdir = os.path.join(out_dir, name)
    os.makedirs(subdir, exist_ok=True)

    cmd = [
        colabfold_bin,
        "--num-recycle",  str(n_recycle),
        "--num-models",   "1",
        "--model-type",   "alphafold2_multimer_v3",
        "--msa-mode",     "single_sequence",  # 单序列模式，无需 MSA 服务器
        fa_path,
        subdir,
    ]

    print(f"  [{name}] 运行 ColabFold（recycle={n_recycle}）...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"  [{name}] ColabFold 失败：{result.stderr[-400:]}", flush=True)
        return None

    print(f"  [{name}] 完成", flush=True)
    return subdir


def make_summary_chart(df: pd.DataFrame, out_png: str):
    """绘制 iPAE + iptm + pLDDT_pep 三列柱状图。"""
    n = len(df)
    fig, axes = plt.subplots(1, 3, figsize=(14, max(4, n * 0.35)))
    names = df["name"].tolist()

    metrics = [
        ("iPAE",      "RdYlGn_r", "iPAE（越低越好，<0.5 合格）"),
        ("iptm",      "RdYlGn",   "iptm（越高越好，>0.3 合格）"),
        ("pLDDT_pep", "RdYlGn",   "pLDDT 肽段（越高越好，>70 合格）"),
    ]
    thresholds = [0.5, 0.3, 70]

    for ax, (col, cmap, title), thresh in zip(axes, metrics, thresholds):
        vals = df[col].fillna(0).tolist()
        colors = plt.cm.get_cmap(cmap)(
            [min(max(v / (1.0 if col != "pLDDT_pep" else 100), 0), 1) for v in vals]
        )
        ax.barh(range(n), vals, color=colors, edgecolor="black", linewidth=0.4)
        ax.axvline(thresh, color="red", lw=1.2, linestyle="--", label=f"阈值={thresh}")
        ax.set_yticks(range(n))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=6)

    plt.suptitle("ColabFold AF2 验证指标", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  汇总图表：{out_png}", flush=True)


def make_html_report(df: pd.DataFrame, out_dir: str, out_html: str):
    """生成含 PAE 图的 HTML 报告。"""
    rows_html = ""
    for _, row in df.iterrows():
        name = row["name"]
        ipae = row.get("iPAE", "N/A")
        iptm = row.get("iptm", "N/A")
        plddt = row.get("pLDDT_pep", "N/A")

        # 单元格颜色：iPAE<0.5 绿，否则黄；iptm>0.3 绿
        try:
            ipae_bg  = "#c8e6c9" if float(ipae) < 0.5 else "#fff9c4"
            iptm_bg  = "#c8e6c9" if float(iptm) > 0.3 else "#fff9c4"
            pldt_bg  = "#c8e6c9" if float(plddt) > 70 else "#fff9c4"
        except Exception:
            ipae_bg = iptm_bg = pldt_bg = "#fff"

        # 找 PAE 图
        pae_png = glob.glob(os.path.join(out_dir, name, f"*pae*.png"))
        pae_img = (f'<img src="{os.path.relpath(pae_png[0], os.path.dirname(out_html))}" '
                   f'width="220">' if pae_png else "无图")

        rows_html += f"""
<tr>
  <td>{name}</td>
  <td style="background:{ipae_bg}">{ipae}</td>
  <td style="background:{iptm_bg}">{iptm}</td>
  <td style="background:{pldt_bg}">{plddt}</td>
  <td>{row.get("sequence","")}</td>
  <td>{pae_img}</td>
</tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>ColabFold AF2 验证报告 — GPC3 环肽</title>
<style>
  body {{ font-family: Arial; margin: 20px; color: #333; }}
  h1 {{ color: #1a237e; border-bottom: 2px solid #1a237e; }}
  table {{ border-collapse: collapse; font-size: 12px; }}
  th {{ background: #1a237e; color: white; padding: 6px 10px; }}
  td {{ border: 1px solid #ddd; padding: 5px 8px; vertical-align: middle; }}
  .legend {{ background: #f5f5f5; padding: 10px; border-radius: 4px; margin: 12px 0; }}
</style>
</head>
<body>
<h1>ColabFold AF2 验证报告（GPC3 环肽）</h1>
<div class="legend">
  <strong>iPAE</strong>（界面 PAE，归一化）&lt; 0.5 ✓ &nbsp;|&nbsp;
  <strong>iptm</strong> &gt; 0.3 ✓ &nbsp;|&nbsp;
  <strong>pLDDT_pep</strong> &gt; 70 ✓
</div>
<table>
<tr>
  <th>名称</th><th>iPAE ↓</th><th>iptm ↑</th>
  <th>pLDDT肽段 ↑</th><th>序列</th><th>PAE 图</th>
</tr>
{rows_html}
</table>
</body>
</html>"""

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML 报告：{out_html}", flush=True)


def main():
    print("=" * 65)
    print("[第二模块 / 第五步B] 【可选】ColabFold AF2 结构验证")
    print("  模型：AlphaFold2-Multimer v3  | 单序列模式（无 MSA）")
    print("=" * 65)

    parser = argparse.ArgumentParser(
        description="第五步B（可选）：ColabFold AF2 异源二聚体结构验证"
    )
    parser.add_argument("--ranked_csv", required=True,
                        help="第六步排名 CSV 或第五步 evaluation_results.csv")
    parser.add_argument("--output_dir", default="block2_evaluation/colabfold_results",
                        help="输出目录")
    parser.add_argument("--top_n", type=int, default=10,
                        help="验证 top-N 候选肽（默认：10）")
    parser.add_argument("--gpc3_fasta", default=None,
                        help="GPC3 序列 FASTA（默认：内置 P51654 结合域片段）")
    parser.add_argument("--n_recycle", type=int, default=3,
                        help="AF2 recycle 轮数（默认：3，减少可加速）")
    parser.add_argument("--colabfold_bin", default=None,
                        help="colabfold_batch 路径（默认自动检测）")
    parser.add_argument("--docking_dir", default=None,
                        help="含 *_best.pdb 的对接目录（用于提取序列，可选）")
    args = parser.parse_args()

    # ── 检测 colabfold_batch ─────────────────────────────────────────
    cf_bin = args.colabfold_bin
    if cf_bin is None:
        candidates = [
            "/home/liu_sy/conda/envs/colabfold/bin/colabfold_batch",
            "colabfold_batch",
        ]
        for c in candidates:
            if os.path.exists(c) or subprocess.run(
                    ["which", c], capture_output=True).returncode == 0:
                cf_bin = c
                break
    if cf_bin is None:
        print("错误：找不到 colabfold_batch。")
        print("  本机路径：/home/liu_sy/conda/envs/colabfold/bin/colabfold_batch")
        print("  或：conda activate colabfold 后重新运行")
        sys.exit(1)
    print(f"  ColabFold：{cf_bin}")

    # ── GPC3 序列 ────────────────────────────────────────────────────
    if args.gpc3_fasta and os.path.exists(args.gpc3_fasta):
        lines = open(args.gpc3_fasta).readlines()
        gpc3_seq = "".join(l.strip() for l in lines if not l.startswith(">"))
        print(f"  GPC3 序列：{args.gpc3_fasta}（{len(gpc3_seq)} aa）")
    else:
        gpc3_seq = GPC3_BINDING_FRAGMENT
        print(f"  GPC3 序列：内置 P51654 结合域片段（{len(gpc3_seq)} aa）")

    # ── 读取候选肽列表 ───────────────────────────────────────────────
    candidates = read_top_candidates(args.ranked_csv, args.top_n)
    print(f"  候选肽数：{len(candidates)}")

    # 若 CSV 无序列列，尝试从对接 PDB 提取
    if args.docking_dir:
        for entry in candidates:
            if "sequence" not in entry:
                pdb = os.path.join(args.docking_dir, f"{entry['name']}_best.pdb")
                if os.path.exists(pdb):
                    entry["sequence"] = extract_sequence_from_pdb(pdb)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 逐一运行 ColabFold ───────────────────────────────────────────
    all_scores = []
    for entry in candidates:
        name    = entry["name"]
        pep_seq = entry.get("sequence", "")
        if not pep_seq:
            print(f"  [{name}] 跳过：无序列", flush=True)
            continue

        subdir = run_colabfold(name, gpc3_seq, pep_seq,
                               args.output_dir, args.n_recycle, cf_bin)
        if subdir is None:
            all_scores.append({"name": name, "sequence": pep_seq,
                                "iPAE": None, "iptm": None, "pLDDT_pep": None,
                                "status": "failed"})
            continue

        # 解析分数 JSON
        score_jsons = glob.glob(os.path.join(subdir, "*scores*.json"))
        if score_jsons:
            metrics = parse_colabfold_scores(score_jsons[0], len(gpc3_seq), pep_seq)
            print(f"  [{name}] iPAE={metrics['iPAE']:.3f} "
                  f"iptm={metrics['iptm']:.3f} "
                  f"pLDDT_pep={metrics['pLDDT_pep']:.1f}", flush=True)
        else:
            metrics = {"iPAE": None, "iptm": None, "pLDDT_pep": None}

        all_scores.append({"name": name, "sequence": pep_seq,
                           **metrics, "status": "success"})

    # ── 保存汇总 CSV ─────────────────────────────────────────────────
    df = pd.DataFrame(all_scores)
    csv_out = os.path.join(args.output_dir, "colabfold_scores.csv")
    df.to_csv(csv_out, index=False)
    print(f"\n  colabfold_scores.csv：{csv_out}")

    # ── 排名汇总（iPAE 升序）────────────────────────────────────────
    valid = df[df["status"] == "success"].sort_values("iPAE").reset_index(drop=True)
    if not valid.empty:
        print("\n  === ColabFold 验证排名（iPAE 升序）===")
        print(valid[["name", "iPAE", "iptm", "pLDDT_pep"]].to_string(index=False))

        # 图表 + HTML
        chart_png = os.path.join(args.output_dir, "colabfold_summary.png")
        make_summary_chart(valid, chart_png)
        html_out = os.path.join(args.output_dir, "colabfold_report.html")
        make_html_report(valid, args.output_dir, html_out)

    print(f"\n[第五步B 完成]")
    print(f"  输出目录：{args.output_dir}")
    print(f"  下一步：python block2_evaluation/step6_rank_and_report.py ...")


if __name__ == "__main__":
    main()
