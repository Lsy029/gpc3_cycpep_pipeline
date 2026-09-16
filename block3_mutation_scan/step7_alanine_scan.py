#!/usr/bin/env python3
"""
================================================================
第三模块 / 第七步：第一轮 — 虚拟丙氨酸扫描
================================================================
功能：通过计算每个位点丙氨酸替换的能量效果，识别候选肽的
     热点残基。

策略：
  - 不修改骨架，不重新堆积侧链
  - 直接读取 REF2015 两体能量图（快速，无需重新对接）
  - ΔΔG = pairwise_interaction(ALA) - pairwise_interaction(WT)
  - 正 ΔΔG = 热点（Ala 减弱结合）
  - 负 ΔΔG = Ala 改善结合

跳过位点：
  - pos1 和 pos14（CIAc-W 环锚点 — Trp 和 Cys）
  - 已为 Ala、Gly、Pro 的残基（赋值 0.0）

输入：docking_results/*_best.pdb（来自第四步）
输出：scan_results/alanine_scan_ddg.csv   （肽×位点 ΔΔG 矩阵）
     scan_results/hotspot_summary.csv    （位点级统计）
     scan_results/alanine_scan_heatmap.png

用法：
    conda activate pyrose
    python block3_mutation_scan/step7_alanine_scan.py \
        --docking_dir block2_evaluation/docking_results \
        --output_dir block3_mutation_scan/scan_results \
        --n_workers 20

运行环境：pyrose（PyRosetta 2026.19）
注意事项：
  - 20 个并行进程（每进程处理一条肽的所有位点）
  - 热点判定阈值：均值 ΔΔG > 0.5 且最大 ΔΔG > 2.0 kcal/mol
  - 热点位点将传入第八步进行饱和突变
================================================================
"""

import argparse
import os
import sys
import glob
import time
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from multiprocessing import Pool

SKIP_POSITIONS = {1, 14}  # CIAc-W 环锚点（pos1=Trp, pos14=Cys）
SKIP_AA        = {"A", "G", "P"}  # 已是 Ala/Gly/Pro 则跳过
DEFAULT_WORKERS    = 20
HOTSPOT_MEAN_THRESH = 0.5   # kcal/mol，热点均值阈值
HOTSPOT_MAX_THRESH  = 2.0   # kcal/mol，热点最大值阈值


def pairwise_interaction(pose, pep_resi: int, chain_a_residues: list, sfxn) -> float:
    """计算肽链残基与受体链所有残基的两体相互作用能之和。"""
    sfxn(pose)
    w      = sfxn.weights()
    egraph = pose.energies().energy_graph()
    total  = 0.0
    for rec_r in chain_a_residues:
        ri, rj = min(pep_resi, rec_r), max(pep_resi, rec_r)
        edge = egraph.find_edge(ri, rj)
        if edge is not None:
            total += w.dot(edge.fill_energy_map())
    return round(total, 3)


def scan_one_peptide(args_tuple):
    """工作进程：对单条肽进行丙氨酸扫描。返回 (名称, {位点: ΔΔG})。"""
    name, pdb_path = args_tuple

    import pyrosetta
    from pyrosetta import pose_from_pdb
    from pyrosetta.rosetta.core.scoring import get_score_function
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue

    pyrosetta.init(
        options="-mute all -ignore_unrecognized_res true -ignore_zero_occupancy false",
        silent=True
    )
    sfxn = get_score_function(True)  # REF2015

    results = {}
    try:
        wt_pose = pose_from_pdb(pdb_path)
        chain_b = [r for r in range(1, wt_pose.total_residue() + 1)
                   if wt_pose.pdb_info().chain(r) == "B"]
        chain_a = [r for r in range(1, wt_pose.total_residue() + 1)
                   if wt_pose.pdb_info().chain(r) == "A"]

        n_pep_res = len(chain_b)
        if n_pep_res < 2:
            print(f"[{name}] 警告：链 B 只有 {n_pep_res} 个残基", flush=True)
            return name, {}

        sfxn(wt_pose)  # 预填充能量图（只需一次）

        for pep_pos in range(1, n_pep_res + 1):
            if pep_pos in SKIP_POSITIONS:
                print(f"[{name}] pos{pep_pos}：环锚点 — 跳过", flush=True)
                continue

            rosetta_resi = chain_b[pep_pos - 1]
            wt_aa = wt_pose.residue(rosetta_resi).name1()

            if wt_aa in SKIP_AA:
                results[pep_pos] = 0.0
                print(f"[{name}] pos{pep_pos}({wt_aa})：跳过（赋值 0.0）", flush=True)
                continue

            # 野生型两体相互作用能
            wt_int = pairwise_interaction(wt_pose, rosetta_resi, chain_a, sfxn)

            # 虚拟 Ala 突变后两体相互作用能（不重新堆积）
            mut_pose = wt_pose.clone()
            MutateResidue(rosetta_resi, "ALA").apply(mut_pose)
            ala_int = pairwise_interaction(mut_pose, rosetta_resi, chain_a, sfxn)

            ddG = round(ala_int - wt_int, 3)
            results[pep_pos] = ddG
            print(f"[{name}] pos{pep_pos}({wt_aa}→A)："
                  f"wt={wt_int:+.2f} ala={ala_int:+.2f} ΔΔG={ddG:+.2f}", flush=True)

    except Exception as e:
        print(f"[{name}] 错误：{e}", flush=True)
        traceback.print_exc()

    return name, results


def make_heatmap(df: pd.DataFrame, out_png: str):
    """绘制 ΔΔG 热图（肽×位点）+ 位点均值柱状图。"""
    pep_names = list(df.index)
    ddg_cols  = [c for c in df.columns if "_ddG" in c]
    pos_ints  = [int(c.replace("pos", "").replace("_ddG", "")) for c in ddg_cols]
    n_peps, n_pos = len(pep_names), len(ddg_cols)
    mat = df[ddg_cols].values

    fig, axes = plt.subplots(1, 2,
                             figsize=(max(14, n_pos * 1.2), max(6, n_peps * 0.4)),
                             gridspec_kw={"width_ratios": [3.5, 1]})
    ax = axes[0]
    norm = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=6)
    im = ax.imshow(mat, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(n_pos))
    ax.set_xticklabels([f"p{p}" for p in pos_ints], fontsize=8)
    ax.set_yticks(range(n_peps))
    ax.set_yticklabels(pep_names, fontsize=7)
    ax.set_title("虚拟丙氨酸扫描 ΔΔG（REF2015 两体）\n"
                 "红色=热点 | 蓝色=Ala 改善结合", fontsize=10)
    for i in range(n_peps):
        for j in range(n_pos):
            if not np.isnan(mat[i, j]):
                c = "white" if abs(mat[i, j]) > 2 else "black"
                ax.text(j, i, f"{mat[i,j]:.1f}", ha="center", va="center",
                        fontsize=5.5, color=c)
    plt.colorbar(im, ax=ax, label="ΔΔG (kcal/mol)")

    ax2 = axes[1]
    mean_ddg = np.nanmean(mat, axis=0)
    bar_colors = ["#d73027" if v > 2.0 else "#fc8d59" if v > 1.0
                  else "#4575b4" if v < -0.5 else "#fee090"
                  for v in mean_ddg]
    ax2.barh(range(n_pos), mean_ddg[::-1], color=bar_colors[::-1],
             edgecolor="black", linewidth=0.5)
    ax2.axvline(0,   color="black", lw=0.8)
    ax2.axvline(2.0, color="red",   lw=1.2, linestyle="--", label="热点(2.0)")
    ax2.axvline(1.0, color="orange", lw=0.8, linestyle=":", label="中等(1.0)")
    ax2.set_yticks(range(n_pos))
    ax2.set_yticklabels([f"p{p}" for p in pos_ints[::-1]], fontsize=7)
    ax2.set_xlabel("均值 ΔΔG (kcal/mol)\n（跨所有肽）")
    ax2.set_title("位点均值 ΔΔG", fontsize=9)
    ax2.legend(fontsize=6)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  热图：{out_png}", flush=True)


def main():
    print("=" * 65)
    print("[第三模块 / 第七步] 第一轮：虚拟丙氨酸扫描")
    print("  方法：REF2015 两体相互作用 ΔΔG（无侧链重堆积）")
    print("=" * 65)

    parser = argparse.ArgumentParser(
        description="第七步：虚拟丙氨酸扫描识别热点位点"
    )
    parser.add_argument("--docking_dir", required=True,
                        help="含 *_best.pdb 的第四步输出目录")
    parser.add_argument("--output_dir", default="block3_mutation_scan/scan_results",
                        help="输出目录")
    parser.add_argument("--pdb_glob", default="*_best.pdb",
                        help="输入 PDB 匹配模式（默认：*_best.pdb）")
    parser.add_argument("--n_workers", type=int, default=DEFAULT_WORKERS,
                        help=f"并行进程数（默认：{DEFAULT_WORKERS}）")
    parser.add_argument("--hotspot_mean_thresh", type=float, default=HOTSPOT_MEAN_THRESH,
                        help=f"热点均值 ΔΔG 阈值（默认：{HOTSPOT_MEAN_THRESH}）")
    parser.add_argument("--hotspot_max_thresh", type=float, default=HOTSPOT_MAX_THRESH,
                        help=f"热点最大 ΔΔG 阈值（默认：{HOTSPOT_MAX_THRESH}）")
    args = parser.parse_args()

    pdb_paths = sorted(glob.glob(os.path.join(args.docking_dir, args.pdb_glob)))
    if not pdb_paths:
        print(f"错误：{args.docking_dir} 中无匹配 {args.pdb_glob} 的 PDB")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"  肽链数量：{len(pdb_paths)}")

    tasks = []
    for pdb_path in pdb_paths:
        name = os.path.splitext(os.path.basename(pdb_path))[0].replace("_best", "")
        tasks.append((name, pdb_path))

    # ── 并行丙氨酸扫描 ────────────────────────────────────────────────
    t0 = time.time()
    all_results = {}
    n_workers = min(args.n_workers, len(tasks))
    print(f"  并行进程数：{n_workers}")

    with Pool(processes=n_workers) as pool:
        for i, (name, res) in enumerate(pool.imap_unordered(scan_one_peptide, tasks), 1):
            all_results[name] = res
            elapsed = (time.time() - t0) / 60
            print(f"  进度：{i}/{len(tasks)}（{elapsed:.1f} 分钟）", flush=True)

    # ── 整理结果 DataFrame ────────────────────────────────────────────
    all_positions = set()
    for res in all_results.values():
        all_positions.update(res.keys())
    all_positions = sorted(all_positions)

    rows = []
    for name, _ in tasks:
        r = {"peptide": name}
        for pos in all_positions:
            r[f"pos{pos}_ddG"] = all_results.get(name, {}).get(pos, np.nan)
        rows.append(r)

    df = pd.DataFrame(rows).set_index("peptide")
    csv_out = os.path.join(args.output_dir, "alanine_scan_ddg.csv")
    df.to_csv(csv_out)
    print(f"\n  丙氨酸扫描 CSV：{csv_out}")
    print(df.to_string())

    # ── 热点位点统计 ──────────────────────────────────────────────────
    hs_rows = []
    for pos in all_positions:
        col  = f"pos{pos}_ddG"
        vals = df[col].dropna()
        is_hotspot = (len(vals) > 0 and
                      vals.mean() > args.hotspot_mean_thresh and
                      vals.max() > args.hotspot_max_thresh)
        hs_rows.append({
            "位点":           pos,
            "均值_ΔΔG":       round(vals.mean(), 3) if len(vals) > 0 else np.nan,
            "最大_ΔΔG":       round(vals.max(), 3)  if len(vals) > 0 else np.nan,
            "最小_ΔΔG":       round(vals.min(), 3)  if len(vals) > 0 else np.nan,
            "热点肽数":        int((vals > 2.0).sum()),
            "中等热点肽数":    int(((vals >= 1.0) & (vals <= 2.0)).sum()),
            "有益肽数":        int((vals < -0.5).sum()),
            "是否热点":        is_hotspot,
        })

    df_hs = pd.DataFrame(hs_rows).sort_values("均值_ΔΔG", ascending=False)
    hs_csv = os.path.join(args.output_dir, "hotspot_summary.csv")
    df_hs.to_csv(hs_csv, index=False)

    hotspot_positions = df_hs[df_hs["是否热点"]]["位点"].tolist()

    print(f"\n  === 热点位点汇总 ===")
    print(df_hs.to_string(index=False))
    print(f"\n  热点位点：{hotspot_positions}")
    print(f"  （阈值：均值>{args.hotspot_mean_thresh}，最大>{args.hotspot_max_thresh}）")

    # ── 生成热图 ──────────────────────────────────────────────────────
    make_heatmap(df, os.path.join(args.output_dir, "alanine_scan_heatmap.png"))

    ela = round((time.time() - t0) / 60, 1)
    print(f"\n[第七步完成] 耗时：{ela} 分钟")
    print(f"  CSV：{csv_out}")
    print(f"  热点汇总：{hs_csv}")
    print(f"  下一步：python block3_mutation_scan/step8_saturation_scan.py "
          f"--docking_dir block2_evaluation/docking_results "
          f"--hotspot_positions {' '.join(map(str, hotspot_positions))}")


if __name__ == "__main__":
    main()
