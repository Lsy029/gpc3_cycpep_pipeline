#!/usr/bin/env python3
"""
================================================================
第二模块 / 第四步：Rosetta DockMCM 对接
================================================================
功能：使用 DockMCMProtocol 对环肽与 GPC3 进行高精度局部对接，
     每条肽生成 50 个 decoy，选取结合能最低的姿态。

输入：cyclic_pdbs/*.pdb（来自第三步，链 A = 肽链）
     gpc3.pdb          （受体，链 B = GPC3）
     -- 或预构建的复合物 PDB（链 A=GPC3，链 B=肽链）
输出：docking_results/{名称}_best.pdb   （最优 decoy 结构）
     docking_results/docking_scores.json（所有指标）

用法：
    conda activate pyrose
    python block2_evaluation/step4_rosetta_dock.py \
        --pep_dir block1_design/cyclic_pdbs \
        --receptor gpc3.pdb \
        --output_dir block2_evaluation/docking_results \
        --n_decoys 50 \
        --n_workers 5

运行环境：pyrose（PyRosetta 2026.19）
注意事项：
  - DockMCMProtocol 执行刚体 MCM + 侧链堆积
  - 对接前对界面进行 FastRelax
  - InterfaceAnalyzerMover 计算：dG、dSASA、packstat、氢键
  - 以 5 条肽为一批并行处理
  - GPC3_NRES 用于修正链 B 残基编号从 1 开始
================================================================
"""

import argparse
import os
import sys
import glob
import json
import multiprocessing as mp
import tempfile
import numpy as np

GPC3_HOTSPOT_RESIDUES = [316, 318, 322, 324, 327, 328, 333, 336]
DEFAULT_N_DECOYS  = 50
DEFAULT_ROT_MAG   = 8.0   # 旋转扰动幅度（度）
DEFAULT_TRANS_MAG = 3.0   # 平移扰动幅度（Å）


def fix_chain_b_numbering(src_pdb: str, dst_pdb: str, gpc3_nres: int = 0):
    """重写 PDB 使链 B 残基从 1 开始编号（PyRosetta 要求）。"""
    lines_a, lines_b = [], []
    with open(src_pdb) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            ch = line[21]
            rn = int(line[22:26].strip())
            if ch == "A":
                lines_a.append(line)
            elif ch == "B":
                new_rn = rn - gpc3_nres if gpc3_nres > 0 else rn
                lines_b.append(line[:22] + f"{new_rn:4d}" + line[26:])
    with open(dst_pdb, "w") as fh:
        for l in lines_a:
            fh.write(l)
        fh.write("TER\n")
        for l in lines_b:
            fh.write(l)
        fh.write("TER\nEND\n")


def build_complex_pdb(pep_pdb: str, receptor_pdb: str, out_pdb: str,
                       receptor_chain: str = "A", pep_chain: str = "B",
                       placement_dist: float = 8.0):
    """将受体和肽链合并为单一复合物 PDB。

    将肽链平移至受体质心附近 placement_dist 埃处。
    """
    # 读取受体原子
    rec_lines, rec_coords = [], []
    with open(receptor_pdb) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                rec_lines.append(line[:21] + receptor_chain + line[22:])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                rec_coords.append([x, y, z])

    # 计算受体质心
    centroid = np.mean(rec_coords, axis=0) if rec_coords else np.zeros(3)

    # 读取肽链原子，平移至质心附近
    pep_lines, pep_coords = [], []
    with open(pep_pdb) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                pep_lines.append(line)
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                pep_coords.append([x, y, z])

    if pep_coords:
        pep_centroid = np.mean(pep_coords, axis=0)
        offset = centroid + np.array([placement_dist, 0, 0]) - pep_centroid

        translated_pep = []
        for line in pep_lines:
            x = float(line[30:38]) + offset[0]
            y = float(line[38:46]) + offset[1]
            z = float(line[46:54]) + offset[2]
            new_line = (line[:21] + pep_chain + line[22:30] +
                        f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:])
            translated_pep.append(new_line)
    else:
        translated_pep = [l[:21] + pep_chain + l[22:] for l in pep_lines]

    with open(out_pdb, "w") as f:
        for l in rec_lines:
            f.write(l)
        f.write("TER\n")
        for l in translated_pep:
            f.write(l)
        f.write("TER\nEND\n")


def dock_one_peptide(args_tuple):
    """工作进程：对单条肽执行 DockMCMProtocol。"""
    name, complex_pdb, out_dir, n_decoys, rot_mag, trans_mag = args_tuple

    import pyrosetta
    from pyrosetta.rosetta.protocols.docking import DockMCMProtocol, setup_foldtree
    from pyrosetta.rosetta.protocols.rigid import RigidBodyPerturbMover
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    from pyrosetta.rosetta.utility import vector1_int

    pyrosetta.init("-mute all -ignore_unrecognized_res true -ignore_zero_occupancy false")
    scorefxn  = pyrosetta.get_fa_scorefxn()
    dock_mcm  = DockMCMProtocol()
    dock_mcm.set_scorefxn(scorefxn)
    dock_mcm.set_scorefxn_pack(scorefxn)
    perturber = RigidBodyPerturbMover(1, rot_mag, trans_mag)

    def _load(src):
        tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
        tmp.close()
        fix_chain_b_numbering(src, tmp.name)
        pose = pyrosetta.pose_from_pdb(tmp.name)
        os.unlink(tmp.name)
        return pose

    def _setup_ft(pose):
        v = vector1_int(1)
        v[1] = 1
        setup_foldtree(pose, "A_B", v)

    def _score_interface(pose):
        """运行 InterfaceAnalyzerMover 返回界面指标字典。"""
        iam = InterfaceAnalyzerMover("A_B")
        iam.set_compute_packstat(True)
        iam.set_compute_interface_delta_hbond_unsat(True)
        iam.set_scorefunction(scorefxn)
        iam.apply(pose)
        return {
            "interface_dG":      round(iam.get_interface_dG(), 3),
            "dSASA":             round(iam.get_interface_delta_sasa(), 1),
            "packstat":          round(iam.get_interface_packstat(), 4),
            "delta_hbond_unsat": int(iam.get_interface_delta_hbond_unsat()),
            "n_interface_res":   int(iam.get_num_interface_residues()),
            "complex_energy":    round(iam.get_complex_energy(), 3),
        }

    if not os.path.exists(complex_pdb):
        return name, {"error": f"PDB 未找到：{complex_pdb}"}

    try:
        ref_pose = _load(complex_pdb)
        _setup_ft(ref_pose)
        print(f"  [{name}] 已加载 {ref_pose.total_residue()} 个残基", flush=True)
    except Exception as e:
        return name, {"error": str(e)}

    # 对接前基准得分
    baseline = _score_interface(ref_pose)
    print(f"  [{name}] 基准 dG={baseline['interface_dG']:.2f}", flush=True)

    # 运行 decoy 对接
    decoy_scores, best_pose, best_dG = [], None, float("inf")
    for i in range(n_decoys):
        tp = ref_pose.clone()
        perturber.apply(tp)
        dock_mcm.apply(tp)
        m = _score_interface(tp)
        m["decoy_idx"] = i
        decoy_scores.append(m)
        if m["interface_dG"] < best_dG:
            best_dG = m["interface_dG"]
            best_pose = tp.clone()
        if (i + 1) % 10 == 0:
            dgs = [d["interface_dG"] for d in decoy_scores]
            print(f"  [{name}] decoy {i+1}/{n_decoys}: 最优={min(dgs):.2f} "
                  f"均值={np.mean(dgs):.2f}", flush=True)

    best_m = min(decoy_scores, key=lambda x: x["interface_dG"])
    print(f"  [{name}] 完成 最优dG={best_m['interface_dG']:.3f} "
          f"packstat={best_m['packstat']:.4f}", flush=True)

    out_pdb = os.path.join(out_dir, f"{name}_best.pdb")
    if best_pose:
        best_pose.dump_pdb(out_pdb)

    return name, {
        "baseline":      baseline,
        "best_decoy":    best_m,
        "dG_mean":       round(float(np.mean([d["interface_dG"] for d in decoy_scores])), 3),
        "dG_std":        round(float(np.std([d["interface_dG"] for d in decoy_scores])), 3),
        "packstat_mean": round(float(np.mean([d["packstat"] for d in decoy_scores])), 4),
        "n_decoys":      n_decoys,
        "best_pdb":      out_pdb,
    }


def main():
    print("=" * 65)
    print("[第二模块 / 第四步] Rosetta DockMCM 对接")
    print(f"  协议：DockMCMProtocol × {DEFAULT_N_DECOYS} decoys/肽")
    print("=" * 65)

    parser = argparse.ArgumentParser(
        description="第四步：Rosetta DockMCM 环肽与 GPC3 对接"
    )
    parser.add_argument("--pep_dir", required=True,
                        help="第三步输出的环肽 PDB 目录")
    parser.add_argument("--receptor", required=True,
                        help="GPC3 受体 PDB")
    parser.add_argument("--complex_dir", default=None,
                        help="预构建复合物 PDB 目录（跳过自动构建）")
    parser.add_argument("--output_dir", default="block2_evaluation/docking_results",
                        help="输出目录")
    parser.add_argument("--n_decoys", type=int, default=DEFAULT_N_DECOYS,
                        help=f"每肽 decoy 数（默认：{DEFAULT_N_DECOYS}）")
    parser.add_argument("--n_workers", type=int, default=5,
                        help="并行进程数（默认：5）")
    parser.add_argument("--rot_mag", type=float, default=DEFAULT_ROT_MAG,
                        help=f"旋转扰动幅度（度，默认：{DEFAULT_ROT_MAG}）")
    parser.add_argument("--trans_mag", type=float, default=DEFAULT_TRANS_MAG,
                        help=f"平移扰动幅度（Å，默认：{DEFAULT_TRANS_MAG}）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 定位肽链 PDB ──────────────────────────────────────────────────
    pep_pdbs = sorted(glob.glob(os.path.join(args.pep_dir, "*.pdb")))
    if not pep_pdbs:
        print(f"错误：{args.pep_dir} 中无 PDB 文件")
        sys.exit(1)
    print(f"  肽链数量：{len(pep_pdbs)}")

    # ── 构建或定位复合物 ──────────────────────────────────────────────
    complex_dir = args.complex_dir or os.path.join(args.output_dir, "complexes")
    os.makedirs(complex_dir, exist_ok=True)

    task_args = []
    for pep_pdb in pep_pdbs:
        name = os.path.splitext(os.path.basename(pep_pdb))[0]
        complex_pdb = os.path.join(complex_dir, f"{name}_complex.pdb")
        if not os.path.exists(complex_pdb):
            build_complex_pdb(pep_pdb, args.receptor, complex_pdb)
            print(f"  已构建复合物：{complex_pdb}")
        task_args.append((name, complex_pdb, args.output_dir,
                          args.n_decoys, args.rot_mag, args.trans_mag))

    # ── 分批并行对接 ──────────────────────────────────────────────────
    all_results = {}
    batch_size = args.n_workers
    for i in range(0, len(task_args), batch_size):
        batch = task_args[i:i+batch_size]
        print(f"\n  批次 {i//batch_size + 1}/{(len(task_args)+batch_size-1)//batch_size} "
              f"（{len(batch)} 条肽）...")
        with mp.Pool(processes=len(batch)) as pool:
            batch_results = pool.map(dock_one_peptide, batch)
        all_results.update(dict(batch_results))

    # ── 汇总表 ────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"{'名称':<20} {'最优dG':>10} {'均值dG':>10} {'packstat':>10} {'dSASA':>8}")
    print("-" * 65)

    ranked = []
    for name, v in sorted(all_results.items()):
        if "error" in v:
            print(f"{name:<20} 错误：{v['error'][:40]}")
            continue
        bm = v["best_decoy"]
        print(f"{name:<20} {bm['interface_dG']:>10.3f} {v['dG_mean']:>10.3f} "
              f"{bm['packstat']:>10.4f} {bm['dSASA']:>8.1f}")
        ranked.append((name, bm["interface_dG"]))

    print("\n  按 interface_dG 排名：")
    for i, (n, dg) in enumerate(sorted(ranked, key=lambda x: x[1]), 1):
        print(f"    {i:2d}. {n}: {dg:.3f} REU")

    # ── 保存 JSON ─────────────────────────────────────────────────────
    out_json = os.path.join(args.output_dir, "docking_scores.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n[第四步完成] 结果：{out_json}")
    print(f"  最优 PDB：{args.output_dir}/*_best.pdb")
    print(f"  下一步：python block2_evaluation/step5_interface_metrics.py "
          f"--docking_dir {args.output_dir}")


if __name__ == "__main__":
    main()
