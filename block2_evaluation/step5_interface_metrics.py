#!/usr/bin/env python3
"""
================================================================
第二模块 / 第五步：综合界面指标计算
================================================================
功能：使用 Rosetta REF2015 能量函数计算每个对接复合物的
     9 项界面质量指标。

计算指标：
  1. total_score     — 复合物 REF2015 总能量（REU）
  2. ddG             — E_复合物 - E_受体 - E_肽链（REU）
  3. dSASA           — 埋藏溶剂可及表面积（Å²）
  4. sc_score        — 形状互补性（0=差，1=完美）
  5. packstat        — 界面堆积密度（0-1）
  6. n_contacts      — 跨链 CA-CA 接触数（≤8Å）
  7. n_hbonds        — 界面氢键数
  8. n_unsatisfied_hb — 未满足氢键供/受体数
  9. hydrophobic_dG  — 界面残基对 fa_atr 能量之和（REU）

输入：docking_results/*_best.pdb（来自第四步）
输出：evaluation_results.csv
     evaluation_metrics.json（完整详情）

用法：
    conda activate pyrose
    python block2_evaluation/step5_interface_metrics.py \
        --docking_dir block2_evaluation/docking_results \
        --output_dir block2_evaluation

运行环境：pyrose（PyRosetta 2026.19）
注意事项：
  - 形状互补性通过 ShapeComplementarityFilter 计算
  - 疏水得分 = 界面处 fa_atr 两体能量之和
  - CA-CA 接触采用 8Å 截断（标准界面定义）
================================================================
"""

import argparse
import os
import sys
import glob
import json
import csv
import numpy as np

CA_CONTACT_CUTOFF = 8.0  # 接触判定距离阈值（Å）


def calc_ca_contacts(pose, chain_a_res: list, chain_b_res: list,
                     cutoff: float = CA_CONTACT_CUTOFF) -> int:
    """统计链 A 与链 B 之间截断距离内的 CA-CA 接触数。"""
    n_contacts = 0
    for ra in chain_a_res:
        try:
            ca_a = pose.residue(ra).xyz("CA")
        except Exception:
            continue
        for rb in chain_b_res:
            try:
                ca_b = pose.residue(rb).xyz("CA")
            except Exception:
                continue
            dist = ((ca_a.x - ca_b.x)**2 +
                    (ca_a.y - ca_b.y)**2 +
                    (ca_a.z - ca_b.z)**2) ** 0.5
            if dist <= cutoff:
                n_contacts += 1
    return n_contacts


def calc_hydrophobic_interface_score(pose, chain_a_res: list, chain_b_res: list,
                                      sfxn) -> float:
    """计算界面处 fa_atr 两体相互作用能之和（疏水得分）。"""
    from pyrosetta.rosetta.core.scoring import fa_atr

    sfxn(pose)
    eg = pose.energies().energy_graph()
    w  = sfxn.weights()

    hydro_total = 0.0
    for ra in chain_a_res:
        for rb in chain_b_res:
            ri, rj = min(ra, rb), max(ra, rb)
            edge = eg.find_edge(ri, rj)
            if edge is None:
                continue
            emap = edge.fill_energy_map()
            hydro_total += w[fa_atr] * emap[fa_atr]
    return round(hydro_total, 3)


def calc_shape_complementarity(pose) -> float:
    """使用 ShapeComplementarityFilter 计算 A_B 界面形状互补性。"""
    try:
        from pyrosetta.rosetta.protocols.simple_filters import ShapeComplementarityFilter
        sc_filter = ShapeComplementarityFilter()
        sc_filter.jump_id(1)
        sc_filter.quick(False)
        sc_val = sc_filter.compute(pose)
        return round(float(sc_val), 4)
    except Exception:
        return -1.0  # 计算失败返回 -1


def score_one_complex(pdb_path: str, sfxn, name: str) -> dict:
    """计算单个复合物 PDB 的全部 9 项界面指标。"""
    import pyrosetta
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

    try:
        pose = pyrosetta.pose_from_pdb(pdb_path)
    except Exception as e:
        return {"name": name, "error": str(e)}

    # 识别链残基
    chain_a_res = [r for r in range(1, pose.total_residue() + 1)
                   if pose.pdb_info().chain(r) == "A"]
    chain_b_res = [r for r in range(1, pose.total_residue() + 1)
                   if pose.pdb_info().chain(r) == "B"]

    if not chain_a_res or not chain_b_res:
        return {"name": name, "error": "无法识别链 A 和链 B"}

    print(f"  [{name}] 链A：{len(chain_a_res)} 残基，链B：{len(chain_b_res)} 残基",
          flush=True)

    # 1. 总能量
    total_score = round(sfxn(pose), 3)
    print(f"  [{name}] total_score={total_score:.2f}", flush=True)

    # 2-5. InterfaceAnalyzerMover 指标
    iam = InterfaceAnalyzerMover("A_B")
    iam.set_compute_packstat(True)
    iam.set_compute_interface_delta_hbond_unsat(True)
    iam.set_scorefunction(sfxn)
    iam.apply(pose)

    interface_dG = round(iam.get_interface_dG(), 3)
    dSASA        = round(iam.get_interface_delta_sasa(), 1)
    packstat     = round(iam.get_interface_packstat(), 4)
    n_hbonds     = int(iam.get_interface_hbonds())
    n_unsat_hb   = int(iam.get_interface_delta_hbond_unsat())
    print(f"  [{name}] interface_dG={interface_dG:.2f} dSASA={dSASA:.0f} "
          f"packstat={packstat:.4f}", flush=True)

    # 6. CA-CA 接触数
    n_contacts = calc_ca_contacts(pose, chain_a_res, chain_b_res)
    print(f"  [{name}] CA 接触数={n_contacts}", flush=True)

    # 7. 形状互补性
    sc_score = calc_shape_complementarity(pose)
    print(f"  [{name}] SC={sc_score:.4f}", flush=True)

    # 8. 疏水得分（界面 fa_atr）
    hydro_dg = calc_hydrophobic_interface_score(pose, chain_a_res, chain_b_res, sfxn)
    print(f"  [{name}] hydrophobic_dG={hydro_dg:.3f}", flush=True)

    # 9. ddG（使用 IAM 界面 dG，比链分割法更可靠）
    ddG = interface_dG
    print(f"  [{name}] ddG={ddG:.3f}", flush=True)

    return {
        "name":             name,
        "pdb":              pdb_path,
        "total_score":      total_score,
        "ddG":              ddG,
        "dSASA":            dSASA,
        "sc_score":         sc_score,
        "packstat":         packstat,
        "n_contacts":       n_contacts,
        "n_hbonds":         n_hbonds,
        "n_unsatisfied_hb": n_unsat_hb,
        "hydrophobic_dG":   hydro_dg,
    }


def main():
    print("=" * 65)
    print("[第二模块 / 第五步] 综合界面指标计算")
    print("  指标：total_score、ddG、dSASA、sc_score、packstat、")
    print("        n_contacts、n_hbonds、n_unsatisfied_hb、hydrophobic_dG")
    print("=" * 65)

    parser = argparse.ArgumentParser(
        description="第五步：计算对接复合物的全面界面指标"
    )
    parser.add_argument("--docking_dir", required=True,
                        help="含 *_best.pdb 的第四步输出目录")
    parser.add_argument("--output_dir", default="block2_evaluation",
                        help="结果输出目录")
    parser.add_argument("--pdb_glob", default="*_best.pdb",
                        help="PDB 文件匹配模式（默认：*_best.pdb）")
    args = parser.parse_args()

    pdb_paths = sorted(glob.glob(os.path.join(args.docking_dir, args.pdb_glob)))
    if not pdb_paths:
        print(f"错误：{args.docking_dir} 中无匹配 {args.pdb_glob} 的 PDB")
        sys.exit(1)

    print(f"  复合物数量：{len(pdb_paths)}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 初始化 PyRosetta（只初始化一次）────────────────────────────────
    try:
        import pyrosetta
        pyrosetta.init("-mute all -ignore_unrecognized_res true -ignore_zero_occupancy false")
        sfxn = pyrosetta.get_fa_scorefxn()
    except ImportError:
        print("错误：未安装 PyRosetta。请 conda activate pyrose")
        sys.exit(1)

    # ── 逐复合物计算指标 ─────────────────────────────────────────────
    all_results = []
    for pdb_path in pdb_paths:
        name = os.path.splitext(os.path.basename(pdb_path))[0]
        name = name.replace("_best", "")
        print(f"\n  处理：{name}")
        result = score_one_complex(pdb_path, sfxn, name)
        all_results.append(result)

    # ── 写 CSV ────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "evaluation_results.csv")
    fieldnames = ["name", "total_score", "ddG", "dSASA", "sc_score",
                  "packstat", "n_contacts", "n_hbonds", "n_unsatisfied_hb",
                  "hydrophobic_dG", "pdb", "error"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    # ── 写 JSON ───────────────────────────────────────────────────────
    json_path = os.path.join(args.output_dir, "evaluation_metrics.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # ── 打印汇总表 ────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"{'名称':<22} {'ddG':>8} {'dSASA':>7} {'SC':>7} {'pack':>7} {'氢键':>5} {'接触':>5}")
    print("-" * 65)
    for r in sorted(all_results, key=lambda x: x.get("ddG", 999)):
        if "error" in r:
            print(f"{r['name']:<22} 错误")
            continue
        print(f"{r['name']:<22} {r['ddG']:>8.3f} {r['dSASA']:>7.0f} "
              f"{r['sc_score']:>7.4f} {r['packstat']:>7.4f} "
              f"{r['n_hbonds']:>5d} {r['n_contacts']:>5d}")

    print(f"\n[第五步完成]")
    print(f"  结果 CSV：{csv_path}")
    print(f"  结果 JSON：{json_path}")
    print(f"  下一步：python block2_evaluation/step6_rank_and_report.py "
          f"--results_csv {csv_path}")


if __name__ == "__main__":
    main()
