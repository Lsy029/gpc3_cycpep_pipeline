#!/usr/bin/env python3
"""
================================================================
GPC3 环肽流程 — 端到端演示案例
================================================================
使用 5 条预设计的 CIAc-W 环肽，演示完整的评估 + 突变扫描流程。

跳过第一模块（RFdiffusion + ProteinMPNN，需要 GPU）。
运行第二模块（评估）+ 第三模块第七步（丙氨酸扫描）。

预计运行时间：CPU 约 15-25 分钟
运行要求：pyrose conda 环境（PyRosetta 2026.19）

用法：
    conda activate pyrose
    cd /path/to/gpc3_pipeline
    python demo/demo.py

    # 使用自定义序列：
    python demo/demo.py --fasta demo/data/test_sequences.fasta

    # 使用真实 GPC3 结构（AlphaFold P51654）：
    python demo/demo.py --receptor /path/to/gpc3_af2.pdb

输出文件：
    demo/output/
        cyclic_pdbs/         3D 环肽结构
        docking_results/     最优对接姿态
        evaluation_results.csv
        ranked_candidates.csv
        evaluation_report.html
        scan_results/
            alanine_scan_ddg.csv
            hotspot_summary.csv
            alanine_scan_heatmap.png
            mutation_scan_report.html
================================================================
"""

import argparse
import os
import sys
import subprocess
import glob

DEMO_DIR      = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR  = os.path.dirname(DEMO_DIR)
DEMO_OUT_DIR  = os.path.join(DEMO_DIR, "output")
TEST_FASTA    = os.path.join(DEMO_DIR, "data", "test_sequences.fasta")
TEST_RECEPTOR = os.path.join(DEMO_DIR, "data", "gpc3_sample.pdb")


def run_step(label: str, cmd: list, log_path: str, cwd: str = None) -> bool:
    """Run a pipeline step, tee output to log file, return success."""
    print(f"\n{'='*60}")
    print(f" {label}")
    print(f"{'='*60}")
    print(f"  Command: {' '.join(cmd[:3])} ...")
    print(f"  Log: {log_path}\n")

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as log_fh:
        result = subprocess.run(
            cmd, cwd=cwd or PIPELINE_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        output = result.stdout.decode(errors="replace")
        log_fh.write(output)
        print(output[-3000:] if len(output) > 3000 else output)  # tail output

    if result.returncode != 0:
        print(f"\n  ERROR: {label} failed (exit code {result.returncode})")
        print(f"  See full log: {log_path}")
        return False
    return True


def check_pyrosetta() -> bool:
    """验证 PyRosetta 是否可导入。"""
    try:
        import pyrosetta
        return True
    except ImportError:
        return False


def generate_minimal_receptor_pdb(out_path: str):
    """写入最小化合成 GPC3 结合域 PDB 用于演示。

    链 A 包含 25 个聚丙氨酸残基（受体占位符）。
    注意：仅用于演示，实际研究请使用真实 GPC3 AlphaFold 结构。
    """
    AA_BACKBONE = [
        # Minimal Ala backbone: N, CA, C, O
        ("N",  [-1.458, 0.000, 0.000]),
        ("CA", [0.000,  0.000, 0.000]),
        ("CB", [0.522,  1.426, 0.000]),
        ("C",  [0.773, -0.712, 1.199]),
        ("O",  [0.204, -1.749, 1.408]),
    ]
    lines = ["REMARK  DEMO synthetic GPC3 receptor (polyAla placeholder)\n"]
    atom_idx = 1
    for res_idx in range(1, 26):  # 25 residues
        for atom_name, base_xyz in AA_BACKBONE:
            x = base_xyz[0] + (res_idx - 1) * 3.8
            y = base_xyz[1]
            z = base_xyz[2]
            line = (f"ATOM  {atom_idx:5d}  {atom_name:<3s} ALA A{res_idx:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           "
                    f"{'N' if atom_name=='N' else 'C' if atom_name in ('CA','CB','C') else 'O'}\n")
            lines.append(line)
            atom_idx += 1
    lines.append("TER\nEND\n")
    with open(out_path, "w") as f:
        f.writelines(lines)


def main():
    print("=" * 60)
    print(" GPC3 环肽流程 — 端到端演示")
    print(" 覆盖范围：第三步（构建）、第二模块（完整）、第七步（丙氨酸扫描）")
    print("=" * 60)

    parser = argparse.ArgumentParser(
        description="端到端演示：评估 5 条 GPC3 候选环肽"
    )
    parser.add_argument("--fasta", default=TEST_FASTA,
                        help=f"FASTA with test sequences (default: {TEST_FASTA})")
    parser.add_argument("--receptor", default=None,
                        help="GPC3 receptor PDB (default: synthetic placeholder)")
    parser.add_argument("--output_dir", default=DEMO_OUT_DIR,
                        help=f"Output directory (default: {DEMO_OUT_DIR})")
    parser.add_argument("--n_decoys", type=int, default=5,
                        help="Decoys per peptide for demo (default: 5, use 50 for real)")
    parser.add_argument("--n_workers", type=int, default=3,
                        help="Parallel workers (default: 3)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")

    # ── Check environment ─────────────────────────────────────────────
    if not check_pyrosetta():
        print("\n错误：未找到 PyRosetta。")
        print("  安装：conda activate pyrose")
        print("  或：  pip install pyrosetta-2026*.whl（需要许可证）")
        sys.exit(1)
    print("\n  PyRosetta：OK")

    if not os.path.exists(args.fasta):
        print(f"错误：FASTA 未找到：{args.fasta}")
        sys.exit(1)
    print(f"  FASTA：{args.fasta}")

    # ── 受体设置 ──────────────────────────────────────────────────────
    receptor_pdb = args.receptor
    if receptor_pdb is None:
        receptor_pdb = os.path.join(args.output_dir, "synthetic_receptor.pdb")
        if not os.path.exists(receptor_pdb):
            print(f"\n  未提供受体，生成合成占位符...")
            print(f"  注意：实际研究请使用真实 GPC3 AF2 结构！")
            generate_minimal_receptor_pdb(receptor_pdb)
            print(f"  合成受体：{receptor_pdb}")
    print(f"  受体：{receptor_pdb}")

    py = sys.executable

    # ================================================================
    # BLOCK 1 / STEP 3: Build cyclic peptides from FASTA
    # ================================================================
    cyclic_dir = os.path.join(args.output_dir, "cyclic_pdbs")
    ok = run_step(
        "BLOCK 1 / STEP 3: Build CIAc-W Cyclic Peptides",
        [py, os.path.join(PIPELINE_DIR, "block1_design", "step3_build_cycpep.py"),
         "--fasta", args.fasta,
         "--output_dir", cyclic_dir,
         "--n_confs", "10"],
        os.path.join(log_dir, "step3_build.log")
    )
    if not ok:
        print("WARN: Build step failed. Continuing with any available PDBs...")

    built_pdbs = glob.glob(os.path.join(cyclic_dir, "*.pdb"))
    print(f"\n  Built {len(built_pdbs)} cyclic peptide PDBs")
    if not built_pdbs:
        print("ERROR: No cyclic PDBs built. Check RDKit installation.")
        sys.exit(1)

    # ================================================================
    # BLOCK 2 / STEP 4: Rosetta Docking
    # ================================================================
    docking_dir = os.path.join(args.output_dir, "docking_results")
    ok = run_step(
        f"BLOCK 2 / STEP 4: Rosetta DockMCM Docking ({args.n_decoys} decoys)",
        [py, os.path.join(PIPELINE_DIR, "block2_evaluation", "step4_rosetta_dock.py"),
         "--pep_dir", cyclic_dir,
         "--receptor", receptor_pdb,
         "--output_dir", docking_dir,
         "--n_decoys", str(args.n_decoys),
         "--n_workers", str(args.n_workers)],
        os.path.join(log_dir, "step4_dock.log")
    )
    if not ok:
        print("ERROR: Docking failed.")
        sys.exit(1)

    # ================================================================
    # BLOCK 2 / STEP 5: Interface Metrics
    # ================================================================
    ok = run_step(
        "BLOCK 2 / STEP 5: Interface Metrics (9 metrics)",
        [py, os.path.join(PIPELINE_DIR, "block2_evaluation", "step5_interface_metrics.py"),
         "--docking_dir", docking_dir,
         "--output_dir", args.output_dir],
        os.path.join(log_dir, "step5_metrics.log")
    )
    if not ok:
        print("WARN: Metrics step had errors. Continuing...")

    # ================================================================
    # BLOCK 2 / STEP 6: Ranking and Report
    # ================================================================
    results_csv = os.path.join(args.output_dir, "evaluation_results.csv")
    if os.path.exists(results_csv):
        run_step(
            "BLOCK 2 / STEP 6: Ranking and HTML Report",
            [py, os.path.join(PIPELINE_DIR, "block2_evaluation", "step6_rank_and_report.py"),
             "--results_csv", results_csv,
             "--output_dir", args.output_dir],
            os.path.join(log_dir, "step6_rank.log")
        )

    # ================================================================
    # BLOCK 3 / STEP 7: Alanine Scan
    # ================================================================
    scan_dir = os.path.join(args.output_dir, "scan_results")
    docked_pdbs = glob.glob(os.path.join(docking_dir, "*_best.pdb"))
    if docked_pdbs:
        ok = run_step(
            "BLOCK 3 / STEP 7: Round 1 — Virtual Alanine Scan",
            [py, os.path.join(PIPELINE_DIR, "block3_mutation_scan", "step7_alanine_scan.py"),
             "--docking_dir", docking_dir,
             "--output_dir", scan_dir,
             "--n_workers", str(min(args.n_workers, len(docked_pdbs)))],
            os.path.join(log_dir, "step7_alanine.log")
        )

        # ================================================================
        # BLOCK 3 / STEP 9: Scan Report (skip step8 for demo brevity)
        # ================================================================
        if os.path.exists(os.path.join(scan_dir, "alanine_scan_ddg.csv")):
            run_step(
                "BLOCK 3 / STEP 9: Mutation Scan Report",
                [py, os.path.join(PIPELINE_DIR, "block3_mutation_scan", "step9_scan_report.py"),
                 "--scan_dir", scan_dir],
                os.path.join(log_dir, "step9_report.log")
            )

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 60)
    print(" 演示完成！")
    print("=" * 60)
    print(f"\n  输出目录：{args.output_dir}")
    print("\n  关键输出文件：")

    outputs = [
        (os.path.join(args.output_dir, "ranked_candidates.csv"),
         "候选肽综合得分排名表"),
        (os.path.join(args.output_dir, "evaluation_report.html"),
         "交互式评估报告（浏览器打开）"),
        (os.path.join(scan_dir, "hotspot_summary.csv"),
         "丙氨酸扫描热点位点汇总"),
        (os.path.join(scan_dir, "alanine_scan_heatmap.png"),
         "丙氨酸扫描热图"),
        (os.path.join(scan_dir, "mutation_scan_report.html"),
         "突变扫描 HTML 报告"),
    ]

    for path, desc in outputs:
        status = "OK  " if os.path.exists(path) else "缺失"
        print(f"  [{status}] {os.path.basename(path)}：{desc}")

    print(f"\n  在真实 GPC3 上运行完整流程：")
    print(f"    bash run_pipeline.sh --receptor gpc3_af2.pdb --n_designs 15")
    print()


if __name__ == "__main__":
    main()
