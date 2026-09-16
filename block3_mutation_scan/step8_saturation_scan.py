#!/usr/bin/env python3
"""
================================================================
第三模块 / 第八步：第二轮 — 全量饱和突变扫描
================================================================
功能：在第七步丙氨酸扫描确定的热点位点上，测试所有 19 种
     氨基酸替换（排除 Cys 以保留环化）。

策略：
  - 与第七步相同的两体 REF2015 能量图方法
  - 不修改骨架，不重新堆积侧链
  - ΔΔG = pairwise_interaction(突变) - pairwise_interaction(WT)
  - 负 ΔΔG = 突变改善结合
  - 每位点 19 种氨基酸（排除 Cys 以保留硫醚环）

并行策略：
  - 基于 subprocess（避免 multiprocessing 管理线程死锁）
  - 文件锁序列化 PyRosetta 初始化，防止 SIGSEGV
  - 默认 10 个工作进程

输入：docking_results/*_best.pdb（来自第四步）
     热点位点（来自第七步输出）
输出：scan_results/saturation_ddg.csv   （完整结果）
     scan_results/pep_results/{名称}.json（每肽结果）
     scan_results/top_mutations.csv    （按均值 ΔΔG 排序）

用法：
    conda activate pyrose
    python block3_mutation_scan/step8_saturation_scan.py \
        --docking_dir block2_evaluation/docking_results \
        --hotspot_positions 2 3 9 11 12 \
        --output_dir block3_mutation_scan/scan_results \
        --n_workers 10

运行环境：pyrose（PyRosetta 2026.19）
注意事项：
  - --worker 参数由子进程内部调用，请勿手动使用
================================================================
"""

import argparse
import os
import sys
import glob
import json
import time
import traceback
import subprocess
import numpy as np
import pandas as pd

DEFAULT_WORKERS = 10

# 氨基酸单字母→三字母映射
ONE_TO_THREE = {
    "A": "ALA", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY",
    "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET",
    "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER",
    "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}
ALL_AAS = list("ADEFGHIKLMNPQRSTVWY")  # 19 种氨基酸，不含 Cys（保留环化）


def pairwise_interaction(pose, pep_resi: int, chain_a: list, sfxn) -> float:
    """计算肽链残基与受体链残基的两体相互作用能之和。"""
    sfxn(pose)
    w  = sfxn.weights()
    eg = pose.energies().energy_graph()
    total = 0.0
    for rec_r in chain_a:
        ri, rj = min(pep_resi, rec_r), max(pep_resi, rec_r)
        edge = eg.find_edge(ri, rj)
        if edge is not None:
            total += w.dot(edge.fill_energy_map())
    return round(total, 3)


def run_worker(name: str, pdb_path: str, hotspot_positions: list, pep_results_dir: str):
    """工作进程：对单条肽的热点位点进行饱和突变扫描。结果写入 JSON。"""
    import fcntl
    import pyrosetta
    from pyrosetta import pose_from_pdb
    from pyrosetta.rosetta.core.scoring import get_score_function
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue

    # 用文件锁序列化 PyRosetta 初始化，防止多进程 SIGSEGV
    lock_path = "/tmp/pyrosetta_saturation_init.lock"
    print(f"[{name}] 等待初始化锁...", flush=True)
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        pyrosetta.init(
            options="-mute all -ignore_unrecognized_res true -ignore_zero_occupancy false",
            silent=True
        )
        print(f"[{name}] PyRosetta 就绪，释放锁", flush=True)
        fcntl.flock(lf, fcntl.LOCK_UN)

    sfxn = get_score_function(True)

    records = []
    try:
        wt_pose  = pose_from_pdb(pdb_path)
        chain_b  = [r for r in range(1, wt_pose.total_residue() + 1)
                    if wt_pose.pdb_info().chain(r) == "B"]
        chain_a  = [r for r in range(1, wt_pose.total_residue() + 1)
                    if wt_pose.pdb_info().chain(r) == "A"]
        if len(chain_b) < 2:
            print(f"[{name}] 警告：链 B 只有 {len(chain_b)} 个残基", flush=True)

        sfxn(wt_pose)  # 预填充能量图

        for pep_pos in hotspot_positions:
            if pep_pos > len(chain_b):
                print(f"[{name}] pos{pep_pos} 超出范围（共 {len(chain_b)} 残基）", flush=True)
                continue

            rosetta_resi = chain_b[pep_pos - 1]
            wt_aa  = wt_pose.residue(rosetta_resi).name1()
            wt_int = pairwise_interaction(wt_pose, rosetta_resi, chain_a, sfxn)

            for mut_aa in ALL_AAS:
                # 野生型氨基酸赋值 0.0
                if mut_aa == wt_aa:
                    records.append({
                        "peptide": name, "position": pep_pos,
                        "wt_aa": wt_aa, "mut_aa": mut_aa, "ddG": 0.0
                    })
                    continue

                mut_pose = wt_pose.clone()
                MutateResidue(rosetta_resi, ONE_TO_THREE[mut_aa]).apply(mut_pose)
                mut_int  = pairwise_interaction(mut_pose, rosetta_resi, chain_a, sfxn)
                ddG      = round(mut_int - wt_int, 3)

                records.append({
                    "peptide": name, "position": pep_pos,
                    "wt_aa": wt_aa, "mut_aa": mut_aa, "ddG": ddG
                })
                print(f"[{name}] pos{pep_pos}({wt_aa}→{mut_aa})："
                      f"wt={wt_int:+.2f} mut={mut_int:+.2f} ΔΔG={ddG:+.2f}", flush=True)

    except Exception as e:
        print(f"[{name}] 错误：{e}", flush=True)
        traceback.print_exc()

    os.makedirs(pep_results_dir, exist_ok=True)
    out_file = os.path.join(pep_results_dir, f"{name}.json")
    with open(out_file, "w") as f:
        json.dump(records, f)
    print(f"[{name}] 完成 → {out_file}", flush=True)


def main_coordinator(args):
    """协调器：派发子进程工作，收集结果。"""
    pdb_paths = sorted(glob.glob(os.path.join(args.docking_dir, args.pdb_glob)))
    if not pdb_paths:
        print(f"错误：{args.docking_dir} 中无匹配 {args.pdb_glob} 的 PDB")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    pep_results_dir = os.path.join(args.output_dir, "pep_results")
    os.makedirs(pep_results_dir, exist_ok=True)

    tasks = []
    for pdb_path in pdb_paths:
        name = os.path.splitext(os.path.basename(pdb_path))[0].replace("_best", "")
        tasks.append((name, pdb_path))

    n_calcs = len(tasks) * len(args.hotspot_positions) * len(ALL_AAS)
    print(f"\n  肽链数：{len(tasks)}")
    print(f"  热点位点：{args.hotspot_positions}")
    print(f"  氨基酸数：{len(ALL_AAS)}（不含 Cys）")
    print(f"  总计算量：{n_calcs} 个突变")
    print(f"  并行进程：{args.n_workers}")

    python_exe  = sys.executable
    script_path = os.path.abspath(__file__)

    t0 = time.time()
    done, running = 0, {}
    task_iter = iter(tasks)
    exhausted = False

    def fill_slots():
        nonlocal exhausted
        while len(running) < args.n_workers and not exhausted:
            try:
                name, pdb_path = next(task_iter)
            except StopIteration:
                exhausted = True
                break
            # 删除旧结果文件
            old_out = os.path.join(pep_results_dir, f"{name}.json")
            if os.path.exists(old_out):
                os.remove(old_out)
            hs_arg = list(map(str, args.hotspot_positions))
            proc = subprocess.Popen(
                [python_exe, "-u", script_path, "--worker", name, pdb_path,
                 "--hotspot_positions"] + hs_arg +
                ["--pep_results_dir", pep_results_dir],
                stdout=sys.stdout, stderr=sys.stderr
            )
            running[proc.pid] = (name, proc)
            print(f"  → 启动工作进程 PID={proc.pid}：{name}", flush=True)

    fill_slots()

    while running:
        time.sleep(2)
        for pid in list(running.keys()):
            name, proc = running[pid]
            ret = proc.poll()
            if ret is not None:
                del running[pid]
                done += 1
                ela = (time.time() - t0) / 60
                eta = (ela / done) * (len(tasks) - done) if done > 1 else 0
                print(f"  完成 {done}/{len(tasks)}（{ela:.1f}分钟，预计还需 {eta:.1f}分钟）",
                      flush=True)
                if ret != 0:
                    print(f"  警告：{name} 退出码 {ret}", flush=True)
                fill_slots()

    # ── 汇总结果 ──────────────────────────────────────────────────────
    all_records = []
    for name, _ in tasks:
        result_file = os.path.join(pep_results_dir, f"{name}.json")
        if os.path.exists(result_file):
            with open(result_file) as f:
                all_records.extend(json.load(f))
        else:
            print(f"  警告：{name} 无结果文件", flush=True)

    df = pd.DataFrame(all_records)
    csv_out = os.path.join(args.output_dir, "saturation_ddg.csv")
    df.to_csv(csv_out, index=False)
    print(f"\n  saturation_ddg.csv：{len(df)} 行")

    # ── 最优突变统计 ──────────────────────────────────────────────────
    top_rows = []
    for pos in args.hotspot_positions:
        sub = df[df["position"] == pos]
        for aa in ALL_AAS:
            vals = sub[sub["mut_aa"] == aa]["ddG"]
            if vals.empty:
                continue
            top_rows.append({
                "位点":      pos,
                "突变氨基酸": aa,
                "均值_ΔΔG":  round(vals.mean(), 3),
                "最小_ΔΔG":  round(vals.min(), 3),
                "改善肽数":  int((vals < -0.3).sum()),
                "破坏肽数":  int((vals > 1.0).sum()),
            })
    df_top = (pd.DataFrame(top_rows)
              .sort_values("均值_ΔΔG")
              .reset_index(drop=True))
    top_csv = os.path.join(args.output_dir, "top_mutations.csv")
    df_top.to_csv(top_csv, index=False)

    print("\n  === 最有益突变 Top 15（均值 ΔΔG 最低）===")
    print(df_top.head(15).to_string(index=False))
    print("\n  === 关键热点突变 Top 10（均值 ΔΔG 最高）===")
    print(df_top.tail(10).sort_values("均值_ΔΔG", ascending=False).to_string(index=False))

    ela_total = round((time.time() - t0) / 60, 1)
    print(f"\n[第八步完成] 耗时：{ela_total} 分钟")
    print(f"  saturation_ddg.csv：{csv_out}")
    print(f"  top_mutations.csv：{top_csv}")
    print(f"  下一步：python block3_mutation_scan/step9_scan_report.py "
          f"--scan_dir {args.output_dir}")


def main():
    print("=" * 65)
    print("[第三模块 / 第八步] 第二轮：饱和突变扫描")
    print("  方法：热点位点 × 19 种氨基酸 REF2015 两体 ΔΔG")
    print("=" * 65)

    parser = argparse.ArgumentParser(
        description="第八步：热点位点饱和突变扫描"
    )
    parser.add_argument("--docking_dir", default=None,
                        help="含 *_best.pdb 的对接结果目录（协调器模式）")
    parser.add_argument("--hotspot_positions", type=int, nargs="+",
                        default=[2, 3, 9, 11, 12],
                        help="来自第七步的热点位点（默认：2 3 9 11 12）")
    parser.add_argument("--output_dir", default="block3_mutation_scan/scan_results",
                        help="输出目录")
    parser.add_argument("--pdb_glob", default="*_best.pdb",
                        help="输入 PDB 匹配模式")
    parser.add_argument("--n_workers", type=int, default=DEFAULT_WORKERS,
                        help=f"子进程工作数（默认：{DEFAULT_WORKERS}）")
    # 子进程内部参数（不对外暴露）
    parser.add_argument("--worker", nargs=2, metavar=("NAME", "PDB_PATH"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--pep_results_dir", default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        # 作为单条肽的子进程运行
        name, pdb_path = args.worker
        run_worker(name, pdb_path, args.hotspot_positions,
                   args.pep_results_dir or "/tmp/pep_results")
    else:
        if args.docking_dir is None:
            print("错误：协调器模式需要 --docking_dir")
            sys.exit(1)
        main_coordinator(args)


if __name__ == "__main__":
    main()
