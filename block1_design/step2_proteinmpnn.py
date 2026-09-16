#!/usr/bin/env python3
"""
================================================================
第一模块 / 第二步：ProteinMPNN 序列设计
================================================================
功能：为 RFdiffusion 生成的骨架设计氨基酸序列。
     链 A（肽链）重新设计；链 B（GPC3）序列固定。

输入：backbone_pdbs/design_*.pdb（来自第一步）
输出：mpnn_seqs/{pdb名}.fa（每个骨架 75 条序列的 FASTA）

用法：
    conda activate SE3nv2
    python block1_design/step2_proteinmpnn.py \
        --pdb_dir block1_design/backbone_pdbs \
        --output_dir block1_design/mpnn_seqs \
        --num_seq 75 \
        --temperature 0.1

运行环境：SE3nv2（与第一步相同，ProteinMPNN 已打包）
注意事项：
  - 设计链 A（肽链），冻结链 B（GPC3 受体）
  - 温度 0.1 = 低多样性、高置信度
  - 序列以 FASTA 格式写出供第三步解析
================================================================
"""

import argparse
import os
import sys
import glob
import json
import subprocess

DEFAULT_NUM_SEQ   = 75     # 每个骨架生成序列数
DEFAULT_TEMP      = "0.1"  # 采样温度
DEFAULT_MODEL     = "v_48_020"


def find_proteinmpnn(script_dir: str) -> str:
    """定位 ProteinMPNN 运行脚本。"""
    candidates = [
        os.path.join(script_dir, "..", "ProteinMPNN", "protein_mpnn_run.py"),
        os.path.expanduser("~/ProteinMPNN/protein_mpnn_run.py"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


def get_chain_ids(pdb_path: str) -> tuple:
    """返回 PDB 的（设计链列表，固定链列表）。

    RFdiffusion 输出：链 A = 肽链（设计），链 B = 受体（固定）。
    """
    chains = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                chains.add(line[21])
    chains = sorted(chains)

    if "A" in chains and "B" in chains:
        designed = ["A"]
        fixed = [c for c in chains if c != "A"]
    else:
        designed = chains
        fixed = []
    return designed, fixed


def write_chain_id_jsonl(pdb_paths: list, out_path: str):
    """写 chain_id.jsonl 供 ProteinMPNN 使用：{名称: ([设计链], [固定链])}。"""
    entries = []
    for p in pdb_paths:
        name = os.path.splitext(os.path.basename(p))[0]
        designed, fixed = get_chain_ids(p)
        entries.append(json.dumps({name: (designed, fixed)}))
    with open(out_path, "w") as f:
        f.write("\n".join(entries) + "\n")


def main():
    print("=" * 65)
    print("[第一模块 / 第二步] ProteinMPNN 序列设计")
    print("  设计：链 A（肽链）| 固定：链 B（GPC3）")
    print("=" * 65)

    parser = argparse.ArgumentParser(
        description="第二步：ProteinMPNN 在 RFdiffusion 骨架上进行序列设计"
    )
    parser.add_argument("--pdb_dir", required=True,
                        help="第一步输出的骨架 PDB 目录")
    parser.add_argument("--output_dir", default="block1_design/mpnn_seqs",
                        help="FASTA 序列输出目录")
    parser.add_argument("--num_seq", type=int, default=DEFAULT_NUM_SEQ,
                        help=f"每个骨架生成序列数（默认：{DEFAULT_NUM_SEQ}）")
    parser.add_argument("--temperature", type=str, default=DEFAULT_TEMP,
                        help=f"采样温度（默认：{DEFAULT_TEMP}）")
    parser.add_argument("--model_name", default=DEFAULT_MODEL,
                        help=f"ProteinMPNN 模型名（默认：{DEFAULT_MODEL}）")
    parser.add_argument("--omit_AAs", default="CX",
                        help="排除的氨基酸（默认：CX，保留 C 用于环化）")
    parser.add_argument("--seed", type=int, default=37,
                        help="随机种子（默认：37）")
    parser.add_argument("--proteinmpnn_dir", default=None,
                        help="ProteinMPNN 目录路径（未设置时自动检测）")
    args = parser.parse_args()

    # ── 定位 PDB 文件 ────────────────────────────────────────────────
    pdb_paths = sorted(glob.glob(os.path.join(args.pdb_dir, "*.pdb")))
    if not pdb_paths:
        print(f"错误：{args.pdb_dir} 中未找到 PDB 文件")
        sys.exit(1)
    print(f"  找到 {len(pdb_paths)} 个骨架 PDB")

    # ── 定位 ProteinMPNN ─────────────────────────────────────────────
    script_dir = os.path.dirname(__file__)
    mpnn_script = find_proteinmpnn(script_dir)
    if args.proteinmpnn_dir:
        mpnn_script = os.path.join(args.proteinmpnn_dir, "protein_mpnn_run.py")
    if mpnn_script is None or not os.path.exists(mpnn_script):
        print("错误：未找到 ProteinMPNN。请设置 --proteinmpnn_dir 或安装至 ../ProteinMPNN/")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 写链 ID JSONL ────────────────────────────────────────────────
    chain_jsonl = os.path.join(args.output_dir, "chain_ids.jsonl")
    write_chain_id_jsonl(pdb_paths, chain_jsonl)
    print(f"  链分配文件：{chain_jsonl}")

    # ── 逐骨架运行 ProteinMPNN ───────────────────────────────────────
    all_fastas = []
    for pdb_path in pdb_paths:
        name = os.path.splitext(os.path.basename(pdb_path))[0]
        designed, fixed = get_chain_ids(pdb_path)
        print(f"\n  处理：{name}")
        print(f"    设计链：{designed}  固定链：{fixed}")

        cmd = [
            sys.executable, mpnn_script,
            "--pdb_path", pdb_path,
            "--pdb_path_chains", " ".join(designed),
            "--out_folder", args.output_dir,
            "--num_seq_per_target", str(args.num_seq),
            "--sampling_temp", args.temperature,
            "--model_name", args.model_name,
            "--omit_AAs", args.omit_AAs,
            "--seed", str(args.seed),
            "--save_score", "0",
        ]

        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"    警告：{name} 的 ProteinMPNN 失败（退出码 {result.returncode}）")
            continue

        fasta_out = os.path.join(args.output_dir, "seqs", f"{name}.fa")
        if os.path.exists(fasta_out):
            all_fastas.append(fasta_out)
            print(f"    输出：{fasta_out}")

    # ── 合并序列供第三步使用 ─────────────────────────────────────────
    combined_fasta = os.path.join(args.output_dir, "all_sequences.fasta")
    n_seqs = 0
    with open(combined_fasta, "w") as fout:
        for fa in all_fastas:
            with open(fa) as fin:
                lines = fin.readlines()
            # 只写入设计序列（跳过原生 header 行）
            for i, line in enumerate(lines):
                if line.startswith(">"):
                    backbone = os.path.splitext(os.path.basename(fa))[0]
                    header_info = line.strip().lstrip(">")
                    fout.write(f">seq_{n_seqs:04d}|{backbone}|{header_info}\n")
                    if i + 1 < len(lines):
                        # 只取链 A 序列（第一个 '/' 之前）
                        seq_line = lines[i + 1].strip()
                        chain_a_seq = seq_line.split("/")[0]
                        fout.write(chain_a_seq + "\n")
                    n_seqs += 1

    print(f"\n[第二步完成] 共设计 {n_seqs} 条序列")
    print(f"  合并 FASTA：{combined_fasta}")
    print(f"  下一步：python block1_design/step3_build_cycpep.py --fasta {combined_fasta}")


if __name__ == "__main__":
    main()
