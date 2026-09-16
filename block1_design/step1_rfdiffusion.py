#!/usr/bin/env python3
"""
================================================================
第一模块 / 第一步：RFdiffusion 骨架生成
================================================================
功能：使用 RFdiffusion 生成以 GPC3 结合位点为条件的
     从头设计环肽骨架结构。

输入：GPC3 受体 PDB（链 B，热点残基 316-336）
输出：backbone_pdbs/design_*.pdb  （纯骨架，无侧链）
     backbone_pdbs/design_*.trb  （元数据：pLDDT、contig 映射）

用法：
    conda activate SE3nv2
    python block1_design/step1_rfdiffusion.py \
        --receptor gpc3.pdb \
        --output_dir block1_design/backbone_pdbs \
        --n_designs 15 \
        --pep_length 14 \
        --seed 42

运行环境：SE3nv2（PyTorch 2.7 + CUDA 12.8，RFdiffusion）
注意事项：
  - 需要 GPU（建议显存 ≥24GB 用于完整 GPC3）
  - 须先安装 RFdiffusion：pip install -e RFdiffusion/
  - 热点残基默认为 GPC3 结合口袋 [316,318,322,324,327,328,333,336]
================================================================
"""

import argparse
import os
import sys
import subprocess
import glob
import re

# ── GPC3 结合口袋默认热点残基 ────────────────────────────────────────
GPC3_HOTSPOT_RESIDUES = [316, 318, 322, 324, 327, 328, 333, 336]
DEFAULT_PEP_LENGTH_MIN = 10
DEFAULT_PEP_LENGTH_MAX = 14
RFDIFFUSION_SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "RFdiffusion", "scripts", "run_inference.py"
)


def build_hotspot_string(residues: list, chain: str = "B") -> str:
    """将残基编号列表转为 RFdiffusion 热点格式：B316,B318,..."""
    return ",".join(f"{chain}{r}" for r in residues)


def build_contig_string(receptor_chain: str, receptor_nres: int,
                        pep_len_min: int, pep_len_max: int) -> str:
    """构建 RFdiffusion contigmap 字符串。

    格式：'{链}{1}-{nres}/0 {肽最小长度}-{肽最大长度}'
    受体链固定，肽链长度可变。
    """
    return f"{receptor_chain}1-{receptor_nres}/0 {pep_len_min}-{pep_len_max}"


def count_receptor_residues(pdb_path: str, chain: str = "B") -> int:
    """统计 PDB 文件中指定链的残基数量。"""
    residues = set()
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")) and line[21] == chain:
                resnum = int(line[22:26].strip())
                residues.add(resnum)
    return len(residues)


def main():
    print("=" * 65)
    print("[第一模块 / 第一步] RFdiffusion 骨架生成")
    print("  靶点：GPC3 (P51654) 环肽结合物从头设计")
    print("=" * 65)

    parser = argparse.ArgumentParser(
        description="第一步：使用 RFdiffusion 为 GPC3 环肽结合物生成骨架"
    )
    parser.add_argument("--receptor", required=True,
                        help="GPC3 受体 PDB 路径")
    parser.add_argument("--receptor_chain", default="B",
                        help="PDB 中受体链 ID（默认：B）")
    parser.add_argument("--output_dir", default="block1_design/backbone_pdbs",
                        help="骨架 PDB 输出目录")
    parser.add_argument("--n_designs", type=int, default=15,
                        help="生成骨架数量（默认：15）")
    parser.add_argument("--pep_length", type=int, default=14,
                        help="固定肽链长度；设置后忽略 min/max（默认：14）")
    parser.add_argument("--pep_length_min", type=int, default=None,
                        help="肽链最小长度（未设置 --pep_length 时使用）")
    parser.add_argument("--pep_length_max", type=int, default=None,
                        help="肽链最大长度（未设置 --pep_length 时使用）")
    parser.add_argument("--hotspot_residues", type=int, nargs="+",
                        default=GPC3_HOTSPOT_RESIDUES,
                        help="GPC3 热点残基编号（默认：结合口袋残基）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（默认：42）")
    parser.add_argument("--partial_diffusion", action="store_true",
                        help="使用部分扩散模式进行骨架精修")
    parser.add_argument("--noise_scale", type=float, default=1.0,
                        help="部分扩散噪声尺度（0-1，默认：1.0）")
    parser.add_argument("--rfdiffusion_dir", default=None,
                        help="RFdiffusion 根目录路径（未设置时自动检测）")
    args = parser.parse_args()

    # ── 输入验证 ──────────────────────────────────────────────────────
    if not os.path.exists(args.receptor):
        print(f"错误：受体 PDB 未找到：{args.receptor}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 定位 RFdiffusion ─────────────────────────────────────────────
    rfdiff_dir = args.rfdiffusion_dir
    if rfdiff_dir is None:
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "RFdiffusion"),
            os.path.expanduser("~/RFdiffusion"),
        ]
        for c in candidates:
            if os.path.exists(c):
                rfdiff_dir = os.path.abspath(c)
                break
    if rfdiff_dir is None:
        print("错误：未找到 RFdiffusion。请设置 --rfdiffusion_dir 或安装至 ./RFdiffusion/")
        sys.exit(1)

    inference_script = os.path.join(rfdiff_dir, "scripts", "run_inference.py")
    if not os.path.exists(inference_script):
        print(f"错误：run_inference.py 未找到：{inference_script}")
        sys.exit(1)

    # ── 构建参数 ─────────────────────────────────────────────────────
    receptor_nres = count_receptor_residues(args.receptor, args.receptor_chain)
    print(f"  受体：{args.receptor}（{receptor_nres} 个残基，链 {args.receptor_chain}）")

    if args.pep_length:
        pep_min = pep_max = args.pep_length
    else:
        pep_min = args.pep_length_min or DEFAULT_PEP_LENGTH_MIN
        pep_max = args.pep_length_max or DEFAULT_PEP_LENGTH_MAX

    contig = build_contig_string(args.receptor_chain, receptor_nres, pep_min, pep_max)
    hotspot_str = build_hotspot_string(args.hotspot_residues, args.receptor_chain)

    print(f"  Contigmap：{contig}")
    print(f"  热点残基：{hotspot_str}")
    print(f"  设计数量：{args.n_designs}")
    print(f"  随机种子：{args.seed}")

    output_prefix = os.path.join(args.output_dir, "design")

    # ── 构建 Hydra override 参数 ─────────────────────────────────────
    hydra_overrides = [
        f"inference.input_pdb={args.receptor}",
        f"inference.output_prefix={output_prefix}",
        f"inference.num_designs={args.n_designs}",
        f"inference.deterministic=True",
        f"contigmap.contigs=['{contig}']",
        f"ppi.hotspot_res=[{hotspot_str}]",
    ]

    if args.partial_diffusion:
        hydra_overrides += [
            f"diffuser.partial_T=20",
            f"noise_scale_ca={args.noise_scale}",
            f"noise_scale_frame={args.noise_scale}",
        ]

    cmd = [sys.executable, inference_script] + hydra_overrides

    print(f"\n  执行：{' '.join(cmd[:3])} [... {len(hydra_overrides)} 个覆盖参数]")
    print("  （根据 GPU 速度，可能需要 5-30 分钟...）\n")

    result = subprocess.run(cmd, cwd=rfdiff_dir)

    if result.returncode != 0:
        print(f"\n错误：RFdiffusion 退出码 {result.returncode}")
        sys.exit(result.returncode)

    # ── 输出汇报 ─────────────────────────────────────────────────────
    pdbs = sorted(glob.glob(output_prefix + "_*.pdb"))
    print(f"\n[第一步完成] 已生成 {len(pdbs)} 个骨架 PDB：")
    for p in pdbs:
        print(f"  {p}")

    # 写入文件清单供第二步使用
    manifest_path = os.path.join(args.output_dir, "backbone_manifest.txt")
    with open(manifest_path, "w") as f:
        for p in pdbs:
            f.write(p + "\n")
    print(f"\n  文件清单：{manifest_path}")
    print(f"  下一步：python block1_design/step2_proteinmpnn.py --pdb_dir {args.output_dir}")


if __name__ == "__main__":
    main()
