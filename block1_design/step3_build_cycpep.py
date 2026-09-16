#!/usr/bin/env python3
"""
================================================================
第一模块 / 第三步：RDKit 环肽 3D 结构构建
================================================================
功能：构建 CIAc-W 环肽的三维结构。
     环化化学：N 端氯乙酰基（CIAc）+
     C 端 Cys 硫醚键闭环（CIAc-W 结构模块）。

输入：含肽链序列的 FASTA 文件（序列须以 'C' 结尾）
输出：cyclic_pdbs/{名称}.pdb  （最优 UFF 构象）
     cyclic_pdbs/{名称}.sdf  （SDF 格式供对接工具使用）
     cyclic_pdbs/build_report.csv  （每条肽的能量及状态）

用法：
    conda activate pyrose
    python block1_design/step3_build_cycpep.py \
        --fasta block1_design/mpnn_seqs/all_sequences.fasta \
        --output_dir block1_design/cyclic_pdbs \
        --n_confs 20

运行环境：pyrose（RDKit 可通过 conda-forge 安装）或 SE3nv2
注意事项：
  - 序列须以 'C'（Cys）结尾才能进行硫醚环化
  - 不以 C 结尾的序列可通过 --auto_add_cys 自动补加
  - UFF 力场对 20 个构象进行优化，选取能量最低的
  - CIAc = 氯乙酰基（Cl-CH2-CO-），连接至 N 端
================================================================
"""

import argparse
import os
import sys
import csv
from typing import List, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    print("错误：未安装 RDKit。运行：conda install -c conda-forge rdkit")
    sys.exit(1)

# ── RDKit 辅助函数（改编自 build_cyclic_peptides.py）────────────────

def safe_name(seq: str) -> str:
    """生成文件安全名称（仅保留字母数字及 _ -）。"""
    return "".join(ch for ch in seq if ch.isalnum() or ch in ("_", "-"))


def find_nterm_n_idx(pept: Chem.Mol) -> int:
    """定位 N 端主链氮原子（只有一个重原子邻居）。"""
    candidates = []
    for atom in pept.GetAtoms():
        if atom.GetSymbol() != "N":
            continue
        heavy_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if len(heavy_nbrs) == 1:
            candidates.append(atom.GetIdx())
    if candidates:
        return candidates[0]
    for atom in pept.GetAtoms():
        if atom.GetSymbol() == "N":
            return atom.GetIdx()
    raise ValueError("未找到 N 端氮原子")


def find_last_cys_sg_idx(pept: Chem.Mol, seq: str) -> int:
    """定位 C 端 Cys 的 SG 原子用于硫醚闭环。"""
    sulfur_atoms = [a.GetIdx() for a in pept.GetAtoms() if a.GetSymbol() == "S"]
    if not sulfur_atoms:
        raise ValueError("分子中未找到硫原子（环化需要 Cys）")

    if seq.count("C") == 1 and seq.endswith("C"):
        return sulfur_atoms[-1]

    # 多 Cys 序列：尝试 PDB 残基信息
    cys_sg_candidates = []
    for atom in pept.GetAtoms():
        if atom.GetSymbol() != "S":
            continue
        info = atom.GetPDBResidueInfo()
        if info is not None:
            res_name = info.GetResidueName().strip()
            atom_name = (info.GetName().strip() if info.GetName() else "")
            if res_name == "CYS" and atom_name in ("SG", "S", ""):
                cys_sg_candidates.append(atom.GetIdx())

    return cys_sg_candidates[-1] if cys_sg_candidates else sulfur_atoms[-1]


def add_chloroacetyl_to_nterm(pept: Chem.Mol) -> Tuple[Chem.Mol, int, int]:
    """在 N 端添加 CIAc（Cl-CH2-CO-）。返回 (mol, ch2_idx, cl_idx)。"""
    frag = Chem.MolFromSmiles("ClCC(=O)")
    if frag is None:
        raise ValueError("氯乙酰基片段构建失败")

    combo = Chem.CombineMols(pept, frag)
    rw = Chem.RWMol(combo)

    n_pept_atoms = pept.GetNumAtoms()
    frag_cl_idx  = n_pept_atoms + 0   # Cl 原子
    frag_ch2_idx = n_pept_atoms + 1   # CH2 原子
    frag_c_idx   = n_pept_atoms + 2   # C=O 原子

    nterm_n_idx = find_nterm_n_idx(pept)
    rw.AddBond(nterm_n_idx, frag_c_idx, Chem.BondType.SINGLE)

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol, frag_ch2_idx, frag_cl_idx


def cyclize_clac_to_cys(mol: Chem.Mol, ch2_idx: int, cl_idx: int,
                         cys_sg_idx: int) -> Chem.Mol:
    """形成硫醚键：Cys-S 与 CIAc-CH2 成键，删除 Cl 原子。"""
    rw = Chem.RWMol(mol)
    rw.AddBond(cys_sg_idx, ch2_idx, Chem.BondType.SINGLE)
    rw.RemoveAtom(cl_idx)
    cyc = rw.GetMol()
    Chem.SanitizeMol(cyc)
    return cyc


def validate_cyclization(mol: Chem.Mol, seq: str) -> None:
    """验证环化结果：存在环、含硫原子、无残留 Cl。"""
    Chem.SanitizeMol(mol)
    ring_info = mol.GetRingInfo()
    if ring_info is None or ring_info.NumRings() < 1:
        raise ValueError(f"{seq}：环化后未检测到环结构")
    if sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "S") < 1:
        raise ValueError(f"{seq}：环化后未检测到硫原子")
    if sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "Cl") != 0:
        raise ValueError(f"{seq}：环化后仍残留 Cl 原子")


def optimize_best_conformer(mol: Chem.Mol, n_confs: int = 20,
                             seed: int = 20260415,
                             max_iters: int = 1000) -> Tuple[Chem.Mol, int, float]:
    """生成 n_confs 个构象，UFF 优化后返回能量最低的。"""
    molH = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
    params.pruneRmsThresh = 0.5
    params.numThreads = 0  # 使用所有可用 CPU 线程

    conf_ids = list(AllChem.EmbedMultipleConfs(molH, numConfs=n_confs, params=params))
    if not conf_ids:
        raise ValueError("3D 构象生成失败（EmbedMultipleConfs 返回 0）")

    results = []
    for cid in conf_ids:
        try:
            AllChem.UFFOptimizeMolecule(molH, confId=cid, maxIters=max_iters)
            ff = AllChem.UFFGetMoleculeForceField(molH, confId=cid)
            energy = ff.CalcEnergy()
            results.append((cid, energy))
        except Exception:
            continue

    if not results:
        raise ValueError("所有构象的 UFF 优化均失败")

    best_conf_id, best_energy = sorted(results, key=lambda x: x[1])[0]
    return molH, best_conf_id, best_energy


def build_cyclic_peptide(seq: str, n_confs: int = 20,
                          seed: int = 20260415,
                          max_iters: int = 1000) -> Tuple[Chem.Mol, Chem.Mol, int, float]:
    """完整的 CIAc-W 环化流程（单条序列）。

    返回：(二维环状分子, 三维带氢环状分子, 最优构象 ID, 最优能量)
    """
    if not seq.endswith("C"):
        raise ValueError(f"序列须以 C 结尾才能进行 CIAc 环化：{seq}")

    linear = Chem.MolFromSequence(seq)
    if linear is None:
        raise ValueError(f"RDKit 无法解析序列：{seq}")
    Chem.SanitizeMol(linear)

    mol_clac, ch2_idx, cl_idx = add_chloroacetyl_to_nterm(linear)
    cys_sg_idx = find_last_cys_sg_idx(mol_clac, seq)
    cyclic     = cyclize_clac_to_cys(mol_clac, ch2_idx, cl_idx, cys_sg_idx)
    validate_cyclization(cyclic, seq)

    cyc3dH, best_conf_id, best_energy = optimize_best_conformer(
        cyclic, n_confs=n_confs, seed=seed, max_iters=max_iters
    )
    return cyclic, cyc3dH, best_conf_id, best_energy


def parse_fasta(fasta_path: str) -> List[Tuple[str, str]]:
    """解析 FASTA 文件，返回 (header, sequence) 元组列表。"""
    entries = []
    header, seq = None, []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None and seq:
                    entries.append((header, "".join(seq)))
                header = line[1:].split()[0]
                seq = []
            else:
                seq.append(line.upper())
    if header is not None and seq:
        entries.append((header, "".join(seq)))
    return entries


def main():
    print("=" * 65)
    print("[第一模块 / 第三步] RDKit CIAc-W 环肽 3D 构建")
    print("  化学：Cl-CH2-CO-[肽]-C（硫醚键闭环）")
    print("=" * 65)

    parser = argparse.ArgumentParser(
        description="第三步：构建 CIAc-W 环肽的三维结构"
    )
    parser.add_argument("--fasta", required=True,
                        help="输入 FASTA（含肽链序列）")
    parser.add_argument("--output_dir", default="block1_design/cyclic_pdbs",
                        help="PDB/SDF 输出目录")
    parser.add_argument("--n_confs", type=int, default=20,
                        help="每条肽生成的构象数（默认：20）")
    parser.add_argument("--seed", type=int, default=20260415,
                        help="构象生成随机种子")
    parser.add_argument("--max_iters", type=int, default=1000,
                        help="UFF 优化最大迭代次数")
    parser.add_argument("--auto_add_cys", action="store_true",
                        help="自动在不以 C 结尾的序列末尾追加 C")
    args = parser.parse_args()

    if not os.path.exists(args.fasta):
        print(f"错误：FASTA 文件未找到：{args.fasta}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    entries = parse_fasta(args.fasta)
    print(f"  已加载序列数：{len(entries)}")

    report_csv = os.path.join(args.output_dir, "build_report.csv")
    n_ok, n_fail = 0, 0
    pdb_paths = []

    with open(report_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["名称", "序列", "状态", "最优UFF能量",
                         "PDB文件", "SDF文件", "备注"])

        for idx, (header, seq) in enumerate(entries, 1):
            name = f"cyclic_{header}"[:60]
            print(f"\n  [{idx}/{len(entries)}] {name} | {seq}")

            # 自动补 C
            if not seq.endswith("C"):
                if args.auto_add_cys:
                    seq = seq + "C"
                    print(f"    自动追加 C：{seq}")
                else:
                    print(f"    跳过：序列不以 C 结尾（使用 --auto_add_cys）")
                    writer.writerow([name, seq, "已跳过", "", "", "",
                                     "序列不以 C 结尾"])
                    continue

            try:
                cyclic2d, cyc3dH, best_conf_id, best_energy = build_cyclic_peptide(
                    seq, n_confs=args.n_confs, seed=args.seed, max_iters=args.max_iters
                )

                out_pdb = os.path.join(args.output_dir, f"{safe_name(name)}.pdb")
                out_sdf = os.path.join(args.output_dir, f"{safe_name(name)}.sdf")

                # 写 PDB
                block = Chem.MolToPDBBlock(cyc3dH, confId=best_conf_id)
                with open(out_pdb, "w") as f:
                    f.write(f"REMARK {name}\n{block}")

                # 写 SDF
                cyc3dH.SetProp("_Name", name)
                writer_sdf = Chem.SDWriter(out_sdf)
                writer_sdf.write(cyc3dH, confId=best_conf_id)
                writer_sdf.close()

                print(f"    成功  能量={best_energy:.3f}  -> {out_pdb}")
                writer.writerow([name, seq, "成功", f"{best_energy:.6f}",
                                 out_pdb, out_sdf, ""])
                pdb_paths.append(out_pdb)
                n_ok += 1

            except Exception as e:
                print(f"    失败：{e}")
                writer.writerow([name, seq, "失败", "", "", "", str(e)])
                n_fail += 1

    print(f"\n[第三步完成] 构建成功 {n_ok} 个环肽，失败 {n_fail} 个")
    print(f"  输出目录：{args.output_dir}")
    print(f"  构建报告：{report_csv}")

    # 写 PDB 清单供第四步使用
    manifest = os.path.join(args.output_dir, "cyclic_pdb_manifest.txt")
    with open(manifest, "w") as f:
        for p in pdb_paths:
            f.write(p + "\n")
    print(f"  PDB 清单：{manifest}")
    print(f"  下一步：python block2_evaluation/step4_rosetta_dock.py "
          f"--pep_dir {args.output_dir} --receptor <gpc3.pdb>")


if __name__ == "__main__":
    main()
