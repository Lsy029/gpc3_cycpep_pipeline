#!/usr/bin/env bash
# =============================================================================
# GPC3 Cyclic Peptide Pipeline — Master Orchestrator
# =============================================================================
# Usage:
#   Full pipeline:
#     bash run_pipeline.sh --receptor gpc3.pdb --n_designs 15
#
#   Skip design (use existing sequences):
#     bash run_pipeline.sh --skip_design --fasta my_seqs.fasta --receptor gpc3.pdb
#
#   Evaluation + scan only:
#     bash run_pipeline.sh --pep_dir cyclic_pdbs/ --receptor gpc3.pdb
#
# Requirements:
#   conda environments: SE3nv2 (Block 1), pyrose (Block 2-3)
#   PyRosetta academic license required
# =============================================================================

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
RECEPTOR=""
N_DESIGNS=15
PEP_LENGTH=14
N_DECOYS=50
N_MPNN_SEQ=75
N_WORKERS=5
HOTSPOT_POSITIONS="2 3 9 11 12"
SKIP_DESIGN=false
FASTA=""
PEP_DIR=""
SEED=42

LOG_DIR="logs"
BLOCK1_DIR="block1_design"
BLOCK2_DIR="block2_evaluation"
BLOCK3_DIR="block3_mutation_scan/scan_results"

mkdir -p "$LOG_DIR"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --receptor)        RECEPTOR="$2";          shift 2 ;;
    --n_designs)       N_DESIGNS="$2";          shift 2 ;;
    --pep_length)      PEP_LENGTH="$2";         shift 2 ;;
    --n_decoys)        N_DECOYS="$2";           shift 2 ;;
    --n_workers)       N_WORKERS="$2";          shift 2 ;;
    --hotspot_positions) HOTSPOT_POSITIONS="$2"; shift 2 ;;
    --skip_design)     SKIP_DESIGN=true;        shift ;;
    --fasta)           FASTA="$2";              shift 2 ;;
    --pep_dir)         PEP_DIR="$2";            shift 2 ;;
    --seed)            SEED="$2";               shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$RECEPTOR" ]]; then
  echo "ERROR: --receptor <gpc3.pdb> is required"
  exit 1
fi

echo "============================================================"
echo " GPC3 Cyclic Peptide Pipeline"
echo "  Receptor:  $RECEPTOR"
echo "  Designs:   $N_DESIGNS"
echo "  Decoys:    $N_DECOYS"
echo "  Workers:   $N_WORKERS"
echo "  Skip Design: $SKIP_DESIGN"
echo "============================================================"

# =============================================================================
# BLOCK 1: DESIGN
# =============================================================================
if [[ "$SKIP_DESIGN" == "false" ]]; then
  echo ""
  echo "=== BLOCK 1: DESIGN ==="

  # STEP 1: RFdiffusion backbone generation
  echo "[$(date '+%H:%M:%S')] STEP 1: RFdiffusion..."
  conda run -n SE3nv2 --no-capture-output python block1_design/step1_rfdiffusion.py \
    --receptor "$RECEPTOR" \
    --output_dir "$BLOCK1_DIR/backbone_pdbs" \
    --n_designs "$N_DESIGNS" \
    --pep_length "$PEP_LENGTH" \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/step1_rfdiffusion.log"
  echo "[$(date '+%H:%M:%S')] STEP 1 done."

  # STEP 2: ProteinMPNN sequence design
  echo "[$(date '+%H:%M:%S')] STEP 2: ProteinMPNN..."
  conda run -n SE3nv2 --no-capture-output python block1_design/step2_proteinmpnn.py \
    --pdb_dir "$BLOCK1_DIR/backbone_pdbs" \
    --output_dir "$BLOCK1_DIR/mpnn_seqs" \
    --num_seq "$N_MPNN_SEQ" \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/step2_proteinmpnn.log"
  echo "[$(date '+%H:%M:%S')] STEP 2 done."

  FASTA="$BLOCK1_DIR/mpnn_seqs/all_sequences.fasta"
fi

# Use provided FASTA or from design step
if [[ -z "$FASTA" && -z "$PEP_DIR" ]]; then
  echo "ERROR: No FASTA or peptide PDB directory. Use --fasta or --pep_dir"
  exit 1
fi

if [[ -z "$PEP_DIR" ]]; then
  # STEP 3: Build cyclic peptide 3D structures
  echo ""
  echo "[$(date '+%H:%M:%S')] STEP 3: Building cyclic PDBs..."
  conda run -n pyrose --no-capture-output python block1_design/step3_build_cycpep.py \
    --fasta "$FASTA" \
    --output_dir "$BLOCK1_DIR/cyclic_pdbs" \
    2>&1 | tee "$LOG_DIR/step3_build_cycpep.log"
  echo "[$(date '+%H:%M:%S')] STEP 3 done."
  PEP_DIR="$BLOCK1_DIR/cyclic_pdbs"
fi

# =============================================================================
# BLOCK 2: EVALUATION
# =============================================================================
echo ""
echo "=== BLOCK 2: EVALUATION ==="

# STEP 4: Rosetta docking
echo "[$(date '+%H:%M:%S')] STEP 4: Rosetta DockMCM docking..."
conda run -n pyrose --no-capture-output python block2_evaluation/step4_rosetta_dock.py \
  --pep_dir "$PEP_DIR" \
  --receptor "$RECEPTOR" \
  --output_dir "$BLOCK2_DIR/docking_results" \
  --n_decoys "$N_DECOYS" \
  --n_workers "$N_WORKERS" \
  2>&1 | tee "$LOG_DIR/step4_rosetta_dock.log"
echo "[$(date '+%H:%M:%S')] STEP 4 done."

# STEP 5: Interface metrics
echo "[$(date '+%H:%M:%S')] STEP 5: Interface metrics..."
conda run -n pyrose --no-capture-output python block2_evaluation/step5_interface_metrics.py \
  --docking_dir "$BLOCK2_DIR/docking_results" \
  --output_dir "$BLOCK2_DIR" \
  2>&1 | tee "$LOG_DIR/step5_metrics.log"
echo "[$(date '+%H:%M:%S')] STEP 5 done."

# STEP 6: Ranking and report
echo "[$(date '+%H:%M:%S')] STEP 6: Ranking..."
conda run -n pyrose --no-capture-output python block2_evaluation/step6_rank_and_report.py \
  --results_csv "$BLOCK2_DIR/evaluation_results.csv" \
  --output_dir "$BLOCK2_DIR" \
  2>&1 | tee "$LOG_DIR/step6_ranking.log"
echo "[$(date '+%H:%M:%S')] STEP 6 done."

# =============================================================================
# BLOCK 3: MUTATION SCANNING
# =============================================================================
echo ""
echo "=== BLOCK 3: MUTATION SCANNING ==="

# STEP 7: Alanine scan
echo "[$(date '+%H:%M:%S')] STEP 7: Alanine scan (Round 1)..."
conda run -n pyrose --no-capture-output python block3_mutation_scan/step7_alanine_scan.py \
  --docking_dir "$BLOCK2_DIR/docking_results" \
  --output_dir "$BLOCK3_DIR" \
  --n_workers "$N_WORKERS" \
  2>&1 | tee "$LOG_DIR/step7_alanine_scan.log"
echo "[$(date '+%H:%M:%S')] STEP 7 done."

# Extract hotspot positions from CSV
if [[ -f "$BLOCK3_DIR/hotspot_summary.csv" ]]; then
  HOTSPOT_POSITIONS=$(python3 -c "
import pandas as pd
df = pd.read_csv('$BLOCK3_DIR/hotspot_summary.csv')
hs = df[df['is_hotspot']==True]['position'].tolist()
print(' '.join(map(str, hs)) if hs else '2 3 9 11 12')
" 2>/dev/null || echo "2 3 9 11 12")
fi
echo "  Hotspot positions: $HOTSPOT_POSITIONS"

# STEP 8: Saturation scan
echo "[$(date '+%H:%M:%S')] STEP 8: Saturation mutagenesis (Round 2)..."
conda run -n pyrose --no-capture-output python block3_mutation_scan/step8_saturation_scan.py \
  --docking_dir "$BLOCK2_DIR/docking_results" \
  --hotspot_positions $HOTSPOT_POSITIONS \
  --output_dir "$BLOCK3_DIR" \
  --n_workers "$N_WORKERS" \
  2>&1 | tee "$LOG_DIR/step8_saturation.log"
echo "[$(date '+%H:%M:%S')] STEP 8 done."

# STEP 9: Report generation
echo "[$(date '+%H:%M:%S')] STEP 9: Generating reports..."
conda run -n pyrose --no-capture-output python block3_mutation_scan/step9_scan_report.py \
  --scan_dir "$BLOCK3_DIR" \
  2>&1 | tee "$LOG_DIR/step9_report.log"
echo "[$(date '+%H:%M:%S')] STEP 9 done."

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "============================================================"
echo " Pipeline Complete!"
echo "============================================================"
echo " Key outputs:"
echo "   Block 1: $BLOCK1_DIR/cyclic_pdbs/"
echo "   Block 2: $BLOCK2_DIR/evaluation_report.html"
echo "   Block 2: $BLOCK2_DIR/ranked_candidates.csv"
echo "   Block 3: $BLOCK3_DIR/mutation_scan_report.html"
echo "   Block 3: $BLOCK3_DIR/top_mutations.csv"
echo "   Logs:    $LOG_DIR/"
echo "============================================================"
