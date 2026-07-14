#!/bin/bash
#SBATCH --job-name=reformir_dl19
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=/tudelft.net/staff-groups/ewi/st/reformir-self-play/logs/run_dl19_%j.log

set -euo pipefail

SHARED=/tudelft.net/staff-groups/ewi/st/reformir-self-play
SIF=$SHARED/reformir.sif
CODE=$SHARED/ReformIR-self-play/code

# Use local /tmp for PYTERRIER_HOME — SMB mounts reject os.chmod on temp files
# (pyterrier artifact downloader and terrier jar fetcher both call os.chmod).
# Symlink the already-built index from shared storage so retrieval still works.
LOCAL=/tmp/vvjumle-reformir-run
export PYTERRIER_HOME=$LOCAL/.pyterrier
export IR_DATASETS_HOME=$LOCAL/.ir_datasets
mkdir -p "$PYTERRIER_HOME/corpora" "$IR_DATASETS_HOME"

# Symlink pre-built index into local pyterrier home
ln -sfn "$SHARED/.pyterrier/corpora/msmarco_passage" "$PYTERRIER_HOME/corpora/msmarco_passage"

mkdir -p "$LOCAL/runs/reform/dl19" "$LOCAL/runs/reform/dl20"

apptainer exec --nv \
  --bind /tudelft.net \
  --pwd "$LOCAL" \
  --env PYTERRIER_HOME="$PYTERRIER_HOME" \
  --env IR_DATASETS_HOME="$IR_DATASETS_HOME" \
  "$SIF" python "$CODE/run_dl19.py" \
    --dl 19 \
    --budget 100 \
    --batch 16 \
    --ce 7 \
    --reformulation_file "$SHARED/ReformIR-self-play/data/genqr_ensemble/dl19.csv"

# Copy run results back to shared storage
echo "Copying results to shared storage..."
cp -r "$LOCAL/runs" "$SHARED/"
echo "Done. Results at $SHARED/runs"
