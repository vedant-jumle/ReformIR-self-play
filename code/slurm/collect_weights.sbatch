#!/bin/bash
#SBATCH --job-name=reformir_weights
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=/tudelft.net/staff-groups/ewi/st/reformir-self-play/logs/collect_weights_%j.log

set -euo pipefail

SHARED=/tudelft.net/staff-groups/ewi/st/reformir-self-play
SIF=$SHARED/reformir.sif
CODE=$SHARED/ReformIR-self-play/code
LOCAL=/tmp/vvjumle-reformir-weights
DL=${DL:-19}

export PYTERRIER_HOME=$LOCAL/.pyterrier
export IR_DATASETS_HOME=$LOCAL/.ir_datasets
mkdir -p "$PYTERRIER_HOME/corpora" "$IR_DATASETS_HOME"
ln -sfn "$SHARED/.pyterrier/corpora/msmarco_passage" "$PYTERRIER_HOME/corpora/msmarco_passage"
mkdir -p "$LOCAL/weights"

apptainer exec --nv --bind /tudelft.net --pwd "$LOCAL" \
  --env PYTERRIER_HOME="$PYTERRIER_HOME" \
  --env IR_DATASETS_HOME="$IR_DATASETS_HOME" \
  "$SIF" python "$CODE/collect_weights.py" \
    --dl "$DL" \
    --budget 100 \
    --batch 16 \
    --ce 7 \
    --reformulation_file "$SHARED/ReformIR-self-play/data/genqr_ensemble/dl${DL}.csv" \
    --weights_out "$LOCAL/weights/dl${DL}_weights.csv"

mkdir -p "$SHARED/weights"
cp -r "$LOCAL/weights/." "$SHARED/weights/"
echo "Done. Weights at $SHARED/weights/dl${DL}_weights.csv"
