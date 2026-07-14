#!/bin/bash
#SBATCH --job-name=reformir_build_index
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/tudelft.net/staff-groups/ewi/st/reformir-self-play/logs/build_index_%j.log

set -euo pipefail

SHARED=/tudelft.net/staff-groups/ewi/st/reformir-self-play
SIF=$SHARED/reformir.sif

# Build on local node disk (SMB mounts under /tudelft.net reject chmod,
# which pyterrier's jar/dataset downloaders require). Copy results back after.
LOCAL=/tmp/vvjumle-reformir-build
export PYTERRIER_HOME=$LOCAL/.pyterrier
export IR_DATASETS_HOME=$LOCAL/.ir_datasets
mkdir -p "$PYTERRIER_HOME" "$IR_DATASETS_HOME"

apptainer exec --nv \
  --bind /tudelft.net \
  --env PYTERRIER_HOME="$PYTERRIER_HOME" \
  --env IR_DATASETS_HOME="$IR_DATASETS_HOME" \
  "$SIF" python "$SHARED/ReformIR-self-play/code/build_indexes.py"

echo "Build finished, copying results back to shared storage..."
mkdir -p "$SHARED/.pyterrier" "$SHARED/.ir_datasets"
cp -r "$PYTERRIER_HOME/." "$SHARED/.pyterrier/"
cp -r "$IR_DATASETS_HOME/." "$SHARED/.ir_datasets/"
echo "Done. Index available under $SHARED/.pyterrier"
