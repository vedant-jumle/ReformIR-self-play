#!/bin/bash
#SBATCH --job-name=reformir_dpo
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=/tudelft.net/staff-groups/ewi/st/reformir-self-play/logs/train_dpo_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=v.v.jumle@student.tudelft.nl

set -euo pipefail

SHARED=/tudelft.net/staff-groups/ewi/st/reformir-self-play
SIF=$SHARED/reformir.sif
CODE=$SHARED/ReformIR-self-play/code
LOCAL=/tmp/vvjumle-reformir-dpo

mkdir -p "$LOCAL/output" "$LOCAL/hf_cache"

apptainer exec --nv --bind /tudelft.net --pwd "$LOCAL" \
  --env HF_HOME="$LOCAL/hf_cache" \
  --env TRANSFORMERS_CACHE="$LOCAL/hf_cache" \
  "$SIF" python "$CODE/train_dpo.py" \
    --dpo_data "$SHARED/ReformIR-self-play/data/dpo/msmarco_train_1k_dpo.jsonl" \
    --model_name "google/flan-t5-large" \
    --output_dir "$LOCAL/output" \
    --epochs 3 \
    --batch_size 4 \
    --lr 5e-5 \
    --beta 0.1

mkdir -p "$SHARED/models/flan-t5-large-dpo"
cp -r "$LOCAL/output/." "$SHARED/models/flan-t5-large-dpo/"
echo "Done. Model at $SHARED/models/flan-t5-large-dpo/"
