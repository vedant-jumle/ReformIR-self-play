#!/bin/bash
#SBATCH --job-name=reformir_build_container
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/tudelft.net/staff-groups/ewi/st/reformir-self-play/logs/build_container_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=v.v.jumle@student.tudelft.nl

set -euo pipefail

SHARED=/tudelft.net/staff-groups/ewi/st/reformir-self-play
DEF=$SHARED/ReformIR-self-play/reformir.def
SIF=$SHARED/reformir.sif

mkdir -p "$SHARED/logs"

# Build in /tmp (local disk, no SMB chmod issues)
# APPTAINER_TMPDIR and APPTAINER_CACHEDIR must be on local storage
export APPTAINER_TMPDIR=/tmp/vvjumle-apptainer-tmp
export APPTAINER_CACHEDIR=/tmp/vvjumle-apptainer-cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

echo "Building $SIF from $DEF ..."
apptainer build --ignore-fakeroot-command "$SIF" "$DEF"
echo "Done: $SIF"
ls -lh "$SIF"
