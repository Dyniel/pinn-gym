#!/bin/bash
#SBATCH --job-name=pinn_gym
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=/users/scratch1/%u/logs/%x_%j.out
#
# pinn_gym pipeline launcher.
#
# Usage:
#   sbatch slurm/pinn_gym.slurm configs/sr_full.yaml
#   sbatch --partition=cpu slurm/pinn_gym.slurm configs/sr_full.yaml --stage build
#
# Anything past the YAML path is forwarded verbatim to `pinn-gym run`, so you
# can pass --stage build,train to subset the pipeline, --dry-run to skip the
# scientific work, etc.

set -euo pipefail

CONFIG_PATH="${1:-}"
if [[ -z "$CONFIG_PATH" ]]; then
    echo "usage: sbatch slurm/pinn_gym.slurm <config.yaml> [pinn-gym run flags...]" >&2
    exit 2
fi
shift

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "config not found: $CONFIG_PATH" >&2
    exit 2
fi

unalias python 2>/dev/null || true
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
unset PYTHONPATH

# Override these to point at your cluster's curated conda env.
PINN_GYM_ROOT="${PINN_GYM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PINN_GYM_CONDA_ENV="${PINN_GYM_CONDA_ENV:-/users/scratch1/$USER/conda_envs/py312}"

mkdir -p "/users/scratch1/$USER/logs"
cd "$PINN_GYM_ROOT"

if ! type module >/dev/null 2>&1; then
    [[ -f /etc/profile ]] && source /etc/profile
fi
module load trytonp/conda/py313_25.9.1-3 || true
eval "$(conda shell.bash hook)"
conda activate "$PINN_GYM_CONDA_ENV"
export LD_LIBRARY_PATH="$PINN_GYM_CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"

PYBIN="$PINN_GYM_CONDA_ENV/bin/python"

echo "pinn_gym launcher"
echo "  root   : $PINN_GYM_ROOT"
echo "  config : $CONFIG_PATH"
echo "  env    : $PINN_GYM_CONDA_ENV"
echo "  extras : $*"
echo

# Validate first — fail fast on schema errors before queueing GPUs.
"$PYBIN" -m pinn_gym.cli validate "$CONFIG_PATH"

# Run the configured pipeline. Forward any extra flags to `pinn-gym run`.
"$PYBIN" -m pinn_gym.cli run "$CONFIG_PATH" "$@"
