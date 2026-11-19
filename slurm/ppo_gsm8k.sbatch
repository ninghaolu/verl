#!/bin/bash
#SBATCH --job-name=verl-ppo-gsm8k
#SBATCH --account=torch_pr_147_syang
#SBATCH --partition=h200_syang_2
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/projects/work/yang-lab/users/nl2752/verl/slurm/logs/%x_%j.out
# PPO on GSM8K, Qwen2.5-0.5B-Instruct. Extra arguments are hydra overrides.
#   sbatch slurm/ppo_gsm8k.sbatch
#   NGPUS=4 sbatch --gres=gpu:h200:4 --cpus-per-task=32 slurm/ppo_gsm8k.sbatch
set -euo pipefail

ROOT=/projects/work/yang-lab/users/nl2752/verl
ENV=/scratch/nl2752/miniconda3/envs/verl

# the env's python first, and nothing that shadows /usr/bin/env
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/projects/work/yang-lab/users/nl2752/tools$' | paste -sd:)
export PATH=$ENV/bin:$PATH
export HF_HOME=/scratch/nl2752/_cache/hf
export HF_HUB_OFFLINE=1
export WANDB_DIR=/projects/work/yang-lab/users/nl2752/verl_work
export WANDB_ENTITY=${WANDB_ENTITY:-nl2752-new-york-university}
# the key of the account that may write to that team (the default key in ~/.netrc may not)
WANDB_KEY_FILE=${WANDB_KEY_FILE:-$HOME/.config/pear/wandb_nyu.key}
if [ -r "$WANDB_KEY_FILE" ]; then
  WANDB_API_KEY=$(tr -d "[:space:]" < "$WANDB_KEY_FILE")
  export WANDB_API_KEY
fi
export NGPUS=${NGPUS:-$(nvidia-smi -L | wc -l)}
export MODEL=${MODEL:-$HF_HOME/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775}

mkdir -p "$WANDB_DIR"
cd "$ROOT"
nvidia-smi -L
bash examples/ppo_trainer/run_qwen2_5_0_5b_ppo_gsm8k.sh "$@"
