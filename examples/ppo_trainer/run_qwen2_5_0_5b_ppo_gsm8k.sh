#!/usr/bin/env bash
# PPO on GSM8K with Qwen2.5-0.5B-Instruct (actor and critic), the quickstart recipe on NGPUS GPUs of one machine.
# Run from the repo root: NGPUS=2 bash examples/ppo_trainer/run_qwen2_5_0_5b_ppo_gsm8k.sh [hydra overrides]
set -euo pipefail

NGPUS=${NGPUS:-2}
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
DATA_DIR=${DATA_DIR:-$HOME/data/gsm8k}
PROJECT=${PROJECT:-rl-verl}
RUN_NAME=${RUN_NAME:-qwen2.5-0.5b-ppo-gsm8k-${NGPUS}gpu}
OUT_DIR=${OUT_DIR:-/projects/work/yang-lab/users/nl2752/verl_work/$RUN_NAME}
NUM_CPUS=${NUM_CPUS:-${SLURM_CPUS_PER_TASK:-16}}

# FORMAT_SCORE=0.1 gives that much reward to a response in the right format ("#### <number>") with a wrong number,
# through verl/utils/reward_score/gsm8k_reward.py. Unset: the built-in 0/1 reward.
REWARD_ARGS=()
if [ -n "${FORMAT_SCORE:-}" ]; then
  REWARD_ARGS=(
    custom_reward_function.path=verl/utils/reward_score/gsm8k_reward.py
    custom_reward_function.name=compute_score
    "+custom_reward_function.reward_kwargs.format_score=$FORMAT_SCORE"
  )
fi

mkdir -p "$OUT_DIR"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/test.parquet" \
  data.train_batch_size=256 \
  data.max_prompt_length=512 \
  data.max_response_length=256 \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
  "actor_rollout_ref.actor.checkpoint.contents=[model,optimizer,extra,hf_model]" \
  critic.optim.lr=1e-5 \
  critic.model.path="$MODEL" \
  critic.ppo_micro_batch_size_per_gpu=32 \
  algorithm.use_kl_in_reward=True \
  algorithm.kl_ctrl.kl_coef=0.001 \
  "trainer.logger=[console,wandb]" \
  trainer.project_name="$PROJECT" \
  trainer.experiment_name="$RUN_NAME" \
  trainer.val_before_train=True \
  trainer.log_val_generations=10 \
  trainer.validation_data_dir="$OUT_DIR/val_generations" \
  trainer.default_local_dir="$OUT_DIR/checkpoints" \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node="$NGPUS" \
  trainer.nnodes=1 \
  trainer.save_freq=200 \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 \
  trainer.test_freq=10 \
  trainer.total_epochs=15 \
  ray_init.num_cpus="$NUM_CPUS" \
  "${REWARD_ARGS[@]}" \
  "$@" 2>&1 | tee "$OUT_DIR/train.log"
