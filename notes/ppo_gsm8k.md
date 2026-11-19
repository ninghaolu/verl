# PPO on GSM8K with verl (Qwen2.5-0.5B-Instruct)

## Overview

We fine-tune `Qwen2.5-0.5B-Instruct` with PPO so that it answers GSM8K word problems in the format `#### <number>`. The repo is verl v0.4.0 (branch `lab/ppo-gsm8k`).

Four components take part:

- **Actor** $\pi_\theta$: the LLM being trained (FSDP, one copy per GPU).
- **Rollout engine**: vLLM, sharing the actor's weights on the same GPUs (`hybrid_engine`). It samples the responses.
- **Reference policy** $\pi_{\text{ref}}$: a frozen copy of the initial model, used only for the KL penalty.
- **Critic** $V_\phi$: a second copy of Qwen2.5-0.5B with a 1-output token-classification head. It predicts the future reward at every response token.

The reward is a rule, not a model: 1 if the number after `####` equals the ground truth, else 0.

How to run it:

```bash
sbatch slurm/ppo_gsm8k.sbatch                                              # 2x H200
NGPUS=4 sbatch --gres=gpu:h200:4 --cpus-per-task=32 slurm/ppo_gsm8k.sbatch # 4x H200
```

The job calls `examples/ppo_trainer/run_qwen2_5_0_5b_ppo_gsm8k.sh`.

## 1. Dataset Selection

- **Dataset**: `openai/gsm8k`, config `main`. 7,473 train and 1,319 test problems.
- **Files**: `~/data/gsm8k/train.parquet` (rollouts) and `~/data/gsm8k/test.parquet` (validation).
- **Fields the pipeline consumes**: `prompt` (model input), `reward_model.ground_truth` (reward reference), `data_source` (selects the reward function).
- **Why it fits**: every problem has one numeric answer, so a rule gives an exact 0/1 reward with no reward model.

## 2. Data Processing

`examples/data_preprocess/gsm8k.py` converts each raw `{question, answer}` row:

```python
instruction_following = 'Let\'s think step by step and output the final answer after "####".'
question = question_raw + " " + instruction_following
solution = extract_solution(answer_raw)   # text after "#### ", commas removed -> "72"
```

At training time `verl/utils/dataset/rl_dataset.py` turns `prompt` into tokens:

```python
raw_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
input_ids  = tokenizer(raw_prompt)   # then LEFT-padded to max_prompt_length = 512
```

Prompts are left-padded and responses are right-padded, so one sequence is `[pad ... prompt | response ... pad]`, and the response always occupies the last `max_response_length = 256` positions.

### 2.1 Data Example

Row 0 of `train.parquet`:

```json
{
  "data_source": "openai/gsm8k",
  "prompt": [
    {
      "role": "user",
      "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let's think step by step and output the final answer after \"####\"."
    }
  ],
  "ability": "math",
  "reward_model": {
    "style": "rule",
    "ground_truth": "72"
  },
  "extra_info": {
    "split": "train",
    "index": 0,
    "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
    "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
  }
}
```

| Field | Use |
| --- | --- |
| `prompt` | Chat messages; becomes the model input after the chat template. |
| `reward_model.ground_truth` | Compared with the model's extracted answer to give the reward. |
| `data_source` | `"openai/gsm8k"` routes to `verl/utils/reward_score/gsm8k.py`; also the key validation scores are grouped by. |
| `extra_info.answer` | The human solution. **Not used by RL**: the model never sees it. |
| `ability`, `extra_info.split/index/question` | Metadata only. |

## 3. Loss Design

### 3.0 Notation

| Symbol | Meaning | Tensor in code |
| --- | --- | --- |
| $x$ | prompt tokens | `prompts` |
| $y = (y_1..y_T)$ | sampled response tokens; $T \le 256$ | `responses` |
| $s_t = (x, y_{<t})$ | state before token $t$ | prefix of `input_ids` |
| $m_t$ | 1 for real response tokens, 0 for padding | `response_mask` |
| $\pi_{\text{old}}$ | actor at sampling time (weights before this step's update) | `old_log_probs` |
| $\pi_\theta$ | actor being updated | `log_prob` |
| $\pi_{\text{ref}}$ | frozen initial model | `ref_log_prob` |
| $S$ | sequence score, 0 or 1 | `token_level_scores` (at last token) |
| $r_t$ | per-token reward after the KL penalty | `token_level_rewards` |
| $V_\phi(s_t)$ | critic value | `values`, `vpreds` |
| $R_t$ | return (reward-to-go) | `returns` |
| $A_t$ | advantage | `advantages` |
| $\beta$ | KL coefficient, 0.001 | `kl_ctrl.value` |
| $\epsilon$ | PPO clip range, 0.2 | `clip_ratio` |

### 3.1 Formulas

**Score.** The whole response earns one number, placed on its last token:

$$S = \mathbb{1}[\text{answer}(y) = \text{ground truth}], \qquad \text{score}_t = S \cdot \mathbb{1}[t = T]$$

**KL penalty in the reward.** Each token pays for moving away from the reference model. This is a single-sample estimate of $\mathrm{KL}(\pi_{\text{old}} \| \pi_{\text{ref}})$ at that position:

$$k_t = \log \pi_{\text{old}}(y_t \mid s_t) - \log \pi_{\text{ref}}(y_t \mid s_t), \qquad r_t = \text{score}_t - \beta\, k_t$$

**Advantage (GAE).**

$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t), \qquad A_t = \sum_{l \ge 0} (\gamma\lambda)^l\, \delta_{t+l}, \qquad R_t = A_t + V_\phi(s_t)$$

We use $\gamma = \lambda = 1$, so this collapses to the Monte Carlo form: $R_t = \sum_{t' \ge t} r_{t'}$ and $A_t = R_t - V_\phi(s_t)$. The advantages are then whitened over all valid tokens in the batch: $\hat{A}_t = (A_t - \mu_A) / \sigma_A$.

**Actor loss (clipped PPO).**

$$\rho_t = \frac{\pi_\theta(y_t \mid s_t)}{\pi_{\text{old}}(y_t \mid s_t)}, \qquad L^{\text{actor}} = -\frac{1}{\sum m_t} \sum_t m_t \, \min\!\big(\rho_t \hat{A}_t,\ \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\, \hat{A}_t\big)$$

**Critic loss (clipped squared error).** The new prediction may move at most $c = 0.5$ away from the value computed before the update:

$$L^{\text{critic}} = \frac{1}{\sum m_t} \sum_t m_t \, \max\!\Big( (V_\phi(s_t) - R_t)^2,\ \big(\mathrm{clip}(V_\phi(s_t),\, V_{\text{old}}(s_t) \pm c) - R_t\big)^2 \Big)$$

There is no entropy bonus (`entropy_coeff = 0`) and no KL term in the loss (`use_kl_loss = False`). KL acts only through the reward.

### 3.2 One training step

```text
for each step (256 prompts):
    # 1. sample
    responses  = vllm.generate(prompts, temperature=1.0, n=1, max_tokens=256)
    # 2. reward
    score[i]   = gsm8k.compute_score(decode(responses[i]), ground_truth[i])     # 0 or 1, on the last token
    # 3. log-probs of the sampled tokens (no grad)
    old_log_probs = actor(prompt + response)        # also gives the entropy metric
    ref_log_prob  = ref_policy(prompt + response)
    values        = critic(prompt + response)
    # 4. reward and advantage (on the driver, plain tensors)
    rewards    = score - beta * (old_log_probs - ref_log_prob)
    adv, ret   = GAE(rewards, values, gamma=1, lam=1);  adv = whiten(adv)
    # 5. updates: 1 PPO epoch, mini-batches of 64
    for mini_batch in split(batch, 64):
        critic:  vf_loss = clipped_mse(critic(mini_batch), ret, values);  step (lr 1e-5)
    for mini_batch in split(batch, 64):
        actor:   pg_loss = clipped_ppo(actor(mini_batch), old_log_probs, adv);  step (lr 1e-6)
    # 6. vLLM picks up the new actor weights for the next step
```

One epoch is 29 steps (7,473 / 256). 15 epochs is 435 steps, and each step makes 4 optimizer updates per model.

## 4. Implementation Details: Math to Code

The loop itself is `RayPPOTrainer.fit` in `verl/trainer/ppo/ray_trainer.py`.

### 4.1 Score → `token_level_scores`

`verl/workers/reward_manager/naive.py` decodes each response, calls the reward function, and writes the score at the last valid token:

```python
score = self.compute_score(data_source=data_source, solution_str=response_str, ground_truth=ground_truth, extra_info=extra_info)
reward_tensor[i, valid_response_length - 1] = score
```

`verl/utils/reward_score/gsm8k.py` (strict mode) requires the `####` marker:

```python
solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)   # None -> score 0
final_answer = solution.group(0).split("#### ")[1].replace(",", "").replace("$", "")
return 1.0 if final_answer == ground_truth else 0.0
```

So the reward tests format and correctness together. A correct answer written as "**3**" scores 0.

### 4.2 Log-probabilities and the logit shift

`logits[:, t, :]` predicts `input_ids[:, t + 1]`. The response is the last `R = 256` positions, so the logits that predict it are one position to the left (`verl/workers/actor/dp_actor.py`):

```python
logits    = logits[:, -response_length - 1 : -1, :]              # (bsz, R, vocab)
log_probs = logprobs_from_logits(logits, micro_batch["responses"])  # (bsz, R)
```

The critic uses the same slice, so `values[:, t]` is $V_\phi(s_t)$, the value **before** token $t$ is emitted (`verl/workers/critic/dp_critic.py`):

```python
values = output.logits[:, -response_length - 1 : -1].squeeze(-1)
```

The critic is `AutoModelForTokenClassification` with `num_labels = 1` (`verl/workers/fsdp_workers.py`): the Qwen backbone plus a freshly initialised linear head.

### 4.3 KL penalty → `token_level_rewards`

$r_t = \text{score}_t - \beta k_t$, in `apply_kl_penalty` (`ray_trainer.py`):

```python
kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty="kl")  # logprob - ref_logprob
kld = kld * response_mask
token_level_rewards = token_level_scores - beta * kld
current_kl = masked_mean(kld, mask=response_mask, axis=-1).mean()    # -> actor/reward_kl_penalty
```

### 4.4 GAE → `advantages`, `returns`

`compute_gae_advantage_return` in `verl/trainer/ppo/core_algos.py`:

```python
for t in reversed(range(gen_len)):
    nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
    delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
    lastgaelam = delta + gamma * lam * lastgaelam
    advantages_reversed.append(lastgaelam)
advantages = torch.stack(advantages_reversed[::-1], dim=1)
returns = advantages + values                                  # before whitening
advantages = verl_F.masked_whiten(advantages, response_mask)   # (A - mean) / std over valid tokens
```

### 4.5 Actor loss → `actor/pg_loss`

`compute_policy_loss` in `core_algos.py`:

```python
negative_approx_kl = log_prob - old_log_prob
ratio  = torch.exp(negative_approx_kl)
ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)           # actor/ppo_kl
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="token-mean")
```

verl also applies a dual clip: for $\hat{A}_t < 0$ the loss is capped at $-\hat{A}_t \cdot c$ with `clip_ratio_c = 3`, which only matters when the ratio exceeds 3.

### 4.6 Critic loss → `critic/vf_loss`

`compute_value_loss` in `core_algos.py`:

```python
vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
vf_losses1 = (vpreds - returns) ** 2
vf_losses2 = (vpredclipped - returns) ** 2
vf_loss = agg_loss(torch.max(vf_losses1, vf_losses2), response_mask, "token-mean")
```

## 5. Key Hyperparameters

Only values we set, or defaults that matter for the method. All are in `examples/ppo_trainer/run_qwen2_5_0_5b_ppo_gsm8k.sh`.

| Group | Parameter | Value | Note |
| --- | --- | --- | --- |
| Learning rate | `actor_rollout_ref.actor.optim.lr` | 1e-6 | Constant, no warmup. |
| | `critic.optim.lr` | 1e-5 | 10x the actor: the value head starts from random. |
| KL | `algorithm.use_kl_in_reward` | True | Default is False; we turn it on, which loads the reference policy. |
| | `algorithm.kl_ctrl.kl_coef` ($\beta$) | 0.001 | Fixed controller. |
| | `algorithm.kl_penalty` | `kl` (default) | $\log\pi_{\text{old}} - \log\pi_{\text{ref}}$ per token. |
| PPO | `actor.ppo_epochs` | 1 (default) | Each rollout batch is used once. |
| | `actor.clip_ratio` ($\epsilon$) | 0.2 (default) | |
| | `critic.cliprange_value` ($c$) | 0.5 (default) | |
| | `algorithm.gamma`, `algorithm.lam` | 1.0, 1.0 (default) | Monte Carlo returns. |
| | `actor.grad_clip` | 1.0 (default) | |
| Batch | `data.train_batch_size` | 256 | Prompts per step; `rollout.n = 1`. |
| | `actor.ppo_mini_batch_size` | 64 | 4 optimizer updates per step. |
| Lengths | `data.max_prompt_length` / `max_response_length` | 512 / 256 | |
| Sampling | `rollout.temperature` | 1.0 (default) | Validation is greedy. |
| Schedule | `trainer.total_epochs` | 15 | 435 steps. |
| | `trainer.test_freq` | 10 | Full test set every 10 steps. |

`ppo_micro_batch_size_per_gpu = 32` only sets how many sequences go through one forward pass. It affects memory and speed, not the gradient.

## 6. Metrics Logged to W&B

Run: https://forge.coreweave.com/wandb/nl2752-new-york-university/rl-verl/runs/d7wtuezp (project `rl-verl`, run `qwen2.5-0.5b-ppo-gsm8k-2gpu`)

Metrics from the actor and critic updates are averaged over the mini-batches of the step. The `critic/*` statistics come from `compute_data_metrics` in `verl/trainer/ppo/metric_utils.py`. Example values are from step 1 of this run.

| Metric | Computation | Step 1 |
| --- | --- | --- |
| `critic/score/mean` | `token_level_scores.sum(-1).mean()`: fraction of sampled responses that are correct. **The training reward curve.** | 0.004 |
| `critic/rewards/mean` | `token_level_rewards.sum(-1).mean()`: score minus the summed KL penalty. | 0.004 |
| `val-core/openai/gsm8k/reward/mean@1` | Mean score on the 1,319 test problems with greedy decoding. **The headline metric.** | 0.000 (step 0) |
| `actor/pg_loss` | $L^{\text{actor}}$ of 3.1. | 0.005 |
| `actor/pg_clipfrac` | Fraction of tokens where the clipped term is the active one. | 0.001 |
| `actor/ppo_kl` | $\text{mean}(\log\pi_{\text{old}} - \log\pi_\theta)$: how far this step's update has moved the policy. | 0.000 |
| `actor/reward_kl_penalty` | $\text{mean}(k_t)$: drift from the reference model. | 0.000 |
| `actor/entropy_loss` | Mean token entropy of the actor on its own samples. Logged only; not in the loss. | 0.425 |
| `actor/grad_norm` | Gradient norm before clipping to 1.0. | 2.06 |
| `critic/vf_loss` | $L^{\text{critic}}$ of 3.1. | 28.8 |
| `critic/vf_explained_var` | $1 - \mathrm{Var}(R - V) / \mathrm{Var}(R)$. 1 is a perfect critic; negative is worse than predicting a constant. | -4684 |
| `critic/values/mean`, `critic/returns/mean` | Means over valid tokens. | 3.13, 0.004 |
| `critic/advantages/max`, `min` | After whitening; the mean is 0 by construction. | 4.25, -3.89 |
| `response_length/mean`, `clip_ratio` | Mean response tokens; fraction cut at 256. | 234.7, 0.61 |

### 6.1 Reward: a worked example

Step 1 samples 256 responses and exactly one is correct:

```text
critic/score/mean = 1 / 256 = 0.0039   -> logged as 0.004
critic/score/max  = 1.0
```

For that response (say 200 tokens long), `token_level_scores` is `[0, 0, ..., 0, 1]`. The KL penalty then changes every token slightly. At step 1 the actor still equals the reference, and $k_t$ is nonzero only through numerical noise, so `critic/rewards/max = 1.001` and `critic/rewards/min = -0.002`.

### 6.2 Actor loss: a worked example

Take three tokens with whitened advantages and ratios:

```text
A_hat = [ 2.0, -1.0,  0.5]
ratio = [ 1.3,  0.7,  1.0]      clip to [0.8, 1.2] -> [1.2, 0.8, 1.0]

unclipped  -A*ratio = [-2.6,  0.7, -0.5]
clipped    -A*clip  = [-2.4,  0.8, -0.5]
loss = max of both  = [-2.4,  0.8, -0.5]      pg_loss = mean = -0.70
pg_clipfrac = 2/3   (tokens 1 and 2 use the clipped term)
```

Token 1: the policy already raised a good token's probability by 30%, so the gain is capped at 20%. Token 2: it lowered a bad token's probability by 30%, and the credit is capped the same way. Beyond the clip range the gradient is zero.

Two consequences explain the curves:

- At the first mini-batch $\rho_t = 1$, so `pg_loss` $= -\text{mean}(\hat{A}_t)$ over that mini-batch. The batch mean is 0 after whitening, so `pg_loss` hovers around 0 for the whole run (between -0.08 and +0.08 here). **Its value says nothing about progress**; watch the score.
- With `ppo_epochs = 1` and lr 1e-6 the ratio barely leaves 1, so `pg_clipfrac` and `ppo_kl` stay near 0.

### 6.3 Critic loss: a worked example

For a wrong response every return is about 0. The untrained value head predicts anything from -16 to 21 (mean 3.1). One token with $V = 3.1$, $R = 0$ contributes $(3.1 - 0)^2 = 9.6$, and the spread around the mean adds the rest, which is why `critic/vf_loss` starts at 28.8 and `vf_explained_var` at -4684. Both recover fast: by step 10 `vf_loss` is 0.33, and by step 435 it is 0.105 with `vf_explained_var` 0.47.

## 7. Results

Run `qwen2.5-0.5b-ppo-gsm8k-2gpu`: 435 steps on 2x H200 in 60 minutes (Slurm job 23570694). Outputs are under `/projects/work/yang-lab/users/nl2752/verl_work/qwen2.5-0.5b-ppo-gsm8k-2gpu/`: `val_generations/<step>.jsonl` (every test response at each validation), `checkpoints/global_step_435/` (actor in HF format, critic), `train.log`.

### 7.1 Training curves

| Step | Train score (sampled) | Test score (greedy) | Response tokens | Cut at 256 | KL to reference | Entropy | `vf_loss` | Explained var |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.004 | 0.000 (step 0) | 235 | 61% | 0.000 | 0.425 | 28.8 | -4684 |
| 10 | 0.012 | 0.005 | 231 | 62% | 0.002 | 0.392 | 0.33 | -3.8 |
| 50 | 0.570 | 0.494 | 163 | 9% | 0.145 | 0.222 | 0.30 | 0.12 |
| 100 | 0.633 | 0.528 | 153 | 5% | 0.220 | 0.127 | 0.21 | 0.27 |
| 200 | 0.695 | 0.534 | 147 | 3% | 0.314 | 0.066 | 0.18 | 0.34 |
| 300 | 0.707 | 0.543 | 149 | 3% | 0.321 | 0.068 | 0.20 | 0.35 |
| 435 | 0.742 | 0.546 | 152 | 4% | 0.338 | 0.059 | 0.11 | 0.47 |

- Almost all of the gain comes between steps 10 and 60 (under 2 epochs). After that the test score moves between 0.51 and 0.55; the best is 0.552 at step 270.
- The train score keeps rising (0.63 → 0.74) while the test score is flat. The model sees each training problem 15 times, so part of this is memorisation.
- Entropy falls from 0.43 to 0.06: the policy becomes close to deterministic, which is also why later epochs add little.

### 7.2 Evaluation: before vs after RL

The trainer's own validation is the evaluation: all 1,319 test problems, greedy decoding, max 256 tokens. Step 0 is the model before RL and step 435 the model after. `notes/eval_compare.py` recomputes the table from the dumps:

```bash
python notes/eval_compare.py /projects/work/yang-lab/users/nl2752/verl_work/qwen2.5-0.5b-ppo-gsm8k-2gpu/val_generations
```

| Test set (n = 1,319) | Before RL (step 0) | After RL (step 435) |
| --- | --- | --- |
| Strict accuracy (the training reward) | 0.0% | 54.6% |
| Flexible accuracy (last number in the response is correct) | 28.3% | 54.6% |
| Response contains `####` | 0.9% | 95.5% |
| Response writes `####` twice | 0.2% | 79.2% |
| Mean response length (characters) | 797 | 477 |

- "Before RL" is the pretrained `Qwen2.5-0.5B-Instruct`. **No model was fine-tuned with SFT on GSM8K in this experiment**, so this is a zero-shot baseline, not an SFT-on-GSM8K one. `examples/sft/gsm8k` would give that.
- The strict number overstates the gain, since 0.0% is a format failure. The flexible number is the fair one: **28.3% → 54.6%**. Part of that is still format (shorter answers no longer run out of tokens) and part is better arithmetic.
- Per problem: 404 went from wrong to right, 57 from right (flexibly) to wrong, 316 stayed right, 542 stayed wrong.
- For reference, verl's published baseline for this model is 36.4 pretrained and 56.7 after PPO.

### 7.3 Sample responses before and after

**Wrong → right (ground truth 64).** *Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?*

```text
BEFORE (score 0)
... 2. **Calculate the cost of one glass:**
   - The cost of one glass is the sum of the cost of the first glass and the cost of one second glass.
   - Therefore, the cost of one glass is \( 5 + 3 = 8 \) dollars.
3. **Calculate the total cost for 16 glasses:** ... \( 8 \times 16 = 128 \) dollars.
So, Kylar needs to pay **128 dollars** for 16 glasses.

AFTER (score 1)
First, let's calculate the cost of each glass. Each glass costs $5, and every second glass costs 60% of the price. So the cost of each second glass is \(5 \times 0.60 = 3\) dollars.
There are 16 glasses, so the total cost for the first 8 glasses is \(8 \times 5 = 40\) dollars. The cost for the 8 second glasses is \(8 \times 3 = 24\) dollars.
So the total cost is \(40 + 24 = 64\) dollars.

#### 64
#### 64
```

**Truncated → right (ground truth 70000).** *Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?*

```text
BEFORE (score 0; wrong reasoning, and cut off at 256 tokens)
... 2. **Determine the increase in value:** ... 130,000 \times 1.50 = 195,000
3. **Calculate the profit:** ... 195,000 - 130,000 = 65,000
Thus, Josh's profit is \(\boxed{65,000}\

AFTER (score 1)
First, let's calculate the total cost of the house. ... the total cost is \(80000 + 50000 = 130000\) dollars.
The increased value of the house by 150% means the new value is \(80000 + 1.5 \times 80000 = 80000 + 120000 = 200000\) dollars.
The profit is the difference between the new value and the total cost, which is \(200000 - 130000 = 70000\) dollars.
So the profit is 70000 dollars.

#### 70000
#### 70000
```

**Still wrong, format learned (ground truth 540).** *James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?*

```text
BEFORE (score 0)
... Total distance = 60 meters/sprint × 3 sprints/week = 180 meters/week
So, the total distance James runs in a week is **180 meters**.

AFTER (score 0)
First, let's calculate the distance James runs in one sprint. He runs 60 meters.
He runs 3 sprints per week, so the distance for one week is \(3 \times 60 = 180\) meters.
Therefore, the total distance he runs a week is \(180\) meters.

#### 180
#### 180
```

**Reward loophole (ground truth 3).** *A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?*

```text
BEFORE (score 0; correct answer, wrong format)
... Total bolts = 2 + 1 = 3
So, the total number of bolts needed for the robe is **3**.

AFTER (score 1; the answer is not actually 3)
Let the amount of blue fiber be B. ... The total amount of fiber needed is B + \( \frac{B}{2} = \frac{3B}{2} \). So the final answer is \( \frac{3B}{2} \).#### 3B/2
```

The regex `#### (\-?[0-9\.\,]+)` matches the `3` in `#### 3B/2`, so this response is scored correct. Before RL the model solved this problem; after RL it gets credit for a worse answer.

What changed overall: the Markdown-heavy "numbered steps with bold headers" style is gone, responses are about 40% shorter, and nearly every response ends with `####`. The doubled `#### N` line (79% of responses) is a quirk the reward does not penalise, so nothing removes it.

## 8. Variants

Two more runs, each changing one thing from the baseline of section 7. Same data, model, 435 steps, 2x H200, one seed each. All three are in W&B project `rl-verl`.

| Run | Change | Command |
| --- | --- | --- |
| `qwen2.5-0.5b-ppo-gsm8k-2gpu` ([W&B](https://forge.coreweave.com/wandb/nl2752-new-york-university/rl-verl/runs/d7wtuezp)) | Baseline: actor lr 1e-6, 0/1 reward | `sbatch slurm/ppo_gsm8k.sbatch` |
| `qwen2.5-0.5b-ppo-gsm8k-lr1e-4` ([W&B](https://forge.coreweave.com/wandb/nl2752-new-york-university/rl-verl/runs/2r01t0os)) | Actor lr 1e-4 (100x). Critic lr stays 1e-5. | `RUN_NAME=... sbatch slurm/ppo_gsm8k.sbatch actor_rollout_ref.actor.optim.lr=1e-4` |
| `qwen2.5-0.5b-ppo-gsm8k-fmt0.1` ([W&B](https://forge.coreweave.com/wandb/nl2752-new-york-university/rl-verl/runs/wymt8pbz)) | Format score 0.1 | `FORMAT_SCORE=0.1 RUN_NAME=... sbatch slurm/ppo_gsm8k.sbatch` |

### 8.1 The format score

`verl/utils/reward_score/gsm8k_reward.py`, loaded through `custom_reward_function`:

```python
answer = extract_solution(solution_str=solution_str, method="strict")
has_format = answer is not None
correct = has_format and answer == ground_truth
score = 1.0 if correct else (float(format_score) if has_format else 0.0)
return {"score": score, "acc": float(correct), "format": float(has_format)}
```

Only `score` is the training reward. Validation logs `acc` as `val-core/openai/gsm8k/acc/mean@1`, which is the number to compare with the baseline's `val-core/openai/gsm8k/reward/mean@1`.

### 8.2 Test accuracy over training

Strict accuracy on the 1,319 test problems, greedy decoding.

| Step | Baseline | lr 1e-4 | Format 0.1 | Format 0.1: has `#### N` |
| --- | --- | --- | --- | --- |
| 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| 10 | 0.005 | 0.000 | 0.012 | 2.5% |
| 20 | 0.236 | 0.000 | 0.324 | 60.5% |
| 30 | 0.423 | 0.000 | 0.459 | 92.3% |
| 50 | 0.494 | 0.000 | 0.493 | 94.3% |
| 100 | 0.528 | 0.000 | 0.530 | |
| 200 | 0.534 | 0.000 | 0.531 | |
| 435 | **0.546** | **0.000** | **0.531** | 97.7% |
| Best | 0.552 (step 270) | 0.000 | 0.552 (step 190) | |

### 8.3 lr 1e-4: the policy collapses in two steps and never recovers

| Step | Train score | `ppo_kl` | `pg_clipfrac` | Entropy | `grad_norm` | Response tokens |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.004 | 1.90 | 25.5% | 0.425 | 2.9 | 235 |
| 2 | 0.000 | 4.75 | 22.3% | 0.049 | 7.3 | 256 |
| 5 | 0.000 | 6.17 | 19.7% | 0.010 | 11.3 | 256 |
| 10 → 435 | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 256 |

For comparison the baseline's `ppo_kl` stays at 0.000 and `pg_clipfrac` under 0.5% for the whole run.

- After step 1 every response is the same degenerate string, 256 tokens of `" " " " " ...`, for every prompt. The test score is 0.000 at every validation from step 0 to 435.
- It is a dead end, not a slow start. The policy is deterministic (entropy 0), so every sample is the same wrong string, the reward is always 0, and the gradient is exactly 0 from step 10 on. Nothing can pull it back.
- **The PPO clip did not prevent it.** Clipping zeroes the gradient of a token only once its ratio has left $[0.8, 1.2]$. It does not bound the size of an optimizer step. At the first mini-batch the ratio is exactly 1 and nothing is clipped, and one Adam step at 1e-4 already moves the policy far outside the range (`ppo_kl` 1.9 after four updates).
- The KL penalty (0.001 × 0.28 per token) is far too small to matter against this.

### 8.4 Format score 0.1: faster start, same end point

- **Faster start.** At step 20 the accuracy is 0.324 against 0.236, and 60% of test responses already have the format. With the 0/1 reward the format is only reinforced through the rare correct answers; with the format score every response containing `#### N` earns something from step 1.
- **Same plateau.** From step 50 on the two runs are indistinguishable: both move between 0.51 and 0.55, and both peak at 0.552. The final 0.531 against 0.546 is inside that step-to-step spread, with one seed each.
- **No sign of gaming.** The risk of a format score is a model that writes `#### <any number>` and stops trying. That did not happen at 0.1: accuracy is not lower than the baseline. A larger value was not tested.
- **Cleaner output, cause unclear.** The doubled `#### N` line is in 0.2% of responses against 79% for the baseline, and 97.7% contain the marker against 95.5%. This is one run of each, so it may be chance in what the policy locked onto early and not an effect of the format score.

### 8.5 Findings

1. **The learning rate is the parameter that can break the run.** 1e-6 trains; 1e-4 destroys the model within two steps, permanently. PPO's clip is not a safety net for a learning rate that is too high.
2. **Watch `actor/ppo_kl`, `actor/pg_clipfrac` and `actor/entropy_loss` in the first steps.** They showed the collapse at step 1 (1.9 and 25%, against 0.000 and 0.1%), nine steps before the gradient went to zero. A healthy run here keeps `ppo_kl` near 0 and lets entropy fall slowly (0.43 → 0.06 over 435 steps, not over 2).
3. **Reward shaping changed the speed, not the result.** The format score reached 0.32 at step 20 against 0.24, then landed on the same ~0.53–0.55 plateau. The limit at this point is the 0.5B model's arithmetic, not the format.
4. **The plateau arrives after about 2 of the 15 epochs.** Steps 60 to 435 add about 0.03 on the test set in every working run, while the training score keeps rising. For this recipe 3 to 4 epochs would give the same test accuracy.
5. **Caveat.** One seed per variant, and the test score moves by about ±0.02 between validations. Differences of that size between runs are not evidence.

## 9. Practical Notes

- **Strict reward = format + correctness.** The base model scores 0.0 on the test set because it almost never writes `#### <number>`, not because it cannot do the arithmetic. Early RL progress is mostly learning the format.
- **Truncation.** 61% of step-1 responses hit the 256-token limit. A truncated response has no final answer and scores 0.
- **Loss is on response tokens only.** `response_mask = attention_mask[:, -response_length:]` masks the prompt and the padding for the actor loss, critic loss, KL and whitening.
- **Reward is sparse.** One nonzero entry per sequence, on the last valid token. With $\gamma = \lambda = 1$ every earlier token sees it through the return.
- **Advantages are whitened per batch**, over tokens, not per sequence. Long responses contribute more tokens to the loss (`loss_agg_mode = token-mean`).
- **KL is in the reward, not the loss.** Direction is $\mathrm{KL}(\pi_{\text{old}} \| \pi_{\text{ref}})$, estimated per token on sampled tokens. The reference model is frozen.
- **`old_log_probs` are recomputed by the actor**, not taken from vLLM. `training/rollout_probs_diff_*` logs the gap between the two.
- **`kl_coef` alone does nothing.** Without `algorithm.use_kl_in_reward=True` the reference policy is never loaded.
- **The reward regex can be gamed.** It reads only the leading digits after `#### `, so `#### 3B/2` counts as `3` (see 7.3). Check samples, not only the score.
- **`pg_loss` is not a progress signal.** It stays near 0 because advantages are whitened. Watch `critic/score/mean` and `val-core/openai/gsm8k/reward/mean@1`.

