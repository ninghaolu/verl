"""GSM8K reward with a format score, for custom_reward_function.

The built-in verl/utils/reward_score/gsm8k.py gives 1 for a correct answer and 0 otherwise. This one adds partial
credit for a response that follows the format ("#### <number>") with a wrong number:

    no "#### <number>"        -> 0
    "#### <wrong number>"     -> format_score
    "#### <correct number>"   -> 1

Use it with
    custom_reward_function.path=verl/utils/reward_score/gsm8k_reward.py
    custom_reward_function.name=compute_score
    +custom_reward_function.reward_kwargs.format_score=0.1

It returns a dict: "score" is the training reward; "acc" and "format" are logged only. Validation reports "acc" as
val-core/openai/gsm8k/acc/mean@1, so the test accuracy stays comparable with runs that have no format score.
"""

from verl.utils.reward_score.gsm8k import extract_solution


def compute_score(data_source, solution_str, ground_truth, extra_info=None, format_score=0.1):
    answer = extract_solution(solution_str=solution_str, method="strict")
    has_format = answer is not None
    correct = has_format and answer == ground_truth
    if correct:
        score = 1.0
    elif has_format:
        score = float(format_score)
    else:
        score = 0.0
    return {"score": score, "acc": float(correct), "format": float(has_format)}
