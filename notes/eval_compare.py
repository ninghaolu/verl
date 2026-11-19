"""Compare the model before and after RL from the trainer's validation dumps (trainer.validation_data_dir).

Usage: python notes/eval_compare.py <val_generations dir> [before_step] [after_step]
strict = "#### <number>" equals the ground truth (the dump's "acc" when the reward has a format score, else "score"); flexible = the last number in the response equals it.
"""
import json
import os
import re
import sys

import pandas as pd


def flexible(output, ground_truth):
    numbers = [n for n in re.findall(r"(\-?[0-9\.\,]+)", output) if n not in ("", ".")]
    if not numbers:
        return 0.0
    try:
        return float(float(numbers[-1].replace(",", "").rstrip(".")) == float(ground_truth))
    except ValueError:
        return 0.0


def load(directory, step, ground_truths):
    rows = [json.loads(line) for line in open(os.path.join(directory, f"{step}.jsonl"))]
    for row in rows:
        row["question"] = row["input"].split("\nuser\n")[1].rsplit("\nassistant\n", 1)[0]
        row["ground_truth"] = ground_truths[row["question"]]
    return rows


def main():
    directory = sys.argv[1]
    steps = sorted(int(name.split(".")[0]) for name in os.listdir(directory))
    before = int(sys.argv[2]) if len(sys.argv) > 2 else steps[0]
    after = int(sys.argv[3]) if len(sys.argv) > 3 else steps[-1]
    test = pd.read_parquet(os.path.expanduser("~/data/gsm8k/test.parquet"))
    ground_truths = {row["prompt"][0]["content"]: row["reward_model"]["ground_truth"] for _, row in test.iterrows()}
    for step in (before, after):
        rows = load(directory, step, ground_truths)
        n = len(rows)
        print(
            f"step {step}: n={n}"
            f" strict={sum(r.get('acc', r['score']) for r in rows) / n:.3f}"
            f" flexible={sum(flexible(r['output'], r['ground_truth']) for r in rows) / n:.3f}"
            f" has_marker={sum('####' in r['output'] for r in rows) / n:.3f}"
            f" marker_twice={sum(r['output'].count('####') > 1 for r in rows) / n:.3f}"
            f" mean_chars={sum(len(r['output']) for r in rows) / n:.0f}"
        )


if __name__ == "__main__":
    main()
