#!/usr/bin/env python3
"""Build the 20-prompt text acceptance probe: the first 4 rows of each dataset used in Liquid's DSpark blog
(GSM8K, MATH-500, HumanEval, MBPP, MT-Bench). Deterministic. Writes $EDGESPECQ_HOME/probe/track_t_probe20.jsonl."""
import json, os
from datasets import load_dataset

HOME = os.path.expanduser(os.environ.get("EDGESPECQ_HOME", "~/edgespecq"))
N = 4


def main():
    out = f"{HOME}/probe/track_t_probe20.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows += [{"src": "gsm8k", "id": f"gsm8k_{i}", "prompt": ds[i]["question"] + "\nSolve step by step."} for i in range(N)]
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows += [{"src": "math500", "id": ds[i]["unique_id"], "prompt": ds[i]["problem"] + "\nSolve step by step."}
             for i in range(N)]
    ds = load_dataset("openai/openai_humaneval", split="test")
    rows += [{"src": "humaneval", "id": ds[i]["task_id"],
              "prompt": "Complete the following Python function.\n\n" + ds[i]["prompt"]} for i in range(N)]
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    rows += [{"src": "mbpp", "id": f"mbpp_{ds[i]['task_id']}",
              "prompt": ds[i]["prompt"] + "\nYour code should pass this test:\n" + ds[i]["test_list"][0]}
             for i in range(N)]
    ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    rows += [{"src": "mtbench", "id": f"mtbench_{ds[i]['prompt_id']}", "prompt": ds[i]["prompt"][0]} for i in range(N)]
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), "prompts ->", out)


if __name__ == "__main__":
    main()
