#!/usr/bin/env python3
"""Build the Japanese pilot probe: every 10th task of ELYZA-tasks-100 (10 prompts, mixed task types).
Deterministic. Writes $EDGESPECQ_HOME/probe/ja_probe10.jsonl in the same format as track_t_probe20.jsonl."""
import json, os
from datasets import load_dataset

HOME = os.path.expanduser(os.environ.get("EDGESPECQ_HOME", "~/edgespecq"))


def main():
    out = f"{HOME}/probe/ja_probe10.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ds = load_dataset("elyza/ELYZA-tasks-100", split="test")
    rows = [{"src": "elyza100", "id": f"elyza_{i}", "prompt": ds[i]["input"]} for i in range(0, len(ds), 10)]
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), "prompts ->", out)


if __name__ == "__main__":
    main()
