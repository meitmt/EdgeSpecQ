#!/usr/bin/env python3
"""Build the 20-sample VL acceptance probe by streaming the first rows of TextVQA (7), ChartQA (7) and
COCO captions (6). Images are saved as PNG under $EDGESPECQ_HOME/probe/vl/; writes probe/vl_probe20.jsonl."""
import json, os, sys
from datasets import load_dataset

HOME = os.path.expanduser(os.environ.get("EDGESPECQ_HOME", "~/edgespecq"))
IMG_DIR = f"{HOME}/probe/vl"


def take(rows, name, split, n, fn):
    k = 0
    for ex in load_dataset(name, split=split, streaming=True):
        rows.append(fn(ex, k))
        k += 1
        if k >= n:
            break
    print(f"{name}: {k}")


def save(img, tag, k):
    p = f"{IMG_DIR}/{tag}_{k}.png"
    img.convert("RGB").save(p)
    return p


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    rows = []
    take(rows, "lmms-lab/textvqa", "validation", 7,
         lambda ex, k: {"src": "textvqa", "id": f"textvqa_{k}", "image": save(ex["image"], "textvqa", k),
                        "prompt": ex["question"] + " Answer, then explain briefly."})
    take(rows, "HuggingFaceM4/ChartQA", "test", 7,
         lambda ex, k: {"src": "chartqa", "id": f"chartqa_{k}", "image": save(ex["image"], "chartqa", k),
                        "prompt": ex["query"] + " Explain your reasoning step by step."})
    take(rows, "lmms-lab/COCO-Caption2017", "val", 6,
         lambda ex, k: {"src": "coco", "id": f"coco_{k}", "image": save(ex["image"], "coco", k),
                        "prompt": "Describe this image in detail."})
    out = f"{HOME}/probe/vl_probe20.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), "VL prompts ->", out)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # streaming datasets can leave non-daemon threads that block interpreter exit
