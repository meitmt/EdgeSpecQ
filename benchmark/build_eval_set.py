#!/usr/bin/env python3
"""Build the six-category multimodal evaluation sets (smoke24 / dev120 / final600), following the task categories of
MMSpec and of Liquid's DSpark-VL speedup table. MMSpec's own sample list is not public, so samples are drawn from
the same source datasets with a fixed seed:

  general_vqa    GQA testdev-balanced (lmms-lab/GQA)            question as is
  text_vqa       TextVQA validation (lmms-lab/textvqa)            question as is
  captioning     COCO 2017 val (lmms-lab/COCO-Caption2017)       "Describe this image in detail."
  chart_vqa      CharXiv validation (princeton-nlp/CharXiv)       reasoning question
  reasoning      MMMU-Pro standard, 10 options (MMMU/MMMU_Pro)    official CoT prompt; at most 2 images
  multi_turn     MM-MT-Bench (mistralai/MM-MT-Bench)              conversations with >= 2 user turns; the model
                                                                  answers every turn itself (reference answers unused)

Per category a seeded permutation is cut into dev = [0, 20) and final = [20, 120); smoke = the first 4 of dev.
MM-MT-Bench has only 23 multi-turn conversations, so its final share is the remaining 3. Rows already used by the
VL probe (first rows of TextVQA / COCO) are excluded. Images are written as PNG under
$EDGESPECQ_HOME/eval/img/; sets go to $EDGESPECQ_HOME/eval/{smoke24,dev120,final600}.jsonl.

Row format: {"src", "id", "turns": [str, ...], "images": [path, ...]} (images belong to the first user turn).
"""
import json, os, random, sys
from datasets import Image, load_dataset

HOME = os.path.expanduser(os.environ.get("EDGESPECQ_HOME", "~/edgespecq"))
OUT = f"{HOME}/eval"
SEED = 2026
N_DEV, N_FINAL, N_SMOKE = 20, 100, 4
MMMU_COT = ("Answer the following multiple choice question. The last line of your response should be of the following "
            "format: 'Answer: $LETTER' (without quotes) where LETTER is one of the options. Think step by step before "
            "answering.\n\n")


def save(img, name):
    p = f"{OUT}/img/{name}.png"
    if not os.path.exists(p):
        img.convert("RGB").save(p)
    return p


def pick(n_rows, exclude=()):
    idx = [i for i in range(n_rows) if i not in set(exclude)]
    random.Random(SEED).shuffle(idx)
    return idx[:N_DEV], idx[N_DEV:N_DEV + N_FINAL]


def gqa():
    q = load_dataset("lmms-lab/GQA", "testdev_balanced_instructions", split="testdev")
    imgs = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev")
    by_id = {r: i for i, r in enumerate(imgs["id"])}
    def row(i):
        r = q[i]
        return {"id": f"gqa_{r['id']}", "turns": [r["question"]],
                "images": [save(imgs[by_id[r["imageId"]]]["image"], f"gqa_{r['imageId']}")]}
    return len(q), row, ()


def textvqa():
    ds = load_dataset("lmms-lab/textvqa", split="validation")
    def row(i):
        r = ds[i]
        return {"id": f"textvqa_{r['question_id']}", "turns": [r["question"]],
                "images": [save(r["image"], f"textvqa_{r['question_id']}")]}
    return len(ds), row, range(7)  # benchmark/build_probe_vl.py uses the first 7


def coco():
    ds = load_dataset("lmms-lab/COCO-Caption2017", split="val")
    def row(i):
        r = ds[i]
        key = str(r.get("question_id") or r.get("file_name") or i).replace("/", "_").replace(".jpg", "")
        return {"id": f"coco_{key}", "turns": ["Describe this image in detail."],
                "images": [save(r["image"], f"coco_{key}")]}
    return len(ds), row, range(6)  # benchmark/build_probe_vl.py uses the first 6


def charxiv():
    ds = load_dataset("princeton-nlp/CharXiv", split="validation")
    def row(i):
        r = ds[i]
        return {"id": f"charxiv_{i}", "turns": [r["reasoning_q"]], "images": [save(r["image"], f"charxiv_{i}")]}
    return len(ds), row, ()


def mmmu_pro():
    ds = load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split="test")
    raw = ds.cast_column("image_1", Image(decode=False))  # count images without decoding them
    for k in range(2, 8):
        raw = raw.cast_column(f"image_{k}", Image(decode=False))
    cols = [raw[f"image_{k}"] for k in range(1, 8)]
    n_img = [sum(c[i] is not None for c in cols) for i in range(len(ds))]
    def row(i):
        r = ds[i]
        opts = eval(r["options"]) if isinstance(r["options"], str) else r["options"]  # stored as a Python list literal
        text = MMMU_COT + r["question"] + "\n" + "\n".join(f"{chr(65 + k)}. {o}" for k, o in enumerate(opts))
        images = [save(r[f"image_{k}"], f"mmmupro_{r['id']}_{k}") for k in range(1, 8) if r[f"image_{k}"] is not None]
        return {"id": f"mmmupro_{r['id']}", "turns": [text], "images": images}
    return len(ds), row, [i for i, n in enumerate(n_img) if n == 0 or n > 2]


def mm_mt_bench():
    ds = load_dataset("mistralai/MM-MT-Bench", split="eval")
    convs = [json.loads(c) if isinstance(c, str) else c for c in ds["conversation"]]
    def user_texts(conv):
        return [" ".join(p["text"] for p in m["content"] if p.get("type") == "text")
                for m in conv if m["role"] == "user"]
    def row(i):
        return {"id": f"mmmtbench_{i}", "turns": user_texts(convs[i]),
                "images": [save(ds[i]["image"], f"mmmtbench_{i}")]}
    return len(ds), row, [i for i, c in enumerate(convs) if len(user_texts(c)) < 2]


CATEGORIES = {"general_vqa": gqa, "text_vqa": textvqa, "captioning": coco, "chart_vqa": charxiv,
              "reasoning": mmmu_pro, "multi_turn": mm_mt_bench}


def main():
    os.makedirs(f"{OUT}/img", exist_ok=True)
    sets = {"smoke24": [], "dev120": [], "final600": []}
    for cat, load in CATEGORIES.items():
        n, row, exclude = load()
        dev, final = pick(n, exclude)
        rows_dev = [{"src": cat, **row(i)} for i in dev]
        rows_final = [{"src": cat, **row(i)} for i in final]
        sets["dev120"] += rows_dev
        sets["smoke24"] += rows_dev[:N_SMOKE]
        sets["final600"] += rows_final
        print(f"{cat:12s} pool={n - len(set(exclude)):5d} dev={len(rows_dev)} final={len(rows_final)}", flush=True)
    for name, rows in sets.items():
        with open(f"{OUT}/{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} -> {OUT}/{name}.jsonl")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # datasets can leave non-daemon threads that block interpreter exit
