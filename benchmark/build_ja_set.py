#!/usr/bin/env python3
"""Build the Japanese pre-confirmation set (E7 pilot, proj v1.9): 60 samples, 15 per source, seeded.

  ja_reasoning   JMMMU (JMMMU/JMMMU, MIT)                          multiple choice, single-image items only,
                                                                    Japanese step-by-step suffix
  ja_vqa_wild    JA-VLM-Bench-In-the-Wild (SakanaAI, license not stated on the card; images from Unsplash)
                                                                    open-ended questions
  ja_vqa_short   JA-VG-VQA-500 test (SakanaAI, CC-BY-4.0)          first QA pair of each image, short answers
  ja_text        ELYZA-tasks-100 (CC-BY-SA-4.0), text only          excluding every 10th task (benchmark/build_probe_ja.py)

Used only for local evaluation; nothing from these datasets is committed. Images go to $EDGESPECQ_HOME/eval/img/,
the set to $EDGESPECQ_HOME/eval/ja60.jsonl (same row format as benchmark/build_eval_set.py).
"""
import ast, json, os, random, sys
from datasets import Image, get_dataset_config_names, load_dataset

HOME = os.path.expanduser(os.environ.get("EDGESPECQ_HOME", "~/edgespecq"))
OUT = f"{HOME}/eval"
SEED, N = 2026, 15
COT_JA = "\n\n段階的に考えてください。まず理由を詳しく説明し、最後の行に「Answer: $LETTER」の形式で答えを書いてください。"


def save(img, name):
    p = f"{OUT}/img/{name}.png"
    if not os.path.exists(p):
        img.convert("RGB").save(p)
    return p


def sample(n_rows, exclude=()):
    idx = [i for i in range(n_rows) if i not in set(exclude)]
    random.Random(SEED).shuffle(idx)
    return idx[:N]


def jmmmu():
    rows = []
    for cfg in get_dataset_config_names("JMMMU/JMMMU"):
        ds = load_dataset("JMMMU/JMMMU", cfg, split="test")
        raw = ds
        for k in range(1, 8):
            raw = raw.cast_column(f"image_{k}", Image(decode=False))
        cols = [raw[f"image_{k}"] for k in range(1, 8)]
        for i in range(len(ds)):
            if cols[0][i] is not None and all(c[i] is None for c in cols[1:]) and ds[i]["question_type"] == "multiple-choice":
                rows.append((cfg, i))
    picks = sample(len(rows))
    out = []
    for j in picks:
        cfg, i = rows[j]
        r = load_dataset("JMMMU/JMMMU", cfg, split="test")[i]
        opts = ast.literal_eval(r["options"]) if isinstance(r["options"], str) else r["options"]
        text = r["question"] + "\n" + "\n".join(f"{chr(65 + k)}. {o}" for k, o in enumerate(opts)) + COT_JA
        out.append({"src": "ja_reasoning", "id": f"jmmmu_{r['id']}", "turns": [text],
                    "images": [save(r["image_1"], f"jmmmu_{r['id']}")]})
    return out


def ja_wild():
    ds = load_dataset("SakanaAI/JA-VLM-Bench-In-the-Wild", split="test")
    return [{"src": "ja_vqa_wild", "id": f"javlmwild_{i}", "turns": [ds[i]["question"]],
             "images": [save(ds[i]["image"], f"javlmwild_{i}")]} for i in sample(len(ds))]


def ja_vg():
    ds = load_dataset("SakanaAI/JA-VG-VQA-500", split="test")
    out = []
    for i in sample(len(ds)):
        r = ds[i]
        qas = r["qas"] if isinstance(r["qas"], list) else ast.literal_eval(r["qas"])
        out.append({"src": "ja_vqa_short", "id": f"javgvqa_{r['image_id']}", "turns": [qas[0]["question"]],
                    "images": [save(r["image"], f"javgvqa_{r['image_id']}")]})
    return out


def elyza():
    ds = load_dataset("elyza/ELYZA-tasks-100", split="test")
    return [{"src": "ja_text", "id": f"elyza_{i}", "turns": [ds[i]["input"]], "images": []}
            for i in sample(len(ds), exclude=range(0, len(ds), 10))]


def main():
    os.makedirs(f"{OUT}/img", exist_ok=True)
    rows = []
    for name, f in (("ja_reasoning", jmmmu), ("ja_vqa_wild", ja_wild), ("ja_vqa_short", ja_vg), ("ja_text", elyza)):
        r = f()
        print(f"{name:13s} {len(r)}", flush=True)
        rows += r
    with open(f"{OUT}/ja60.jsonl", "w") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), "->", f"{OUT}/ja60.jsonl")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
