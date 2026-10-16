#!/usr/bin/env python3
"""E9: token frequencies of the LFM2.5 vocabulary on corpora that do not overlap the probes.

Each source is streamed from the HF Hub up to a character budget, cached as text under $EDGESPECQ_HOME/data/e9corpus/,
tokenized with llama-tokenize against the target GGUF (so ids match the deployed vocab exactly) and counted.
Writes results/e9/freq_<source>.npy (int64 counts, length n_vocab).

usage: e9_corpus_freq.py <llama_dir> <target.gguf> [--sources en_chat,en_wiki,code,math,ja_wiki,ja_chat]
"""
import argparse, os, subprocess
import numpy as np
from datasets import load_dataset
from gguf import GGUFReader

HOME = os.path.expanduser(os.environ.get("EDGESPECQ_HOME", "~/edgespecq"))

# name: (dataset, config, split, text extractor, char budget)
SOURCES = {
    "en_chat": ("HuggingFaceH4/ultrachat_200k", None, "train_sft",
                lambda r: "\n".join(m["content"] for m in r["messages"] if m["role"] == "assistant"), 12_000_000),
    "en_wiki": ("Salesforce/wikitext", "wikitext-103-raw-v1", "train", lambda r: r["text"], 12_000_000),
    "code":    ("codeparrot/codeparrot-clean-valid", None, "train", lambda r: r["content"], 12_000_000),
    "math":    ("meta-math/MetaMathQA", None, "train", lambda r: r["response"], 12_000_000),
    "ja_wiki": ("wikimedia/wikipedia", "20231101.ja", "train", lambda r: r["text"], 5_000_000),
    "ja_chat": ("kunishou/databricks-dolly-15k-ja", None, "train", lambda r: r["output"], 5_000_000),
}
CHUNK = 1_000_000  # characters per llama-tokenize call


def fetch(name):
    path = f"{HOME}/data/e9corpus/{name}.txt"
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds_name, cfg, split, get, budget = SOURCES[name]
    ds = load_dataset(ds_name, cfg, split=split, streaming=True)
    n = 0
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        for r in ds:
            t = get(r).strip()
            if not t:
                continue
            f.write(t + "\n\n")
            n += len(t) + 2
            if n >= budget:
                break
    os.rename(path + ".tmp", path)
    return path


def count(binary, target, path, n_vocab):
    counts = np.zeros(n_vocab, dtype=np.int64)
    text = open(path, encoding="utf-8").read()
    for i in range(0, len(text), CHUNK):
        out = subprocess.run([binary, "-m", target, "--stdin", "--ids", "--no-bos", "--no-parse-special",
                              "--log-disable"], input=text[i:i + CHUNK], capture_output=True, text=True, check=True)
        ids = np.array([int(x) for x in out.stdout.strip().strip("[]").split(",") if x.strip()], dtype=np.int64)
        counts += np.bincount(ids, minlength=n_vocab)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("llama_dir"); ap.add_argument("target")
    ap.add_argument("--sources", default=",".join(SOURCES))
    a = ap.parse_args()
    n_vocab = len(GGUFReader(a.target).get_field("tokenizer.ggml.tokens").data)
    binary = f"{a.llama_dir}/build/bin/llama-tokenize"
    os.makedirs(f"{HOME}/results/e9", exist_ok=True)
    for name in a.sources.split(","):
        path = fetch(name)
        c = count(binary, a.target, path, n_vocab)
        np.save(f"{HOME}/results/e9/freq_{name}.npy", c)
        print(f"{name:8s} chars={os.path.getsize(path):>11,d}B tokens={c.sum():>10,d} distinct={int((c > 0).sum()):>7,d}"
              f" / {n_vocab}", flush=True)


if __name__ == "__main__":
    main()
