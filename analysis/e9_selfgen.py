#!/usr/bin/env python3
"""E9: self-generated reference corpus for draft-vocabulary ranking.

The target answers prompts that do not overlap the probes (GSM8K / MetaMathQA-MATH / MBPP train, UltraChat test_sft,
Dolly-15k-ja), greedy, through llama-server with parallel slots. Responses are written to
$EDGESPECQ_HOME/data/e9corpus/self_{en,ja}.txt, which analysis/e9_corpus_freq.py then counts (--sources self_en,self_ja).

usage: e9_selfgen.py <llama_dir> <target.gguf> [--per-source 150] [--max-tokens 512] [--parallel 8]
"""
import argparse, json, os, subprocess, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset

HOME = os.path.expanduser(os.environ.get("EDGESPECQ_HOME", "~/edgespecq"))
PORT = int(os.environ.get("PORT", "8093"))


def prompts(k):
    en, ja = [], []
    ds = load_dataset("openai/gsm8k", "main", split="train")
    en += [ds[i]["question"] + "\nSolve step by step." for i in range(k)]
    ds = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
    en += [r["query"] + "\nSolve step by step." for r, _ in zip((r for r in ds if r["type"].startswith("MATH")), range(k))]
    ds = load_dataset("google-research-datasets/mbpp", "full", split="train")
    en += [ds[i]["text"] + "\nYour code should pass this test:\n" + ds[i]["test_list"][0] for i in range(k)]
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    en += [ds[i]["prompt"] for i in range(k)]
    ds = load_dataset("kunishou/databricks-dolly-15k-ja", split="train")
    ja += [(r["instruction"] + ("\n\n" + r["input"] if r["input"] else "")) for r in ds.select(range(2 * k))]
    return en, ja


def ask(prompt, max_tokens):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "temperature": 0,
                       "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("llama_dir"); ap.add_argument("target")
    ap.add_argument("--per-source", type=int, default=150)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--parallel", type=int, default=8)
    a = ap.parse_args()
    en, ja = prompts(a.per_source)
    srv = subprocess.Popen([f"{a.llama_dir}/build/bin/llama-server", "-m", a.target, "-ngl", "99", "-fa", "on",
                            "-np", str(a.parallel), "-c", str(a.parallel * 2048), "--port", str(PORT)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); break
            except Exception:
                time.sleep(1)
        os.makedirs(f"{HOME}/data/e9corpus", exist_ok=True)
        for name, ps in (("self_en", en), ("self_ja", ja)):
            t0 = time.time()
            with ThreadPoolExecutor(a.parallel) as ex:
                outs = list(ex.map(lambda p: ask(p, a.max_tokens), ps))
            with open(f"{HOME}/data/e9corpus/{name}.txt", "w", encoding="utf-8") as f:
                f.write("\n\n".join(o.strip() for o in outs) + "\n")
            print(f"{name}: {len(ps)} prompts, {sum(map(len, outs)):,d} chars, {time.time() - t0:.0f}s", flush=True)
    finally:
        srv.terminate(); srv.wait()


if __name__ == "__main__":
    main()
