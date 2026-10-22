#!/usr/bin/env python3
"""Workload and n_max check for the official DSpark drafter on unpatched llama.cpp.

Prompted by https://huggingface.co/LiquidAI/LFM2.5-VL-3B-DSpark-GGUF/discussions/3 (15 % acceptance on Wikitext
with n_max 9): Q8_0 target and the official F16 drafter, n_max 8 / 9 / 10 (10 is clamped to the drafter's
dflash.block_size of 9), target-only as the reference, on two prompt sets:
  chat: the 20 chat-style text prompts of benchmark/build_probe_text.py (GSM8K, MATH-500, HumanEval, MBPP, MT-Bench)
  wiki: the first 10 Wikitext-103 articles longer than 1500 characters (analysis/e9_corpus_freq.py cache),
        700 characters each, after "Continue the following Wikipedia article."
each greedy and with the model card's sampling (T 0.2, top-k 50, seed 0), up to 256 generated tokens.

usage: nmax_workload_check.py [--out results/summary/nmax_workload_rtx5070.json] [--raw raw.json]
  --raw: summarize an earlier run's raw counters instead of measuring
Env: EDGESPECQ_HOME (default ~/edgespecq), LLAMA_DIR (default $EDGESPECQ_HOME/llama-edge), LIBS (default
$EDGESPECQ_HOME/libs/5070/master, the unpatched runtime).
"""
import argparse, json, os, re, subprocess, time, urllib.request

H = os.path.expanduser(os.environ.get("EDGESPECQ_HOME", "~/edgespecq"))
BIN = f"{os.environ.get('LLAMA_DIR', f'{H}/llama-edge')}/build/bin/llama-server"
LIBS = os.environ.get("LIBS", f"{H}/libs/5070/master")
TGT = f"{H}/models/target/LFM2.5-VL-3B-Q8_0.gguf"
DRAFT = f"{H}/models/draft/LFM2.5-VL-3B-DSpark-F16.gguf"
PORT = int(os.environ.get("PORT", 8193))
N_MAX = (0, 8, 9, 10)  # 0 = target only
BLOCK_SIZE = 9
SAMPLING = {"greedy": {"temperature": 0}, "t0.2_k50": {"temperature": 0.2, "top_k": 50, "seed": 0}}


def prompt_sets():
    chat = [json.loads(l)["prompt"] for l in open(f"{H}/probe/track_t_probe20.jsonl")]
    raw = open(f"{H}/data/e9corpus/en_wiki.txt", encoding="utf-8").read()
    arts = [a for a in re.split(r"\n = (?=[^=])", raw) if len(a) > 1500][:10]
    wiki = ["Continue the following Wikipedia article.\n\n" + a[:700] for a in arts]
    return {"chat": chat, "wiki": wiki}


def server(n_max):
    args = [BIN, "-m", TGT, "-ngl", "99", "-fa", "on", "-c", "8192", "-np", "1", "--port", str(PORT)]
    if n_max:
        args += ["-md", DRAFT, "-ngld", "99", "--spec-type", "draft-dspark", "--spec-draft-n-max", str(n_max),
                 "--spec-draft-n-min", "0"]
    log = open(f"/tmp/nmax_workload_{n_max}.log", "w")
    p = subprocess.Popen(args, env={**os.environ, "LD_LIBRARY_PATH": LIBS}, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            if b"ok" in urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2).read():
                return p
        except Exception:
            time.sleep(1)
    p.terminate()
    raise RuntimeError("server did not start")


def ask(prompt, samp):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 256, **samp}
    req = urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))["timings"]


def measure():
    """raw counters {"n_max|set|sampling": {n, ms, dn, da}}: generated tokens, decode ms, drafted, accepted"""
    sets, raw = prompt_sets(), {}
    for n_max in N_MAX:
        p = server(n_max)
        try:
            for sname, prompts in sets.items():
                for mname, samp in SAMPLING.items():
                    tot = {"n": 0, "ms": 0.0, "dn": 0, "da": 0}
                    for pr in prompts:
                        t = ask(pr, samp)
                        tot["n"] += t["predicted_n"]; tot["ms"] += t["predicted_ms"]
                        tot["dn"] += t.get("draft_n", 0) or 0; tot["da"] += t.get("draft_n_accepted", 0) or 0
                    raw[f"{n_max}|{sname}|{mname}"] = tot
        finally:
            p.terminate(); p.wait()
    return raw


def summarize(raw):
    rows = []
    for key, t in raw.items():
        n_max, sname, mname = key.split("|")
        n_max = int(n_max)
        if not n_max:
            continue
        base = raw[f"0|{sname}|{mname}"]
        tps, b_tps = 1000 * t["n"] / t["ms"], 1000 * base["n"] / base["ms"]
        eff = min(n_max, BLOCK_SIZE)  # every round drafts eff tokens (n_min 0)
        rows.append({"prompts": sname, "sampling": mname, "n_max": n_max, "n_max_effective": eff,
                     "acceptance": round(t["da"] / t["dn"], 4), "accepted_per_round": round(t["da"] / (t["dn"] / eff), 3),
                     "decode_tps": round(tps, 1), "target_only_tps": round(b_tps, 1), "speedup": round(tps / b_tps, 3),
                     "generated_tokens": t["n"], "drafted": t["dn"], "accepted": t["da"]})
    rows.sort(key=lambda r: (r["prompts"], r["sampling"], r["n_max"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/summary/nmax_workload_rtx5070.json")
    ap.add_argument("--raw")
    a = ap.parse_args()
    raw = json.load(open(a.raw)) if a.raw else measure()
    rows = summarize(raw)
    meta = {"_doc": __doc__.split("\n\n")[1].replace("\n", " "), "llama_cpp": "620d2225c, unpatched",
            "gpu": "RTX 5070", "target": "LFM2.5-VL-3B-Q8_0", "drafter": "LFM2.5-VL-3B-DSpark-F16 (official)",
            "max_tokens": 256, "rows": rows, "raw": raw}
    json.dump(meta, open(a.out, "w"), indent=1)
    print(f"{'prompts':7s} {'sampling':9s} {'n_max':>5s} {'accept':>7s} {'acc/round':>9s} {'t/s':>7s} {'speedup':>7s}")
    for r in rows:
        print(f"{r['prompts']:7s} {r['sampling']:9s} {r['n_max']:5d} {r['acceptance']:7.3f} "
              f"{r['accepted_per_round']:9.2f} {r['decode_tps']:7.1f} {r['speedup']:6.2f}x")


if __name__ == "__main__":
    main()
