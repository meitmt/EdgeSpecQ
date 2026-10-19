#!/usr/bin/env python3
"""Why DSpark accepts less on Japanese: collect, per request, every freshly drafted round (draft tokens, accepted
count, the target's tokens; "E9S" lines of the timing build's llama-server) and the target's probability of every
generated token (logprobs), for matched English / Japanese inputs:

  en_caption / ja_caption   the same 15 COCO images (dev120 captioning), described in English / Japanese
  en_text / ja_text         15 open-ended English chat prompts (UltraChat test_sft) / 15 ELYZA-tasks-100 tasks (ja75)

Writes one JSON line per request to <out.jsonl>: {group, id, text, logprobs: [[token, logprob], ...],
rounds: [{n_acc, draft: [...], ids: [...]}, ...]}. Analysis: analysis/e9_lang_analyze.py.

Note: the server merges tokens that end inside a multi-byte UTF-8 character into the next logprobs entry, so for
Japanese the logprobs list is per character run, not per token.

usage: e9_lang_rounds.py <llama_dir> <target.gguf> <drafter.gguf|none> <out.jsonl> [--n-max 8] [--n 15]
"""
import argparse, base64, json, os, re, subprocess, time, urllib.request
from datasets import load_dataset

HOME = os.path.expanduser(os.environ.get("EDGESPECQ_HOME", "~/edgespecq"))
PORT = int(os.environ.get("PORT", 8099))


def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


def samples(n):
    dev = [json.loads(l) for l in open(f"{HOME}/eval/dev120.jsonl", encoding="utf-8")]
    ja = [json.loads(l) for l in open(f"{HOME}/eval/ja75.jsonl", encoding="utf-8")]
    caps = [r for r in dev if r["src"] == "captioning"][:n]
    out = [("en_caption", r["id"], "Describe this image in detail.", r["images"]) for r in caps]
    out += [("ja_caption", r["id"], r["turns"][0], r["images"]) for r in ja if r["src"] == "ja_caption"][:n]
    out += [("ja_text", r["id"], r["turns"][0], []) for r in ja if r["src"] == "ja_text"][:n]
    uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    out += [("en_text", f"ultrachat_{i}", uc[i]["prompt"], []) for i in range(n)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("llama_dir"); ap.add_argument("target"); ap.add_argument("drafter"); ap.add_argument("out")
    ap.add_argument("--n-max", type=int, default=8)
    ap.add_argument("--n", type=int, default=15)
    a = ap.parse_args()
    todo = samples(a.n)
    args = [f"{a.llama_dir}/build/bin/llama-server", "-m", a.target,
            "--mmproj", f"{HOME}/models/target/mmproj-LFM2.5-VL-3B-Q8_0.gguf", "-ngl", "99", "-fa", "on",
            "-c", "8192", "-np", "1", "--port", str(PORT)]
    # drafter "none": target only. With speculation the server reports ~1 for every accepted draft token, so the
    # target's real per-token probabilities come from a target-only run
    if a.drafter != "none":
        args += ["-md", a.drafter, "-ngld", "99", "--spec-type", "draft-dspark",
                 "--spec-draft-n-max", str(a.n_max), "--spec-draft-n-min", "0"]
    log_path = a.out + ".server.log"
    log = open(log_path, "w")
    srv = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
    pos = 0
    def new_log():
        nonlocal pos
        log.flush()
        with open(log_path, errors="replace") as f:
            f.seek(pos); s = f.read(); pos = f.tell()
        return s
    try:
        for _ in range(600):
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200:
                    break
            except Exception:
                if srv.poll() is not None:
                    raise SystemExit(f"server exited; see {log_path}")
                time.sleep(1)
        post({"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 8, "temperature": 0})
        new_log()
        with open(a.out, "w", encoding="utf-8") as fo:
            for group, sid, prompt, images in todo:
                content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," +
                            base64.b64encode(open(p, "rb").read()).decode()}} for p in images]
                content.append({"type": "text", "text": prompt})
                # post_sampling_probs false: the target's raw distribution (with greedy sampling the post-sampling
                # probability of the chosen token is always ~1)
                r = post({"messages": [{"role": "user", "content": content}], "max_tokens": 512, "temperature": 0,
                          "logprobs": True, "top_logprobs": 2, "post_sampling_probs": False})
                lp = (r["choices"][0].get("logprobs") or {}).get("content") or []
                rounds = [{"n_acc": int(m[0]), "draft": [int(x) for x in m[1].split(",")],
                           "ids": [int(x) for x in m[2].split(",")]}
                          for m in re.findall(r"E9S n_acc=(\d+) draft=([0-9,]+) ids=([0-9,]+)", new_log())]
                fo.write(json.dumps({"group": group, "id": sid, "text": r["choices"][0]["message"]["content"],
                                     "logprobs": [[x["token"], x["logprob"], x.get("bytes"),
                                                   [t["logprob"] for t in x.get("top_logprobs", [])]] for x in lp],
                                     "rounds": rounds},
                                    ensure_ascii=False) + "\n")
                print(f"{group:10s} {sid:24s} tokens={len(lp):4d} rounds={len(rounds):4d} "
                      f"acc/round={sum(x['n_acc'] for x in rounds) / max(len(rounds), 1):.2f}", flush=True)
    finally:
        srv.terminate(); srv.wait(); log.close()


if __name__ == "__main__":
    main()
