#!/usr/bin/env python3
"""Batch-invariance test without speculative decoding.

Evaluates the same prompts with different physical batch sizes (-ub) and compares next-token top-5 logprobs
against ub=1. If logprobs change with ub, verification (which evaluates n_max+1 tokens at once) cannot be
bit-identical to single-token decoding, and greedy outputs may diverge at near-ties.

usage: batch_invariance.py <llama_dir> <target.gguf> <probe.jsonl> [ngl=99]
"""
import json, os, subprocess, sys, time, urllib.request

HOME = os.path.expanduser(os.environ.get("EDGESPECQ_HOME", "~/edgespecq"))
PORT = int(os.environ.get("PORT", 8093))
UBS = (2, 9, 512)


def run(server, target, ngl, prompts, ub):
    os.makedirs(f"{HOME}/logs/batch_invariance", exist_ok=True)
    log = open(f"{HOME}/logs/batch_invariance/{os.path.basename(target)}.ngl{ngl}.ub{ub}.log", "w")
    srv = subprocess.Popen([server, "-m", target, "-ngl", ngl, "-fa", "on", "-c", "2048", "-np", "1",
                            "-b", "512", "-ub", str(ub), "--port", str(PORT), "-t", str(os.cpu_count())],
                           stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
        res = []
        for p in prompts:
            body = {"prompt": p, "n_predict": 1, "n_probs": 5, "temperature": 0, "cache_prompt": False}
            r = json.load(urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{PORT}/completion", data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"}), timeout=600))
            res.append([(t["token"], t["logprob"]) for t in r["completion_probabilities"][0]["top_logprobs"]])
        return res
    finally:
        srv.terminate()
        srv.wait()
        log.close()


def main():
    llama_dir, target, probe = sys.argv[1:4]
    ngl = sys.argv[4] if len(sys.argv) > 4 else "99"
    server = f"{llama_dir}/build/bin/llama-server"
    prompts = [json.loads(l)["prompt"] for l in open(probe)][::2]
    ref = run(server, target, ngl, prompts, 1)
    print(f"target={os.path.basename(target)} ngl={ngl}  ({len(prompts)} prompts, reference ub=1)")
    for ub in UBS:
        cur = run(server, target, ngl, prompts, ub)
        maxd = max(abs(a[1] - b[1]) for R, C in zip(ref, cur) for a, b in zip(R, C) if a[0] == b[0])
        flips = sum(R[0][0] != C[0][0] for R, C in zip(ref, cur))
        exact = sum(R == C for R, C in zip(ref, cur))
        print(f"  ub={ub:3d}: bit-identical top-5 on {exact}/{len(ref)} prompts; "
              f"max |dlogprob| {maxd:.5f}; top-1 flips {flips}")


if __name__ == "__main__":
    main()
