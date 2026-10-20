#!/usr/bin/env python3
"""Batch-invariance test without speculative decoding.

Evaluates the same prompts with different physical batch sizes (-ub) and compares next-token top-5 logprobs
against ub=1. If logprobs change with ub, verification (which evaluates n_max+1 tokens at once) cannot be
bit-identical to single-token decoding, and greedy outputs may diverge at near-ties.

usage: batch_invariance.py <llama_dir> <target.gguf> <probe.jsonl> [ngl=99]
       batch_invariance.py --platform <platform> <config> <probe.jsonl>
         any platform of benchmark/configs.json, including adb ones (phone CPU / Hexagon NPU): the server is started
         exactly as regression/run_eval.py starts it for <config> (a config without drafter), plus -b 512 -ub N
"""
import copy, importlib.util, json, os, subprocess, sys, time, urllib.request

HOME = os.path.expanduser(os.environ.get("EDGESPECQ_HOME", "~/edgespecq"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(os.environ.get("PORT", 8093))
UBS = tuple(int(x) for x in os.environ.get("UBS", "2,9,512").split(","))  # compared against ub=1
N_PROMPTS = int(os.environ.get("N_PROMPTS", 10))


def query(port, prompts):
    res = []
    for p in prompts:
        body = {"prompt": p, "n_predict": 1, "n_probs": 5, "temperature": 0, "cache_prompt": False}
        r = json.load(urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}), timeout=600))
        res.append([(t["token"], t["logprob"]) for t in r["completion_probabilities"][0]["top_logprobs"]])
    return res


def run_platform(platform, config, prompts, ub):
    spec = importlib.util.spec_from_file_location("run_eval", f"{ROOT}/regression/run_eval.py")
    re_ = importlib.util.module_from_spec(spec); spec.loader.exec_module(re_)
    cfgs = json.load(open(f"{ROOT}/benchmark/configs.json"))
    plat = copy.deepcopy(cfgs["platforms"][platform])
    plat["server_args"] = list(plat.get("server_args", [])) + ["-b", "512", "-ub", str(ub)]
    re_.PORT = int(plat.get("port", re_.PORT))
    home = os.path.expanduser(plat.get("log_home", plat["home"]))
    os.makedirs(f"{home}/logs/batch_invariance", exist_ok=True)
    with re_.Server(plat, cfgs["configs"][config], f"{home}/logs/batch_invariance/{platform}.{config}.ub{ub}.log"):
        return query(re_.PORT, prompts)


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
        return query(PORT, prompts)
    finally:
        srv.terminate()
        srv.wait()
        log.close()


def main():
    if sys.argv[1] == "--platform":
        platform, config, probe = sys.argv[2:5]
        prompts = [json.loads(l)["prompt"] for l in open(probe)][::2][:N_PROMPTS]
        measure = lambda ub: run_platform(platform, config, prompts, ub)
        label = f"platform={platform} config={config}"
    else:
        llama_dir, target, probe = sys.argv[1:4]
        ngl = sys.argv[4] if len(sys.argv) > 4 else "99"
        server = f"{llama_dir}/build/bin/llama-server"
        prompts = [json.loads(l)["prompt"] for l in open(probe)][::2]
        measure = lambda ub: run(server, target, ngl, prompts, ub)
        label = f"target={os.path.basename(target)} ngl={ngl}"
    ref = measure(1)
    print(f"{label}  ({len(prompts)} prompts, reference ub=1)", flush=True)
    for ub in UBS:
        cur = measure(ub)
        maxd = max(abs(a[1] - b[1]) for R, C in zip(ref, cur) for a, b in zip(R, C) if a[0] == b[0])
        flips = sum(R[0][0] != C[0][0] for R, C in zip(ref, cur))
        exact = sum(R == C for R, C in zip(ref, cur))
        print(f"  ub={ub:3d}: bit-identical top-5 on {exact}/{len(ref)} prompts; "
              f"max |dlogprob| {maxd:.5f}; top-1 flips {flips}", flush=True)


if __name__ == "__main__":
    main()
