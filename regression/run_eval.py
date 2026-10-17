#!/usr/bin/env python3
"""Run evaluation configurations (benchmark/configs.json) over an evaluation set (benchmark/build_eval_set.py).

For every (config, repetition) one llama-server is started with that config's target, drafter, runtime libraries
(LD_LIBRARY_PATH) and n_max; every sample is sent turn by turn (multi-turn samples get the model's own previous
answers as history; images go with the first user turn). Each finished sample is appended immediately to
  $HOME/logs/eval/<platform>/<set>/<config>[_<sampling>]/r<rep>.jsonl
and samples already there are skipped, so an interrupted run resumes where it stopped. If the server dies during a
request, the crash is logged to crashes_r<rep>.jsonl, the server is restarted and the sample retried once.
Repetitions rotate the config order so slow drift does not favour one config.

usage: run_eval.py <platform> <set.jsonl> <config,config,...> [--reps 1] [--sampling greedy] [--seed 0]
                   [--max-tokens 512] [--n N]   (N: evenly subsample the set, for smoke runs)
"""
import argparse, base64, hashlib, http.client, json, os, re, subprocess, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(os.environ.get("PORT", 8095))


def expand(p):
    return os.path.expanduser(p)


def model_path(home, tag, kind):
    if kind == "target":
        return f"{home}/models/target/LFM2.5-VL-3B-{tag}.gguf"
    if tag == "F16":
        return f"{home}/models/draft/LFM2.5-VL-3B-DSpark-F16.gguf"
    if "-V" in tag:
        return f"{home}/models/e9/LFM2.5-VL-3B-DSpark-{tag}.gguf"
    return f"{home}/models/quant/LFM2.5-VL-3B-DSpark-{tag}.gguf"


def post(body, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def image_part(path):
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


class Server:
    def __init__(self, plat, cfg, logpath):
        home = expand(plat["home"])
        n_max = cfg.get("n_max") or plat["n_max"]
        self.args = [f"{expand(plat['llama_dir'])}/build/bin/llama-server",
                     "-m", model_path(home, cfg["target"], "target"),
                     "--mmproj", f"{home}/models/target/mmproj-LFM2.5-VL-3B-Q8_0.gguf",
                     "-ngl", "99", "-fa", "on", "-c", "8192", "-np", "1", "--port", str(PORT)]
        if cfg.get("draft"):
            self.args += ["-md", model_path(home, cfg["draft"], "draft"), "-ngld", "99",
                          "--spec-type", "draft-dspark", "--spec-draft-n-max", str(n_max), "--spec-draft-n-min", "0"]
        self.args += plat.get("server_args", [])
        libs = ":".join(f"{home}/{d}" for d in plat["runtimes"][cfg["runtime"]])
        self.env = {**os.environ, "LD_LIBRARY_PATH": libs + ":" + os.environ.get("LD_LIBRARY_PATH", "")}
        self.logpath = logpath
        self.log_pos = 0

    def __enter__(self):
        self.log = open(self.logpath, "w")
        self.proc = subprocess.Popen(self.args, stdout=self.log, stderr=subprocess.STDOUT, env=self.env)
        for _ in range(300):
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200:
                    break
            except Exception:
                if self.proc.poll() is not None:
                    raise SystemExit(f"server exited; see {self.logpath}")
                time.sleep(1)
        else:
            raise SystemExit("server did not become ready")
        post({"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 16, "temperature": 0})  # warmup
        self.new_log()
        return self

    def new_log(self):
        """Server log written since the previous call."""
        self.log.flush()
        with open(self.logpath, errors="replace") as f:
            f.seek(self.log_pos)
            s = f.read()
            self.log_pos = f.tell()
        return s

    def __exit__(self, *exc):
        self.proc.terminate()
        self.proc.wait()
        self.log.close()


def mem_state():
    """MemAvailable / swap in use (MiB) and the cumulative swap-in/out page counters, recorded after every sample so
    that samples measured under memory pressure can be flagged (a counter increase between rows = swapping)."""
    try:
        m = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
        v = {l.split()[0]: int(l.split()[1]) for l in open("/proc/vmstat")}
        return {"mem_avail_mb": m["MemAvailable"] // 1024, "swap_mb": (m["SwapTotal"] - m["SwapFree"]) // 1024,
                "pswpin": v["pswpin"], "pswpout": v["pswpout"]}
    except (OSError, KeyError, ValueError, IndexError):
        return {}


def run_sample(srv, s, sampling, seed, max_tokens):
    messages, turns, texts = [], [], []
    for k, text in enumerate(s["turns"]):
        content = [image_part(p) for p in s["images"]] + [{"type": "text", "text": text}] if k == 0 else text
        messages.append({"role": "user", "content": content})
        body = {"messages": messages, "max_tokens": max_tokens, "seed": seed, **sampling}
        t0 = time.time()
        r = post(body)
        wall = time.time() - t0
        msg = r["choices"][0]["message"]
        answer = msg.get("content") or ""
        t = r.get("timings", {})
        lens = re.findall(r"mean len =\s*([0-9.]+)", srv.new_log())
        turns.append({"n_prompt": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"),
                      "n_pred": t.get("predicted_n"), "pred_ms": t.get("predicted_ms"),
                      "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"),
                      "mean_len": float(lens[-1]) if lens else None, "wall_s": round(wall, 4),
                      "finish": r["choices"][0].get("finish_reason"),
                      "sha": hashlib.sha256(((msg.get("reasoning_content") or "") + answer).encode()).hexdigest()[:12]})
        texts.append(answer)
        messages.append({"role": "assistant", "content": answer})
    return turns, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("platform"); ap.add_argument("set"); ap.add_argument("configs")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--sampling", default="greedy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--file", default=f"{ROOT}/benchmark/configs.json")
    a = ap.parse_args()
    spec = json.load(open(a.file))
    plat = spec["platforms"][a.platform]
    sampling = spec["sampling"][a.sampling]
    names = a.configs.split(",")
    samples = [json.loads(l) for l in open(a.set)]
    if a.n:
        samples = samples[::max(1, len(samples) // a.n)][:a.n]
    set_name = os.path.splitext(os.path.basename(a.set))[0] + (f"_n{a.n}" if a.n else "")
    suffix = "" if a.sampling == "greedy" else f"_{a.sampling}_s{a.seed}"
    base = f"{expand(plat['home'])}/logs/eval/{a.platform}/{set_name}"

    for rep in range(1, a.reps + 1):
        k = (rep - 1) % len(names)
        for name in names[k:] + names[:k]:
            cfg = spec["configs"][name]
            outdir = f"{base}/{name}{suffix}"
            os.makedirs(outdir, exist_ok=True)
            out = f"{outdir}/r{rep}.jsonl"
            done = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()
            todo = [s for s in samples if s["id"] not in done]
            if not todo:
                continue
            json.dump({"config": cfg, "sampling": sampling, "seed": a.seed, "max_tokens": a.max_tokens},
                      open(f"{outdir}/config.json", "w"), indent=1)
            t0 = time.time()
            # a server crash is recorded (crashes_r<rep>.jsonl, with the log of that server kept), the server is
            # restarted and the sample retried once; a sample that crashes twice is skipped
            attempts, starts = {}, 0
            queue = list(todo)
            while queue:
                log = f"{outdir}/server_r{rep}.log" if starts == 0 else f"{outdir}/server_r{rep}_restart{starts}.log"
                starts += 1
                with Server(plat, cfg, log) as srv:
                    json.dump({"argv": srv.args, "LD_LIBRARY_PATH": srv.env["LD_LIBRARY_PATH"]},
                              open(f"{outdir}/server_r{rep}.json", "w"), indent=1)
                    while queue:
                        s = queue[0]
                        try:
                            turns, texts = run_sample(srv, s, sampling, a.seed, a.max_tokens)
                        except (OSError, http.client.HTTPException) as e:  # server died mid-request
                            attempts[s["id"]] = attempts.get(s["id"], 0) + 1
                            with open(f"{outdir}/crashes_r{rep}.jsonl", "a") as f:
                                f.write(json.dumps({"id": s["id"], "attempt": attempts[s["id"]], "error": repr(e),
                                                    "server_log": log, "time": time.time()}) + "\n")
                            print(f"{name}{suffix} r{rep}: server crashed on {s['id']} ({e!r}), restarting",
                                  flush=True)
                            if attempts[s["id"]] >= 2:
                                queue.pop(0)
                            break
                        queue.pop(0)
                        with open(out, "a") as f:
                            f.write(json.dumps({"id": s["id"], "src": s["src"], "turns": turns, **mem_state()}) + "\n")
                        with open(f"{outdir}/outputs_r{rep}.jsonl", "a") as f:
                            f.write(json.dumps({"id": s["id"], "texts": texts}, ensure_ascii=False) + "\n")
            n_pred = 0; pred_ms = 0
            for l in open(out):
                for t in json.loads(l)["turns"]:
                    n_pred += t["n_pred"] or 0; pred_ms += t["pred_ms"] or 0
            print(f"{name}{suffix} r{rep}: {len(todo)} new samples in {time.time() - t0:.0f}s, "
                  f"decode {1000 * n_pred / max(pred_ms, 1e-9):.1f} t/s over {n_pred} tokens", flush=True)


if __name__ == "__main__":
    main()
