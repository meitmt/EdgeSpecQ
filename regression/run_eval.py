#!/usr/bin/env python3
"""Run evaluation configurations (benchmark/configs.json) over an evaluation set (benchmark/build_eval_set.py).

For every (config, repetition) one llama-server is started with that config's target, drafter, runtime libraries
(LD_LIBRARY_PATH) and n_max; every sample is sent turn by turn (multi-turn samples get the model's own previous
answers as history; images go with the first user turn). Each finished sample is appended immediately to
  $HOME/logs/eval/<platform>/<set>/<config>[_<sampling>]/r<rep>.jsonl
and samples already there are skipped, so an interrupted run resumes where it stopped. If the server dies during a
request, the crash is logged to crashes_r<rep>.jsonl, the server is restarted and the sample retried once.
Platforms with an "adb" entry (Android) run only llama-server on the device; this script, the images and the
results stay on the host, talking to the server through `adb forward`.
Repetitions rotate the config order so slow drift does not favour one config.

usage: run_eval.py <platform> <set.jsonl> <config,config,...> [--reps 1] [--sampling greedy] [--seed 0]
                   [--max-tokens 512] [--n N] [--tag T]   (N: evenly subsample the set, for smoke runs)
"""
import argparse, base64, hashlib, http.client, json, os, re, shlex, socket, subprocess, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# server port; a platform's "port" overrides it (with WSL mirrored networking, adb forwards and local servers share
# localhost, so concurrent runs on different platforms need different ports)
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


def adb_cmd(plat):
    """adb executable ($ADB, else the platform's "exe", else adb on PATH) plus the device selection: the platform's
    "serial" if set, otherwise adb's own default ($ANDROID_SERIAL, or the only connected device)."""
    a = plat["adb"]
    exe = os.environ.get("ADB") or a.get("exe") or "adb"
    return [exe, *(["-s", a["serial"]] if a.get("serial") else [])]


def adb(plat, *args, **kw):
    """Run an adb command for a remote (Android) platform; returns stdout."""
    return subprocess.run([*adb_cmd(plat), *args], capture_output=True, text=True, **kw).stdout


class Server:
    """llama-server for one config. Local platforms run it as a child process; a platform with an "adb" entry runs
    it on the device (`adb shell`, port forwarded with `adb forward`) and streams its log back over adb."""

    def __init__(self, plat, cfg, logpath):
        self.plat = plat
        self.remote = "adb" in plat
        home = plat["home"] if self.remote else expand(plat["home"])
        n_max = cfg.get("n_max") or plat["n_max"]
        ngl = str(plat.get("ngl", 99))
        if "bin_dirs" in plat:
            binary = f"{home}/{plat['bin_dirs'][cfg['runtime']]}/llama-server"
        else:
            binary = f"{expand(plat['llama_dir'])}/build/bin/llama-server"
        self.args = [binary,
                     "-m", model_path(home, cfg["target"], "target"),
                     "--mmproj", f"{home}/models/target/mmproj-LFM2.5-VL-3B-Q8_0.gguf",
                     "-ngl", ngl, "-fa", "on", "-c", "8192", "-np", "1", "--port", str(PORT)]
        if cfg.get("draft"):
            self.args += ["-md", model_path(home, cfg["draft"], "draft"), "-ngld", ngl,
                          "--spec-type", "draft-dspark", "--spec-draft-n-max", str(n_max), "--spec-draft-n-min", "0"]
        sargs = list(plat.get("server_args", []))
        if cfg.get("device_draft"):  # per-config drafter device, e.g. target on HTP0 and drafter on GPUOpenCL / CPU
            if "--device-draft" in sargs:
                i = sargs.index("--device-draft"); del sargs[i:i + 2]
            if cfg["device_draft"] == "CPU":  # no offload for the drafter
                sargs += ["--device-draft", "none"]
                self.args[self.args.index("-ngld") + 1] = "0"
            else:
                sargs += ["--device-draft", cfg["device_draft"]]
        self.args += sargs
        libs = ":".join(f"{home}/{d}" for d in plat["runtimes"][cfg["runtime"]])
        # extra environment, e.g. ADSP_LIBRARY_PATH for the Hexagon NPU ("{libs}" = the runtime's library dirs)
        extra = {k: v.replace("{libs}", libs) for k, v in plat.get("env", {}).items()}
        if self.remote:
            self.env = {"LD_LIBRARY_PATH": libs, **extra}
            envs = " ".join(f"{k}={shlex.quote(v)}" for k, v in self.env.items())
            self.cmd = [*adb_cmd(plat), "shell",
                        f"cd {home} && {envs} exec " + " ".join(shlex.quote(x) for x in self.args)]
        else:
            self.env = {**os.environ, **extra, "LD_LIBRARY_PATH": libs + ":" + os.environ.get("LD_LIBRARY_PATH", "")}
            self.cmd = self.args
        self.require = plat.get("require_device")  # e.g. "HTP0": refuse to measure if the target is not placed there
        self.logpath = logpath
        self.log_pos = 0

    def probe_placement(self):
        """Load the same config once with a verbose log (-lv 4: buffer layout, graph splits, unsupported ops) on a
        side port, then stop it; the measured server runs at the normal log level."""
        port = str(PORT + 1)
        args = [port if a == str(PORT) else a for a in self.args] + ["-lv", "4"]
        log = self.logpath.replace(".log", "_probe.log")
        envs = " ".join(f"{k}={shlex.quote(v)}" for k, v in self.env.items())
        home = self.plat["home"]
        with open(log, "w") as f:
            p = subprocess.Popen([*adb_cmd(self.plat), "shell", f"cd {home} && {envs} exec " + " ".join(shlex.quote(x) for x in args)],
                                 stdout=f, stderr=subprocess.STDOUT)
            for _ in range(180):
                time.sleep(1)
                if "listening on" in open(log, errors="replace").read() or p.poll() is not None:
                    break
            adb(self.plat, "shell", f"pkill -f '^{home}/.*/bin/llama-server .*--port {port}'")
            p.wait(timeout=60)
        text = open(log, errors="replace").read()
        pl = placement(text)
        pl["clip_backend"] = (re.findall(r"CLIP using (\S+) backend", text) or [None])[0]
        pl["unsupported_ops"] = sorted(set(re.findall(r"warmup:\s+(\w+): type = ", text)))
        return pl

    def __enter__(self):
        self.log = open(self.logpath, "w")
        if self.require:
            self.placement = self.probe_placement()
            if not any(d.startswith(self.require) for d in self.placement["model_buffers_mib"]):
                raise SystemExit(f"target not placed on {self.require}: {self.placement}; see {self.logpath}")
        if self.remote:
            # leftovers of an interrupted run: only servers under our own directory (other experiments on the device
            # may run their own llama-server)
            adb(self.plat, "shell", f"pkill -f '^{self.plat['home']}/.*/bin/llama-server'")
            adb(self.plat, "forward", "--remove", f"tcp:{PORT}")
        # a server already answering on this port (e.g. an orphan of an earlier run) would silently receive every
        # request of this config, so refuse to start
        if port_in_use(PORT):
            raise SystemExit(f"port {PORT} is already in use by another process; not starting {self.logpath}")
        if self.remote:
            adb(self.plat, "forward", f"tcp:{PORT}", f"tcp:{PORT}")
            self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT)
        else:
            self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT, env=self.env)
        try:
            for _ in range(600):
                try:
                    if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200:
                        break
                except Exception:
                    if self.proc.poll() is not None:
                        raise SystemExit(f"server exited; see {self.logpath}")
                    time.sleep(1)
            else:
                raise SystemExit(f"server did not become ready; see {self.logpath}")
            if self.proc.poll() is not None:  # the answer came from some other process
                raise SystemExit(f"our server exited but port {PORT} answered; see {self.logpath}")
            post({"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 16, "temperature": 0})  # warmup
        except BaseException:
            self.__exit__()  # never leave our server running when we give up on it
            raise
        log = self.new_log()
        if not self.require:
            self.placement = placement(log)
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
        if self.remote:
            adb(self.plat, "shell", f"pkill -f '^{self.plat['home']}/.*/bin/llama-server'")
            adb(self.plat, "forward", "--remove", f"tcp:{PORT}")
        self.proc.terminate()
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill(); self.proc.wait()
        self.log.close()


def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def placement(log):
    """Where the weights went and how fragmented the graphs are, from the server's start-up log: model buffer size
    per device (target and drafter loads both listed, in load order) and the scheduler's graph split counts."""
    bufs = {}
    for dev, mib in re.findall(r"load_tensors:\s+(\S+) model buffer size\s*=\s*([0-9.]+) MiB", log):
        bufs[dev] = round(bufs.get(dev, 0) + float(mib), 1)
    splits = [int(s) for s in re.findall(r"splits = (\d+)", log)]
    return {"model_buffers_mib": bufs, "graph_splits": splits}


def mem_state(plat):
    """MemAvailable / swap in use (MiB) and the cumulative swap-in/out page counters, recorded after every sample so
    that samples measured under memory pressure can be flagged (a counter increase between rows = swapping).
    On an adb platform they are read from the device, plus the battery temperature (thermal throttling)."""
    try:
        if "adb" in plat:
            out = adb(plat, "shell", "cat /proc/meminfo /proc/vmstat; dumpsys battery | grep temperature", timeout=30)
            lines = out.splitlines()
        else:
            lines = open("/proc/meminfo").read().splitlines() + open("/proc/vmstat").read().splitlines()
        kv = {}
        for l in lines:
            parts = l.replace(":", " ").split()
            if len(parts) >= 2 and parts[1].lstrip("-").isdigit():
                kv.setdefault(parts[0], int(parts[1]))
        r = {"mem_avail_mb": kv["MemAvailable"] // 1024, "swap_mb": (kv["SwapTotal"] - kv["SwapFree"]) // 1024,
             "pswpin": kv["pswpin"], "pswpout": kv["pswpout"]}
        if "temperature" in kv:
            r["battery_temp_c"] = kv["temperature"] / 10
        return r
    except (OSError, KeyError, ValueError, IndexError, subprocess.SubprocessError):
        return {}


def run_sample(srv, s, sampling, seed, max_tokens, logprobs=False):
    messages, turns, texts, toks = [], [], [], []
    for k, text in enumerate(s["turns"]):
        content = [image_part(p) for p in s["images"]] + [{"type": "text", "text": text}] if k == 0 else text
        messages.append({"role": "user", "content": content})
        body = {"messages": messages, "max_tokens": max_tokens, "seed": seed, **sampling}
        if logprobs:  # top-3 per generated token, for the greedy-equivalence judgment (analysis/judge_greedy.py)
            body.update({"logprobs": True, "top_logprobs": 3})
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
        if logprobs:
            lp = (r["choices"][0].get("logprobs") or {}).get("content") or []
            toks.append([[x["token"], x["logprob"], [[y["token"], y["logprob"]] for y in x.get("top_logprobs", [])]]
                         for x in lp])
        messages.append({"role": "assistant", "content": answer})
    return turns, texts, toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("platform"); ap.add_argument("set"); ap.add_argument("configs")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--sampling", default="greedy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--file", default=f"{ROOT}/benchmark/configs.json")
    ap.add_argument("--logprobs", action="store_true",
                    help="also record per-token top-3 logprobs (tokens_r<rep>.jsonl) into <config>_lp/, for the S3 "
                         "greedy-equivalence judgment; timings from such runs are not used")
    ap.add_argument("--tag", help="write to <set>_<tag>/ instead of <set>/, e.g. a reproduction kept apart from the "
                                  "published runs")
    a = ap.parse_args()
    spec = json.load(open(a.file))
    plat = spec["platforms"][a.platform]
    sampling = spec["sampling"][a.sampling]
    names = a.configs.split(",")
    samples = [json.loads(l) for l in open(a.set)]
    if a.n:
        samples = samples[::max(1, len(samples) // a.n)][:a.n]
    global PORT
    PORT = int(plat.get("port", PORT))
    set_name = os.path.splitext(os.path.basename(a.set))[0] + (f"_n{a.n}" if a.n else "") + \
        (f"_{a.tag}" if a.tag else "")
    suffix = ("" if a.sampling == "greedy" else f"_{a.sampling}_s{a.seed}") + ("_lp" if a.logprobs else "")
    # results stay on the machine running this script (for adb platforms "home" is a path on the device)
    base = f"{expand(plat.get('log_home', plat['home']))}/logs/eval/{a.platform}/{set_name}"

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
            cool = plat.get("cooldown")  # phones: start every config from the same thermal state
            if cool:
                t_c = time.time()
                while time.time() - t_c < cool.get("timeout_s", 900):
                    temp = mem_state(plat).get("battery_temp_c")
                    if temp is None or temp <= cool["max_battery_temp_c"]:
                        break
                    time.sleep(15)
                print(f"{name}{suffix} r{rep}: start at battery {temp} C after {time.time() - t_c:.0f}s cooldown",
                      flush=True)
            t0 = time.time()
            # a server crash is recorded (crashes_r<rep>.jsonl, with the log of that server kept), the server is
            # restarted and the sample retried once; a sample that crashes twice is skipped
            attempts, starts = {}, 0
            queue = list(todo)
            while queue:
                log = f"{outdir}/server_r{rep}.log" if starts == 0 else f"{outdir}/server_r{rep}_restart{starts}.log"
                starts += 1
                with Server(plat, cfg, log) as srv:
                    json.dump({"argv": srv.args, "LD_LIBRARY_PATH": srv.env["LD_LIBRARY_PATH"],
                               "placement": srv.placement},
                              open(f"{outdir}/server_r{rep}.json", "w"), indent=1)
                    while queue:
                        s = queue[0]
                        try:
                            turns, texts, toks = run_sample(srv, s, sampling, a.seed, a.max_tokens, a.logprobs)
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
                        st = mem_state(plat)
                        with open(out, "a") as f:
                            f.write(json.dumps({"id": s["id"], "src": s["src"], "turns": turns, **st}) + "\n")
                        with open(f"{outdir}/outputs_r{rep}.jsonl", "a") as f:
                            f.write(json.dumps({"id": s["id"], "texts": texts}, ensure_ascii=False) + "\n")
                        if a.logprobs:
                            with open(f"{outdir}/tokens_r{rep}.jsonl", "a") as f:
                                f.write(json.dumps({"id": s["id"], "turns": toks}, ensure_ascii=False) + "\n")
                        # thermal guard (phones): pause between samples while the device is too hot
                        guard = (cool or {}).get("pause_above_c")
                        if guard and (st.get("battery_temp_c") or 0) > guard:
                            t_p = time.time()
                            while (mem_state(plat).get("battery_temp_c") or 0) > cool["resume_below_c"]:
                                time.sleep(20)
                            print(f"{name}{suffix} r{rep}: paused {time.time() - t_p:.0f}s at battery "
                                  f"{st['battery_temp_c']} C", flush=True)
            n_pred = 0; pred_ms = 0
            for l in open(out):
                for t in json.loads(l)["turns"]:
                    n_pred += t["n_pred"] or 0; pred_ms += t["pred_ms"] or 0
            print(f"{name}{suffix} r{rep}: {len(todo)} new samples in {time.time() - t0:.0f}s, "
                  f"decode {1000 * n_pred / max(pred_ms, 1e-9):.1f} t/s over {n_pred} tokens", flush=True)


if __name__ == "__main__":
    main()
