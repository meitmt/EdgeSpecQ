#!/usr/bin/env python3
"""Acceptance / greedy-equivalence probe.

Starts llama-server from a given llama.cpp checkout, runs a JSONL prompt set (text or image+text) with greedy
decoding, and records per-prompt draft acceptance, mean accepted length (parsed from the server log), decode
throughput, full outputs (reasoning + content) and, optionally, per-token top-k logprobs.

usage:
  probe_accept.py <llama_dir> <tag> <target.gguf> <draft.gguf|none> <probe.jsonl> [extra llama-server args...]

environment:
  EDGESPECQ_HOME  work dir (default ~/edgespecq); results go to $EDGESPECQ_HOME/logs/probe/<tag>/
  MAX_TOKENS      generation cap per prompt (default 384)
  N_PROMPTS       evenly subsample the probe to N prompts (default: all)
  NGL             layers on GPU for target and draft (default 99; 0 = CPU backend)
  N_MAX           DSpark --spec-draft-n-max (default 8)
  LOGPROBS=1      also save per-token top-3 logprobs (tokens.jsonl), needed by analysis/divergence.py
  PORT            server port (default 8091)
"""
import base64, hashlib, json, os, re, subprocess, sys, time, urllib.request

HOME = os.path.expanduser(os.environ.get("EDGESPECQ_HOME", "~/edgespecq"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 384))
N_PROMPTS = int(os.environ.get("N_PROMPTS", 0))
NGL = os.environ.get("NGL", "99")
N_MAX = os.environ.get("N_MAX", "8")
LOGPROBS = os.environ.get("LOGPROBS") == "1"
PORT = int(os.environ.get("PORT", 8091))


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def wait_ready(srv, outdir):
    for _ in range(180):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200:
                return
        except Exception:
            if srv.poll() is not None:
                raise SystemExit(f"server exited; see {outdir}/server.log")
            time.sleep(1)
    raise SystemExit("server did not become ready")


def user_content(p):
    if not p.get("image"):
        return p["prompt"]
    b64 = base64.b64encode(open(p["image"], "rb").read()).decode()
    return [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": p["prompt"]}]


def main():
    llama_dir, tag, target, draft, probe = sys.argv[1:6]
    extra = sys.argv[6:]
    outdir = f"{HOME}/logs/probe/{tag}"
    os.makedirs(outdir, exist_ok=True)
    for f in ("outputs.jsonl", "tokens.jsonl"):
        if os.path.exists(f"{outdir}/{f}"):
            os.remove(f"{outdir}/{f}")

    args = [f"{llama_dir}/build/bin/llama-server", "-m", target, "-ngl", NGL, "-fa", "on", "-c", "4096",
            "-np", "1", "--temp", "0", "--port", str(PORT)]
    if draft != "none":
        args += ["-md", draft, "-ngld", NGL, "--spec-type", "draft-dspark",
                 "--spec-draft-n-max", N_MAX, "--spec-draft-n-min", "0"]
    args += extra
    json.dump({"argv": args, "max_tokens": MAX_TOKENS, "probe": probe}, open(f"{outdir}/config.json", "w"), indent=1)

    prompts = [json.loads(l) for l in open(probe)]
    if N_PROMPTS:
        prompts = prompts[::max(1, len(prompts) // N_PROMPTS)][:N_PROMPTS]

    log = open(f"{outdir}/server.log", "w")
    srv = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
    rows = []
    try:
        wait_ready(srv, outdir)
        post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello"}],
                                      "max_tokens": 16, "temperature": 0})  # warmup
        for p in prompts:
            body = {"messages": [{"role": "user", "content": user_content(p)}],
                    "max_tokens": MAX_TOKENS, "temperature": 0, "seed": 0}
            if LOGPROBS:
                body.update({"logprobs": True, "top_logprobs": 3})
            r = post("/v1/chat/completions", body)
            t = r.get("timings", {})
            msg = r["choices"][0]["message"]
            # reasoning models return thinking separately; compare the full generation
            text = (msg.get("reasoning_content") or "") + "\n<<content>>\n" + (msg.get("content") or "")
            rows.append({"src": p["src"], "id": p["id"], "n_pred": t.get("predicted_n"),
                         "tps": t.get("predicted_per_second"), "draft_n": t.get("draft_n"),
                         "draft_acc": t.get("draft_n_accepted"),
                         "sha": hashlib.sha256(text.encode()).hexdigest()[:12]})
            with open(f"{outdir}/outputs.jsonl", "a") as f:
                f.write(json.dumps({"id": p["id"], "text": text}, ensure_ascii=False) + "\n")
            if LOGPROBS:
                lp = (r["choices"][0].get("logprobs") or {}).get("content") or []
                with open(f"{outdir}/tokens.jsonl", "a") as f:
                    f.write(json.dumps({"id": p["id"], "tok": [
                        [x["token"], x["logprob"], [[y["token"], y["logprob"]] for y in x.get("top_logprobs", [])]]
                        for x in lp]}, ensure_ascii=False) + "\n")
    finally:
        srv.terminate()
        srv.wait()
        log.close()

    # "mean len = X" is printed once per task; the first one belongs to the warmup request
    lens = [float(m) for m in re.findall(r"mean len =\s*([0-9.]+)", open(f"{outdir}/server.log").read())]
    if len(lens) == len(rows) + 1:
        lens = lens[1:]
    for i, r in enumerate(rows):
        r["mean_len"] = lens[i] if i < len(lens) else None
    json.dump(rows, open(f"{outdir}/rows.json", "w"), indent=1)

    tot_pred = sum(r["n_pred"] or 0 for r in rows)
    tot_s = sum((r["n_pred"] or 0) / r["tps"] for r in rows if r["tps"])
    dn = sum(r["draft_n"] or 0 for r in rows)
    da = sum(r["draft_acc"] or 0 for r in rows)
    print(f"== {tag}: {len(rows)} prompts, {tot_pred} tokens, decode {tot_pred / tot_s:.1f} tok/s")
    if dn:
        print(f"   draft acceptance {da}/{dn} = {da / dn:.3f}")
    w = [(r["mean_len"], r["n_pred"]) for r in rows if r["mean_len"]]
    if w:
        print(f"   mean len: per-prompt avg {sum(x for x, _ in w) / len(w):.2f}, "
              f"token-weighted {sum(x * n for x, n in w) / sum(n for _, n in w):.2f}")
    by = {}
    for r in rows:
        by.setdefault(r["src"], []).append(r)
    for s, rs in by.items():
        ml = [r["mean_len"] for r in rs if r["mean_len"]]
        tps = [r["tps"] for r in rs if r["tps"]]
        print(f"   {s:9s} mean_len {sum(ml) / len(ml) if ml else float('nan'):.2f}  tps {sum(tps) / len(tps):.0f}")


if __name__ == "__main__":
    main()
