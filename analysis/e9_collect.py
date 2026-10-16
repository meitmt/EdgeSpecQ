#!/usr/bin/env python3
"""E9: collect per-round DSpark drafts and how many of their leading tokens the target accepted.

Runs the timing build of llama-speculative-simple (runtime/llama_cpp_patches/e8-timing.patch, which also prints
"E9R n_acc=<k> draft=<id,id,...>" once per freshly drafted round) over probe prompts and writes one JSON line per
round: {"probe", "id", "n_acc", "draft"}. Accepted tokens are draft[:n_acc].

usage: e9_collect.py <llama_dir> <target.gguf> <drafter.gguf> <out.jsonl> [--probes track_t_probe20,ja_probe10] [--n 512]
"""
import argparse, json, os, re, subprocess

HOME = os.path.expanduser(os.environ.get("EDGESPECQ_HOME", "~/edgespecq"))


def run(binary, target, drafter, prompt, n, n_max):
    # -p, not -f: -f drops the trailing newline after "assistant"; --no-escape keeps backslashes in code prompts
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    args = [binary, "-m", target, "-ngl", "99", "-fa", "on", "-c", "4096", "--temp", "0", "-n", str(n),
            "-p", text, "--no-escape", "-md", drafter, "-ngld", "99", "--spec-type", "draft-dspark",
            "--spec-draft-n-max", str(n_max), "--spec-draft-n-min", "0"]
    out = subprocess.run(args, capture_output=True, text=True, errors="replace").stderr
    rounds = [(int(m.group(1)), [int(t) for t in m.group(2).split(",")])
              for m in re.finditer(r"E9R n_acc=(\d+) draft=([0-9,]+)", out)]
    if not rounds:
        raise RuntimeError("no E9R lines:\n" + out[-2000:])
    return rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("llama_dir"); ap.add_argument("target"); ap.add_argument("drafter"); ap.add_argument("out")
    ap.add_argument("--probes", default="track_t_probe20,ja_probe10")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--n-max", type=int, default=8)
    a = ap.parse_args()
    binary = f"{a.llama_dir}/build/bin/llama-speculative-simple"
    with open(a.out, "w") as fo:
        for probe in a.probes.split(","):
            rows = [json.loads(l) for l in open(f"{HOME}/probe/{probe}.jsonl", encoding="utf-8")]
            for r in rows:
                rounds = run(binary, a.target, a.drafter, r["prompt"], a.n, a.n_max)
                for n_acc, draft in rounds:
                    fo.write(json.dumps({"probe": probe, "id": r["id"], "n_acc": n_acc, "draft": draft}) + "\n")
                acc = sum(k for k, _ in rounds)
                print(f"{probe:16s} {r['id']:22s} rounds={len(rounds):4d} accepted/round={acc / len(rounds):.2f}",
                      flush=True)


if __name__ == "__main__":
    main()
