#!/usr/bin/env python3
"""Analyse analysis/e9_lang_rounds.py output: why DSpark accepts less on Japanese.

For every drafted position that verification reached (all accepted positions plus the first rejected one), the
target's token and its probability are known, so per group (en/ja caption, en/ja text) this reports:
  - accepted draft tokens per round, and acceptance per draft position (given the position was reached);
  - how confident the target is (probability of its own greedy token): lower = inherently harder to predict;
  - acceptance at matched target confidence (bins): if Japanese is still lower at the same target confidence,
    the drafter, not the language, is the weaker part;
  - acceptance by the character class of the target token (kanji, kana, ASCII, punctuation, ...), and characters
    per token.

usage: e9_lang_analyze.py <rounds.jsonl> <target.gguf> [--json out.json]
"""
import argparse, json, math, unicodedata
from collections import Counter, defaultdict
import gguf

BINS = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)]


def byte_decoder():
    # GPT-2 byte-level BPE: printable stand-ins for the 256 byte values
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def char_class(text):
    if text is None:
        return "partial"  # an incomplete UTF-8 sequence: part of a multi-byte character
    if text.strip() == "":
        return "newline" if "\n" in text else "space"
    t = text.strip()
    counts = Counter()
    for ch in t:
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
            counts["kanji"] += 1
        elif 0x3040 <= o <= 0x309F:
            counts["hiragana"] += 1
        elif 0x30A0 <= o <= 0x30FF or 0xFF66 <= o <= 0xFF9F:
            counts["katakana"] += 1
        elif 0x3000 <= o <= 0x303F or 0xFF00 <= o <= 0xFF65:
            counts["ja_punct"] += 1
        elif ch.isascii() and ch.isalpha():
            counts["latin"] += 1
        elif ch.isdigit():
            counts["digit"] += 1
        elif unicodedata.category(ch).startswith(("P", "S")):
            counts["punct_symbol"] += 1
        else:
            counts["other"] += 1
    return counts.most_common(1)[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rounds"); ap.add_argument("target"); ap.add_argument("--json")
    ap.add_argument("--target-only", help="e9_lang_rounds.py output of a target-only run (real probabilities)")
    a = ap.parse_args()
    toks = gguf.GGUFReader(a.target).get_field("tokenizer.ggml.tokens").contents()
    bd = byte_decoder()
    def tok_bytes(i):
        return bytes(bd.get(c, 0) for c in toks[i])
    def tok_text(i):
        try:
            return tok_bytes(i).decode("utf-8")
        except UnicodeDecodeError:
            return None

    G = defaultdict(lambda: {"rounds": 0, "acc": 0, "reach": Counter(), "accpos": Counter(), "probs": [],
                             "bins": defaultdict(lambda: [0, 0]), "cls": defaultdict(lambda: [0, 0]),
                             "chars": 0, "ntok": 0, "align_ok": 0, "align_n": 0})
    for line in open(a.rounds, encoding="utf-8"):
        r = json.loads(line)
        g = G[r["group"]]
        lp = r["logprobs"]
        probs = [math.exp(x[1]) for x in lp]
        g["probs"] += probs
        # tokens actually generated: the first one (prompt processing) plus everything the rounds committed (the
        # logprobs list cannot be used: for Japanese it is per character run)
        g["chars"] += len(r["text"]); g["ntok"] += 1 + sum(len(rd["ids"]) for rd in r["rounds"])
        pos = 1  # generation index of the first token committed by a round (index 0 comes from prompt processing)
        for rd in r["rounds"]:
            k, acc, ids, draft = len(rd["draft"]), rd["n_acc"], rd["ids"], rd["draft"]
            g["rounds"] += 1; g["acc"] += acc
            for j in range(min(acc + 1, k)):  # positions verification reached
                gi = pos + j
                ok = j < acc
                g["reach"][j] += 1
                if ok:
                    g["accpos"][j] += 1
                if gi < len(lp):
                    # alignment check: the logprob entry should be the target's token at this position
                    g["align_n"] += 1
                    want = lp[gi][2] if len(lp[gi]) > 2 and lp[gi][2] is not None else None
                    g["align_ok"] += (list(tok_bytes(ids[j])) == want) if want is not None else ((tok_text(ids[j]) or "") == lp[gi][0])
                    p = probs[gi]
                    for lo, hi in BINS:
                        if lo <= p < hi:
                            g["bins"][(lo, hi)][0] += ok; g["bins"][(lo, hi)][1] += 1
                c = char_class(tok_text(ids[j]))
                g["cls"][c][0] += ok; g["cls"][c][1] += 1
            pos += len(ids)

    out = {}
    order = [x for x in ("en_caption", "ja_caption", "en_text", "ja_text") if x in G]
    print("accepted draft tokens per round (excluding the bonus token), characters per token")
    for name in order:
        g = G[name]
        print(f"  {name:10s} rounds {g['rounds']:5d}  accepted/round {g['acc'] / g['rounds']:.2f}  "
              f"chars/token {g['chars'] / g['ntok']:.2f}")
    print("\nacceptance per draft position, given the position was reached")
    print("  " + " ".join(f"{'pos' + str(j):>6s}" for j in range(8)) + "   group")
    for name in order:
        g = G[name]
        print("  " + " ".join(f"{g['accpos'][j] / g['reach'][j]:6.2f}" if g["reach"][j] else f"{'-':>6s}" for j in range(8))
              + f"   {name}")
    print("\nacceptance by character class of the target token: rate (share of reached positions)")
    classes = ["latin", "space", "newline", "punct_symbol", "digit", "kanji", "hiragana", "katakana", "ja_punct",
               "partial", "other"]
    print(f"  {'group':10s} " + " ".join(f"{c:>15s}" for c in classes))
    for name in order:
        g = G[name]
        tot = sum(v[1] for v in g["cls"].values())
        print(f"  {name:10s} " + " ".join(
            f"{g['cls'][c][0] / g['cls'][c][1]:5.2f} ({100 * g['cls'][c][1] / tot:4.1f}%)" if g["cls"][c][1] else f"{'-':>15s}"
            for c in classes))
    conf = {}
    if a.target_only:
        # the target's own confidence, from a target-only run of the same prompts (logprobs entries are per
        # complete character run, see e9_lang_rounds.py), overall and per character class
        print("\ntarget confidence from the target-only run: p of its greedy token (entries are character runs)")
        print(f"  {'group':10s} {'mean p':>7s} {'p<0.5':>6s} {'p<0.8':>6s} | mean p by class: "
              + " ".join(f"{c:>9s}" for c in ("latin", "kanji", "hiragana", "katakana", "ja_punct", "punct_symbol")))
        T = defaultdict(lambda: {"p": [], "cls": defaultdict(list)})
        for line in open(a.target_only, encoding="utf-8"):
            r = json.loads(line)
            for e in r["logprobs"]:
                p = math.exp(e[1]); t = T[r["group"]]
                t["p"].append(p)
                t["cls"][char_class(bytes(e[2]).decode("utf-8", "replace") if e[2] else e[0])].append(p)
        for name in order:
            t = T[name]; p = t["p"]
            cm = {c: sum(v) / len(v) for c, v in t["cls"].items() if len(v) >= 20}
            conf[name] = {"mean_p": sum(p) / len(p), "share_p_lt_0.5": sum(x < 0.5 for x in p) / len(p),
                          "share_p_lt_0.8": sum(x < 0.8 for x in p) / len(p), "mean_p_by_class": cm}
            print(f"  {name:10s} {conf[name]['mean_p']:7.3f} {conf[name]['share_p_lt_0.5']:6.3f} "
                  f"{conf[name]['share_p_lt_0.8']:6.3f} |                  " + " ".join(
                      f"{cm[c]:9.3f}" if c in cm else f"{'-':>9s}"
                      for c in ("latin", "kanji", "hiragana", "katakana", "ja_punct", "punct_symbol")))
    for name in order:
        g = G[name]
        out[name] = {"rounds": g["rounds"], "accepted_per_round": g["acc"] / g["rounds"],
                     "accept_by_position": {j: g["accpos"][j] / g["reach"][j] for j in range(8) if g["reach"][j]},
                     "accept_by_class": {c: v for c, v in g["cls"].items()},
                     "chars_per_token": g["chars"] / g["ntok"], "target_confidence": conf.get(name)}
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
