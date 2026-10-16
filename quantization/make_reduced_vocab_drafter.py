#!/usr/bin/env python3
"""E9: export a reduced-vocabulary DSpark drafter from an existing DSpark GGUF, without retraining.

The draft vocabulary is the N most frequent target tokens on a reference corpus mix (results/e9/freq_<src>.npy, from
analysis/e9_corpus_freq.py), always including control and user-defined tokens (never the [PAD] fillers). The new GGUF
  - slices markov_w2 to the N kept rows,
  - adds its own `output` = the N kept rows of the target's (tied) token_embd,
  - adds `d2t` (I64, draft row -> target token id), which llama.cpp's dflash loader already understands.
Both weights are quantized along the hidden dimension only, so row slicing is lossless: for every kept token the
drafter's logits are bit-identical to the full-vocab drafter.

usage: make_reduced_vocab_drafter.py <drafter.gguf> <target.gguf> <N> <out.gguf> [--mix en_chat=1,en_wiki=1,...]
"""
import argparse, os
import numpy as np
import gguf

HOME = os.path.expanduser(os.environ.get("EDGESPECQ_HOME", "~/edgespecq"))


def draft_vocab(target_reader, n, mix):
    types = np.array(target_reader.get_field("tokenizer.ggml.token_type").contents())
    score = 0
    for name, w in mix.items():
        c = np.load(f"{HOME}/results/e9/freq_{name}.npy").astype(np.float64)
        score = score + w * c / c.sum()
    always = np.flatnonzero((types == gguf.TokenType.CONTROL) | (types == gguf.TokenType.USER_DEFINED))
    score = np.asarray(score).copy()
    score[types == gguf.TokenType.UNUSED] = -np.inf  # [PAD...] filler rows
    score[always] = np.inf
    order = np.argsort(-score, kind="stable")  # ties (unseen tokens) by id
    return np.sort(order[:n]).astype(np.int64), len(always)


def tensor(reader, name):
    for t in reader.tensors:
        if t.name == name:
            return t
    raise KeyError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("drafter"); ap.add_argument("target"); ap.add_argument("n", type=int); ap.add_argument("out")
    ap.add_argument("--mix", default="en_chat=1,en_wiki=1,code=1,math=1,ja_wiki=1,ja_chat=1")
    a = ap.parse_args()
    mix = {k: float(v) for k, v in (kv.split("=") for kv in a.mix.split(","))}

    dr = gguf.GGUFReader(a.drafter)
    tr = gguf.GGUFReader(a.target)
    ids, n_special = draft_vocab(tr, a.n, mix)

    emb = tensor(tr, "output.weight") if any(t.name == "output.weight" for t in tr.tensors) else tensor(tr, "token_embd.weight")
    w2 = tensor(dr, "markov_w2.weight")
    new = {"markov_w2.weight": (w2.data[ids], w2.tensor_type),
           "output.weight":    (emb.data[ids], emb.tensor_type),
           "d2t":              (ids, None)}
    assert not any(t.name in ("output.weight", "d2t") for t in dr.tensors), "drafter already has its own output/d2t"

    arch = dr.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    w = gguf.GGUFWriter(a.out, arch=arch, endianess=dr.endianess)
    for f in dr.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        vt = f.types[0]
        w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
    w.add_uint32("edgespecq.draft_vocab_size", a.n)
    w.add_string("edgespecq.draft_vocab_mix", a.mix)

    tensors = [(t.name, *new.get(t.name, (t.data, t.tensor_type))) for t in dr.tensors]
    tensors += [(k, *v) for k, v in new.items() if k != "markov_w2.weight"]
    for name, data, ttype in tensors:
        w.add_tensor_info(name, data.shape, data.dtype, data.nbytes, ttype)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for _, data, _ in tensors:
        w.write_tensor_data(np.ascontiguousarray(data), tensor_endianess=dr.endianess)
    w.close()
    print(f"{os.path.basename(a.out)}: N={a.n} (always-kept control/user-defined tokens: {n_special}), "
          f"markov_w2 {w2.data.nbytes / 2**20:.1f} -> {new['markov_w2.weight'][0].nbytes / 2**20:.1f} MiB, "
          f"own output head {new['output.weight'][0].nbytes / 2**20:.1f} MiB ({emb.tensor_type.name}), "
          f"file {os.path.getsize(a.out) / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
