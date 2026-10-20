#!/usr/bin/env python3
"""Figures for the tech note and README, from the committed result files only (results/summary/).

  fig1_round_breakdown   where a speculation round's time goes (S4): RTX 5070 / Jetson / phone NPU, F16 vs Q8_0
  fig2_ablation          dev120 ablation ladder A0 -> A7, speedup over unpatched target-only, Jetson vs RTX 5070
  fig3_platforms         dev120 speedup of the official F16 drafter vs the best configuration, per platform
  fig4_japanese          English (dev120) vs Japanese (ja75) speedup per drafter configuration
  fig5_ja_confidence     Japanese captioning: per-character-class acceptance vs the target's own confidence

Each figure is written as SVG and PNG (2x) in a light and a dark variant, plus the plotted numbers as CSV
(results/figures/data/), which is the table view of every chart.

usage: make_figures.py [--out results/figures]
"""
import argparse, csv, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUM = f"{ROOT}/results/summary"

# reference palette (dataviz skill, references/palette.md): categorical slots 1-4 in fixed order, chart chrome
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", series=["#3987e5", "#d95926", "#199e70", "#c98500"]),
}
DPI = 100            # 1 css px = 1/100 inch; PNGs are saved at 2x
BAR_PX = 22          # bar thickness cap (<= 24 px)
RADIUS_PX = 4        # rounded data-end
GAP_PX = 2           # surface gap between touching marks


def read_csv(name):
    with open(f"{SUM}/{name}") as f:
        return {r["config"]: r for r in csv.DictReader(f)}


def write_rows(out, name, header, rows):
    os.makedirs(f"{out}/data", exist_ok=True)
    with open(f"{out}/data/{name}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


# ---------------------------------------------------------------- drawing helpers

def px_to_data(ax, px):
    """(dx, dy) in data units for `px` css pixels at the current layout."""
    bb = ax.get_window_extent()
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    return px * (x1 - x0) / bb.width * (DPI / ax.figure.dpi), px * (y1 - y0) / bb.height * (DPI / ax.figure.dpi)


def rounded_rect(ax, x0, x1, y0, y1, color, end):
    """Rectangle with the data end rounded: end='top' (vertical bar) or 'right' (horizontal bar) or None."""
    rx, ry = px_to_data(ax, RADIUS_PX)
    if end == "top":
        rx = min(rx, (x1 - x0) / 2); ry = min(ry, (y1 - y0))
        v = [(x0, y0), (x0, y1 - ry), (x0, y1), (x0 + rx, y1), (x1 - rx, y1), (x1, y1), (x1, y1 - ry), (x1, y0), (x0, y0)]
        c = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    elif end == "right":
        ry = min(ry, (y1 - y0) / 2); rx = min(rx, (x1 - x0))
        v = [(x0, y0), (x1 - rx, y0), (x1, y0), (x1, y0 + ry), (x1, y1 - ry), (x1, y1), (x1 - rx, y1), (x0, y1), (x0, y0)]
        c = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    else:
        v = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        c = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(v, c), facecolor=color, edgecolor="none", zorder=3))


def style_axes(ax, t, grid_axis="y"):
    ax.set_facecolor(t["surface"])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    base = "bottom" if grid_axis == "y" else "left"
    ax.spines[base].set_visible(True); ax.spines[base].set_color(t["axis"]); ax.spines[base].set_linewidth(1)
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=0.75, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=0, labelsize=9)


def new_fig(t, w, h, title, subtitle):
    fig = plt.figure(figsize=(w, h), dpi=DPI, facecolor=t["surface"])
    fig.text(0.02, 0.97, title, ha="left", va="top", fontsize=13, fontweight="bold", color=t["ink"])
    fig.text(0.02, 0.97 - 0.36 / h, subtitle, ha="left", va="top", fontsize=9.5, color=t["ink2"])
    return fig


def legend(fig, t, labels, colors, y):
    handles = [Patch(facecolor=c, edgecolor="none") for c in colors]
    lg = fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.015, y), ncol=len(labels), frameon=False,
                    fontsize=9, handlelength=1.0, handleheight=1.0, columnspacing=1.4, borderaxespad=0)
    for txt in lg.get_texts():
        txt.set_color(t["ink2"])


def save(fig, out, name, theme):
    suffix = "" if theme == "light" else "_dark"
    for ext in ("svg", "png"):
        fig.savefig(f"{out}/{name}{suffix}.{ext}", dpi=2 * DPI, facecolor=fig.get_facecolor())
    plt.close(fig)


def grouped_columns(ax, t, groups, series, values, labels_at=None, fmt="{:.2f}×", ymax=None):
    """Vertical grouped columns from a single baseline at 0. values[s][g]; labels_at: set of (s, g) to label."""
    n_s = len(series)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    top = max(max(v) for v in values)
    ax.set_ylim(0, ymax or top * 1.18)
    ax.figure.canvas.draw()
    bw, _ = px_to_data(ax, BAR_PX)
    gap, yoff = px_to_data(ax, GAP_PX)[0], px_to_data(ax, 4)[1]
    for s in range(n_s):
        for g, v in enumerate(values[s]):
            xc = g + (s - (n_s - 1) / 2) * (bw + gap)
            rounded_rect(ax, xc - bw / 2, xc + bw / 2, 0, v, t["series"][s], "top")
            if labels_at is None or (s, g) in labels_at:
                # two series: labels grow outward from the pair so neighbours of similar height never collide
                if n_s == 2:
                    lx, ha = (xc + bw / 2, "right") if s == 0 else (xc - bw / 2, "left")
                else:
                    lx, ha = xc, "center"
                ax.text(lx, v + yoff, fmt.format(v), ha=ha, va="bottom", fontsize=8.5, color=t["ink"], zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)


# ---------------------------------------------------------------- figures

def fig1(out, t, theme):
    e8_5070 = json.load(open(f"{SUM}/e8_breakdown/e8_5070_n8.json"))
    e8_j = json.load(open(f"{SUM}/e8_breakdown/e8_jetson_maxn_n8.json"))
    c4_j = json.load(open(f"{SUM}/e9c4_jetson_maxn_breakdown_n6.json"))["P3-V32k-i32"]
    htp = json.load(open(f"{SUM}/phone_breakdown/p3v32k_htp.json"))["P3-V32k-i32"]
    rows = [
        ("RTX 5070 · F16 drafter", e8_5070["F16"]),
        ("RTX 5070 · Q8_0 drafter", e8_5070["Q8_0"]),
        ("Jetson · F16 drafter", e8_j["F16"]),
        ("Jetson · Q8_0 drafter", e8_j["Q8_0"]),
        ("Jetson · P3-V32k, fixes + C4, n_max 6", c4_j),
        ("Phone NPU · P3-V32k + C4", htp),
    ]
    segs = ["Draft", "Verify", "Inject", "Sampling and other"]

    def parts(r):
        return [r["draft_decode"] + r["draft_sample"] + r["draft_other"], r["verify"], r["inject"],
                r["tgt_sample"] + r["other"]]

    fig = new_fig(t, 8.4, 4.4, "Where a speculation round's time goes",
                  "Share of each round, LFM2.5-VL-3B Q4_K_M target, n_max 8 unless noted; "
                  "right: total ms per round and tokens advanced")
    legend(fig, t, segs, t["series"], 0.83)
    ax = fig.add_axes([0.31, 0.14, 0.50, 0.60])
    style_axes(ax, t, "x")
    ax.set_xlim(0, 100); ax.set_ylim(len(rows) - 0.5, -0.5)
    fig.canvas.draw()
    _, bh = px_to_data(ax, BAR_PX)
    gx, _ = px_to_data(ax, GAP_PX)
    table = []
    for i, (name, r) in enumerate(rows):
        p = parts(r); tot = sum(p)
        x = 0.0
        for k, v in enumerate(p):
            w = 100 * v / tot
            last = k == len(p) - 1
            x1 = x + w - (0 if last else gx)
            if x1 > x:
                rounded_rect(ax, x, x1, i - bh / 2, i + bh / 2, t["series"][k], "right" if last else None)
            x += w
        ax.text(102, i, f"{r['total']:.1f} ms · {r['tokens_per_round']:.2f} tok", va="center", ha="left",
                fontsize=8.5, color=t["ink"], clip_on=False)
        table.append([name] + [round(v, 2) for v in p] + [round(r["total"], 2), round(r["tokens_per_round"], 2)])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, _ in rows], fontsize=9, color=t["ink2"])
    ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.text(-0.6, -0.13, "Timing builds. Jetson in MAXN_SUPER; Jetson rows without 'fixes' use the unpatched "
            "runtime (llama.cpp 620d2225c).", transform=ax.transAxes, va="top", fontsize=8, color=t["muted"])
    save(fig, out, "fig1_round_breakdown", theme)
    if theme == "light":
        write_rows(out, "fig1_round_breakdown", ["config"] + [s + " ms" for s in segs] + ["total ms", "tokens/round"],
                   table)


def fig2(out, t, theme):
    j, d = read_csv("dev120_jetson_maxn.csv"), read_csv("dev120_rtx5070.csv")
    steps = [("A0", "Target\nonly"), ("A1", "+ F16\ndrafter"), ("A2", "+ C1–C3"), ("A3", "Q8_0\ndrafter"),
             ("A4", "P3\ndrafter"), ("A5", "n_max per\ndevice"), ("A6", "32k\nvocab"), ("A7", "+ C4")]
    jv = [float(j["B11" if c == "A7" else c]["tps"]) / float(j["A0"]["tps"]) for c, _ in steps]
    dv = [float(d[c]["tps"]) / float(d["A0"]["tps"]) for c, _ in steps]
    fig = new_fig(t, 8.4, 4.4, "Each step of the ablation, dev120",
                  "Decode speedup over unpatched target-only (A0), greedy, 120 samples in 6 task categories")
    legend(fig, t, ["Jetson Orin Nano Super (MAXN)", "RTX 5070"], t["series"][:2], 0.83)
    ax = fig.add_axes([0.07, 0.21, 0.90, 0.53])
    style_axes(ax, t, "y")
    last = len(steps) - 1
    grouped_columns(ax, t, [lab for _, lab in steps], ["Jetson", "RTX 5070"], [jv, dv],
                    labels_at={(0, 1), (1, 1), (0, last), (1, last)}, ymax=3.7)
    ax.set_yticks([0, 1, 2, 3]); ax.set_yticklabels(["0×", "1×", "2×", "3×"])
    ax.tick_params(axis="x", labelsize=8.5)
    ax.text(0, -0.22, "Jetson A7 is B11 (P3 drafter, 32k vocabulary, C1–C4, n_max 6).\nOn the RTX 5070 n_max is 8 "
            "in A4 and A5 alike: their difference is run-to-run drift (about 3 %).", transform=ax.transAxes,
            va="top", fontsize=8, color=t["muted"], linespacing=1.4)
    save(fig, out, "fig2_ablation", theme)
    if theme == "light":
        write_rows(out, "fig2_ablation", ["step", "label", "jetson_speedup", "rtx5070_speedup"],
                   [[c, lab.replace("\n", " "), round(a, 3), round(b, 3)] for (c, lab), a, b in zip(steps, jv, dv)])


def fig3(out, t, theme):
    j, j25, d = read_csv("dev120_jetson_maxn.csv"), read_csv("dev120_jetson_25w.csv"), read_csv("dev120_rtx5070.csv")
    ph, pc = read_csv("phone_htp_dev120.csv"), read_csv("phone_cpu_smoke24.csv")
    r = lambda tab, a, b: float(tab[a]["tps"]) / float(tab[b]["tps"])
    plats = ["Jetson\nMAXN", "Jetson\n25 W default", "RTX 5070", "Phone NPU\n(Hexagon)", "Phone CPU\n(smoke24)"]
    official = [r(j, "A1", "A0"), r(j25, "A1", "A0"), r(d, "A1", "A0"), r(ph, "B1", "B0"), r(pc, "B1", "B0")]
    best = [r(j, "B11", "A0"), r(j25, "B11", "A0"), r(d, "B11", "A0"), r(ph, "B11", "B0"), r(pc, "B11", "B0")]
    fig = new_fig(t, 8.4, 4.4, "Speedup by platform, dev120",
                  "Decode speedup over target-only; best = P3 drafter, 32k vocabulary, runtime fixes (B11)")
    legend(fig, t, ["Official F16 drafter as shipped", "Best configuration (B11)"], t["series"][:2], 0.83)
    ax = fig.add_axes([0.07, 0.17, 0.90, 0.57])
    style_axes(ax, t, "y")
    grouped_columns(ax, t, plats, ["official", "best"], [official, best], ymax=3.7)
    ax.axhline(1, color=t["muted"], linewidth=1, zorder=2)
    ax.set_yticks([0, 1, 2, 3]); ax.set_yticklabels(["0×", "1×", "2×", "3×"])
    ax.text(0, -0.2, "Jetson and RTX 5070: over unpatched target-only (A0); phone: over target-only (B0). "
            "Phone CPU on 24 samples.", transform=ax.transAxes, va="top", fontsize=8, color=t["muted"])
    save(fig, out, "fig3_platforms", theme)
    if theme == "light":
        write_rows(out, "fig3_platforms", ["platform", "official_f16_speedup", "best_b11_speedup"],
                   [[p.replace("\n", " "), round(a, 3), round(b, 3)] for p, a, b in zip(plats, official, best)])


def fig4(out, t, theme):
    cfgs = [("B1", "Official\nF16"), ("B2", "Q8_0"), ("B11", "P3, 32k vocab\n+ C4"), ("B12", "Q4_K_M, 16k\nvocab + C4")]
    panels = [("RTX 5070", read_csv("dev120_rtx5070.csv"), read_csv("ja75_rtx5070.csv")),
              ("Jetson Orin Nano Super (MAXN)", read_csv("dev120_jetson_maxn.csv"), read_csv("ja75_jetson_maxn.csv"))]
    fig = new_fig(t, 8.4, 4.4, "English vs Japanese",
                  "Decode speedup over target-only (B0): English dev120 (120 samples) vs Japanese ja75 (75 samples)")
    legend(fig, t, ["English", "Japanese"], t["series"][:2], 0.83)
    table = []
    for k, (name, en, ja) in enumerate(panels):
        ax = fig.add_axes([0.07 + k * 0.48, 0.19, 0.42, 0.50])
        style_axes(ax, t, "y")
        ev = [float(en[c]["speedup"]) for c, _ in cfgs]
        jv = [float(ja[c]["speedup"]) for c, _ in cfgs]
        grouped_columns(ax, t, [lab for _, lab in cfgs], ["en", "ja"], [ev, jv], labels_at={(0, 2), (1, 2)}, ymax=3.2)
        ax.axhline(1, color=t["muted"], linewidth=1, zorder=2)
        ax.set_yticks([0, 1, 2, 3]); ax.set_yticklabels(["0×", "1×", "2×", "3×"])
        ax.tick_params(axis="x", labelsize=8)
        ax.set_title(name, loc="left", fontsize=9.5, color=t["ink"], pad=6)
        table += [[name, c, round(a, 3), round(b, 3)] for (c, _), a, b in zip(cfgs, ev, jv)]
    save(fig, out, "fig4_japanese", theme)
    if theme == "light":
        write_rows(out, "fig4_japanese", ["platform", "config", "english_speedup", "japanese_speedup"], table)


def fig5(out, t, theme):
    a = json.load(open(f"{SUM}/e9_lang_analysis_q8.json"))
    pts = []
    for grp, cls, name in [("ja_caption", "kanji", "Kanji"), ("ja_caption", "hiragana", "Hiragana"),
                           ("ja_caption", "katakana", "Katakana"), ("ja_caption", "ja_punct", "Japanese punctuation"),
                           ("en_caption", "latin", "English words")]:
        acc, n = a[grp]["accept_by_class"][cls]
        conf = a[grp]["target_confidence"]["mean_p_by_class"][cls]
        pts.append((name, conf, acc / n, n))
    fig = new_fig(t, 8.4, 4.6, "Japanese captions: acceptance follows the target's own confidence",
                  "Per character class, same 15 COCO images, Q8_0 drafter (full vocabulary), RTX 5070")
    ax = fig.add_axes([0.09, 0.13, 0.86, 0.63])
    style_axes(ax, t, "y")
    ax.grid(axis="x", color=t["grid"], linewidth=0.75)
    ax.set_xlim(0.5, 0.85); ax.set_ylim(0.5, 1.0)
    ax.plot([0.5, 0.85], [0.5, 0.85], color=t["muted"], linewidth=1, zorder=2)
    ax.text(0.62, 0.605, "acceptance = confidence", ha="left", va="top", fontsize=8, color=t["muted"])
    # label placement per point (dx, dy, ha, va): the three mid-range classes sit close together
    place = {"Kanji": (0.006, 0.012, "left", "bottom"), "English words": (-0.006, 0.010, "right", "bottom"),
             "Hiragana": (0.0, 0.018, "center", "bottom"), "Katakana": (0.006, -0.012, "left", "top"),
             "Japanese punctuation": (-0.006, 0.010, "right", "bottom")}
    for name, x, y, n in pts:
        ax.scatter([x], [y], s=64, color=t["series"][0], edgecolor=t["surface"], linewidth=2, zorder=4)
        dx, dy, ha, va = place[name]
        ax.text(x + dx, y + dy, f"{name}  {y:.2f}", ha=ha, va=va, fontsize=8.5, color=t["ink"], zorder=5)
    ax.set_xlabel("Target top-1 probability (target-only run)", fontsize=9, color=t["ink2"])
    ax.set_ylabel("Draft acceptance", fontsize=9, color=t["ink2"])
    ax.text(0, -0.15, "Kanji are 37.5 % of drafted positions in Japanese captions; English words are from the "
            "English captions of the same images.", transform=ax.transAxes, va="top", fontsize=8, color=t["muted"])
    save(fig, out, "fig5_ja_confidence", theme)
    if theme == "light":
        write_rows(out, "fig5_ja_confidence", ["class", "target_top1", "acceptance", "drafted_positions"],
                   [[nm, round(x, 3), round(y, 3), n] for nm, x, y, n in pts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/results/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    for theme, t in THEMES.items():
        for f in (fig1, fig2, fig3, fig4, fig5):
            f(a.out, t, theme)
    print("written:", sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
