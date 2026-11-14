"""Regenerate the README charts from an experiment file.

    .venv/bin/python docs/make_charts.py experiment.json

The experiment file holds watermarked and plain replies (token ids and text)
from one model, plus the key that generated them (see docs/run_experiment.py).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ripple_fingerprint import Watermarker  # noqa: E402

OUT = Path(__file__).parent
FPR = 1e-5

# ---- style: validated categorical slots 1-2 (blue, orange) on a light surface
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e7e6e2", "#ffffff"
BLUE, ORANGE, NULL = "#2a78d6", "#eb6834", "#d9d8d3"
plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10.5, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": INK2, "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6, "ytick.major.pad": 6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "legend.frameon": False, "legend.fontsize": 10,
})


def figure(title: str, subtitle: str, size=(8, 4.2)):
    fig, ax = plt.subplots(figsize=size, dpi=200)
    fig.subplots_adjust(left=0.09, right=0.84, top=0.80, bottom=0.14)
    fig.text(0.09, 0.945, title, fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.09, 0.875, subtitle, fontsize=10.5, color=INK2, va="top")
    ax.grid(axis="x", visible=False)
    return fig, ax


def label_end(ax, x, y, text, color, dx=8, dy=0, ha="left"):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", va="center", ha=ha,
                fontsize=10, color=INK, fontweight="bold", annotation_clip=False)
    ax.plot([x], [y], "o", ms=5, color=color, mec=SURF, mew=1.5, zorder=5)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("wrote", OUT / name)


def main(path: str):
    exp = json.loads(Path(path).read_text())
    wm = Watermarker(exp["key"].encode())
    tok = AutoTokenizer.from_pretrained(exp["model"])
    wmk = [r for r in exp["runs"] if r["watermark"]]
    plain = [r for r in exp["runs"] if not r["watermark"]]
    model_name = Path(exp["model"]).name
    N = max(len(r["ids"]) for r in exp["runs"])

    # ------------------------------------------------ 1. evidence accumulates
    fig, ax = figure("The evidence grows with every token",
                     f"Cumulative detector score above what ordinary text gives · {len(wmk)} replies each · {model_name}")
    x = np.arange(1, N + 1)
    ax.fill_between(x, -2 * np.sqrt(x), 2 * np.sqrt(x), color=NULL, alpha=0.55, lw=0, label="ordinary-text range (±2σ)")

    def curves(runs):
        cs = []
        for r in runs:
            s = [v for v in wm.token_scores(r["ids"])]
            c, tot, n = [], 0.0, 0
            for v in s:
                if v is not None:
                    tot += v; n += 1
                c.append(tot - n)
            cs.append(np.array(c))
        return cs

    meds = {}
    for runs, color, name in ((wmk, BLUE, "with fingerprint"), (plain, ORANGE, "without")):
        cs = curves(runs)
        for c in cs:
            ax.plot(np.arange(1, len(c) + 1), c, color=color, lw=0.8, alpha=0.28)
        L = min(len(c) for c in cs)
        med = np.median(np.stack([c[:L] for c in cs]), axis=0)
        ax.plot(np.arange(1, L + 1), med, color=color, lw=2.2)
        meds[name] = (L, med[-1], color)
    for name, (L, y, color) in meds.items():
        label_end(ax, L, y, f"{name}\n(median)", color)
    ax.set_xlabel("tokens generated")
    ax.set_ylabel("excess score")
    ax.set_xlim(0, N)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.legend(loc="upper left")
    save(fig, "chart-evidence.png")

    # ------------------------------------------------ 2. detection rate vs length
    fig, ax = figure("Longer replies are easier to catch",
                     f"Share of replies flagged (p < {FPR:g}) using only their first n tokens · {model_name}")
    lens = np.arange(10, N + 1, 10)

    def rate(runs):
        return np.array([np.mean([wm.detect(r["ids"][:L]).p_value < FPR for r in runs]) * 100 for L in lens])

    rw, rp = rate(wmk), rate(plain)
    ax.plot(lens, rw, color=BLUE, lw=2.2)
    ax.plot(lens, rp, color=ORANGE, lw=2.2)
    label_end(ax, lens[-1], rw[-1], "with fingerprint", BLUE)
    label_end(ax, lens[-1], rp[-1], "without", ORANGE)
    ax.set_ylim(-4, 104)
    ax.set_xlim(0, N + 5)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("tokens checked")
    save(fig, "chart-detection-length.png")

    # ------------------------------------------------ 3. robustness to word edits
    fig, ax = figure("Light editing leaves it detectable",
                     "Share of full replies still flagged after replacing random words (3 random edits per reply)")
    rates_w, rates_p = [], []
    fills = ["the", "very", "also", "quite", "really", "just", "some", "often"]
    rng = np.random.default_rng(0)

    def edit(text, frac):
        parts = re.split(r"(\s+)", text)
        return "".join(rng.choice(fills) if (i % 2 == 0 and p and rng.random() < frac) else p
                       for i, p in enumerate(parts))

    fracs = np.arange(0, 0.51, 0.05)
    for f in fracs:
        def hit(runs):
            return np.mean([wm.detect(tok.encode(edit(r["text"], f), add_special_tokens=False)).p_value < FPR
                            for r in runs for _ in range(3)]) * 100
        rates_w.append(hit(wmk)); rates_p.append(hit(plain))
    xs = fracs * 100
    ax.plot(xs, rates_w, color=BLUE, lw=2.2, marker="o", ms=4, mec=SURF, mew=1.2)
    ax.plot(xs, rates_p, color=ORANGE, lw=2.2, marker="o", ms=4, mec=SURF, mew=1.2)
    label_end(ax, xs[2], rates_w[2], "with fingerprint", BLUE, dx=10, dy=12)
    label_end(ax, xs[-1], rates_p[-1], "without", ORANGE)
    ax.set_ylim(-4, 104)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xticks(xs[::2], [f"{v:.0f}%" for v in xs[::2]])
    ax.set_xlabel("words replaced")
    save(fig, "chart-robustness.png")
    print("robustness:", {f"{v:.0f}%": round(r) for v, r in zip(xs, rates_w)})

    # ------------------------------------------------ 4. false-positive calibration
    K = 60
    pv = []
    for k in range(K):  # each independent key is a fresh, exact null test of the same text
        other = Watermarker(f"calibration-key-{k:04d}-xxxxxxxxxxxx".encode())
        pv += [other.detect(r["ids"]).p_value for r in exp["runs"]]
    pv = np.array(pv)
    fig, ax = figure("Unmarked text looks exactly like chance",
                     f"p-values of {len(pv)} tests on text the key didn't write ({len(exp['runs'])} replies × {K} other keys)")
    bins = np.linspace(0, 1, 21)
    counts, _ = np.histogram(pv, bins)
    ax.bar(bins[:-1], counts, width=0.05, align="edge", color=BLUE, edgecolor=SURF, linewidth=2)
    exp_count = len(pv) / 20
    ax.axhline(exp_count, color=INK, lw=1.2, ls=(0, (4, 3)))
    ax.annotate("expected if exactly uniform", (0.995, exp_count), xytext=(0, 14), textcoords="offset points",
                ha="right", va="center", fontsize=10, color=INK, fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.25", fc=SURF, ec="none"))
    ax.set_ylim(0, max(counts.max(), exp_count) * 1.25)
    ax.set_xlim(0, 1)
    ax.set_xlabel("p-value")
    ax.set_ylabel("tests")
    ax.grid(axis="y")
    save(fig, "chart-calibration.png")
    print(f"calibration: min p {pv.min():.1e}, share below 0.01: {np.mean(pv < 0.01):.3f}")

    # ------------------------------------------------ 5b. speed per prompt, fingerprint on vs off
    # Prefer the dedicated benchmark (several alternating runs per prompt; see run_speed.py)
    sp_path = OUT / "speed-pairs.json"
    if sp_path.exists():
        sp = json.loads(sp_path.read_text())
        rows = [{"prompt": r["prompt"], "w": float(np.median(r["with"])), "p": float(np.median(r["without"]))} for r in sp["rows"]]
        how = f"median of {sp['reps']} alternating runs per prompt, {sp['max_new_tokens']} tokens"
    else:
        rows = [{"prompt": w["prompt"], "w": w["tok_s"], "p": p["tok_s"]} for w, p in zip(wmk, plain)]
        how = "one run per prompt"
    rows.sort(key=lambda r: r["p"])
    ws = np.array([r["w"] for r in rows]); ps = np.array([r["p"] for r in rows])
    ratio = np.median(ws / ps)
    fig, ax = figure("Fingerprinting doesn't slow generation down",
                     f"Tokens per second for each prompt ({how}) · median {np.median(ws):.1f} with vs "
                     f"{np.median(ps):.1f} without ({ratio:.0%})", size=(8, 5.8))
    fig.subplots_adjust(left=0.36, right=0.96, top=0.78, bottom=0.10)
    pairs = [({"prompt": r["prompt"]}, None) for r in rows]
    yy = np.arange(len(pairs))
    for yi, w, p in zip(yy, ws, ps):
        ax.plot([min(w, p), max(w, p)], [yi, yi], color=NULL, lw=3, solid_capstyle="round", zorder=1)
    ax.scatter(ps, yy, s=46, color=ORANGE, edgecolor=SURF, linewidth=1.5, zorder=3, label="without fingerprint")
    ax.scatter(ws, yy, s=46, color=BLUE, edgecolor=SURF, linewidth=1.5, zorder=4, label="with fingerprint")
    short = lambda s: (s[:34] + "…") if len(s) > 35 else s
    ax.set_yticks(yy, [short(w["prompt"]) for w, _ in pairs])
    ax.tick_params(axis="y", labelsize=9, labelcolor=INK2)
    lo = min(ws.min(), ps.min()); hi = max(ws.max(), ps.max())
    ax.set_xlim(max(0, lo - 6), hi + 4)
    ax.set_xlabel("tokens per second")
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(0.085, 0.845), ncol=2,
               handletextpad=0.3, columnspacing=1.6)
    save(fig, "chart-speed-paired.png")
    print(f"speed: median with {np.median(ws):.1f}, without {np.median(ps):.1f}, median ratio {ratio:.3f}")

    # ------------------------------------------------ 5. generation speed
    speed = json.loads((OUT / "speed.json").read_text())
    fig, ax = figure("Fingerprinting at native speed",
                     f"Tokens per second as a share of MLX's own sampling loop · {speed['model']} · {speed['machine']}",
                     size=(8, 3.6))
    fig.subplots_adjust(left=0.36, right=0.92, top=0.76, bottom=0.10)
    rows = speed["rows"][::-1]
    names = [r["name"] for r in rows]
    vals = [r["share"] * 100 for r in rows]
    colors = [NULL if r.get("reference") else (MUTED if r.get("before") else BLUE) for r in rows]
    y = np.arange(len(rows))
    ax.barh(y, vals, height=0.62, color=colors, edgecolor=SURF, linewidth=2)
    for yi, v, r in zip(y, vals, rows):
        ax.text(v + 1.5, yi, f"{v:.0f}%", va="center", fontsize=10.5, color=INK, fontweight="bold")
    ax.set_yticks(y, names)
    ax.tick_params(axis="y", labelsize=10.5, labelcolor=INK)
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.spines["bottom"].set_visible(False)
    save(fig, "chart-speed.png")


if __name__ == "__main__":
    main(sys.argv[1])
