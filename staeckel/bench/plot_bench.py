"""Plot the two benchmark JSONs into PNGs: ``python plot_bench.py`` (any env with matplotlib)."""

import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# validated categorical slots 1-4 (light) + neutral ink for the gala reference
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GALA, INK, MUTED, GRID = "#0b0b0b", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 2, "lines.markersize": 5.5, "legend.frameon": False,
})


def load(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


def per_pt(rows, ns, key="wall"):
    return np.array([r[key] for r in rows]) / np.asarray(ns) * 1e6


def load_avgs(env):
    pick = lambda s: re.search(r"load averages?: ([\d.]+)", s).group(1)  # noqa: E731
    return f"1-min load {pick(env['uptime_before'])} before / {pick(env['uptime_after'])} after"


def caption(fig, env, extra=""):
    fig.text(0.01, 0.005,
             f"{env['machine']} ({env['ncpu']} cores), JAX {env['jax']} x64, gala {env['gala']}, "
             f"galax {env['galax_commit']}, {env['date'][:10]}; {load_avgs(env)} "
             f"(a Zoom call used ~0.6-0.8 core). Best-of-N timings (N>=3). {extra}",
             fontsize=7, color=MUTED, ha="left", va="bottom", wrap=True)


def label_end(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x[-1], y[-1]), xytext=(5, dy), textcoords="offset points",
                color=INK, fontsize=8, va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, lw=1))


def fig_delta():
    r = load("pr969-delta-vs-gala.json")
    env, ns = r["env"], np.array(r["Ns"])
    names = list(r["results"])
    fig, axs = plt.subplots(2, len(names), figsize=(11, 7.4), sharex=True,
                            gridspec_kw={"height_ratios": [1.5, 1]})
    for j, name in enumerate(names):
        d = r["results"][name]
        ax, bx = axs[0, j], axs[1, j]
        series = [("galax, jit (warm)", "galax_jit", BLUE, "o", "-"),
                  ("galax, eager (warm)", "galax_eager", ORANGE, "s", "-"),
                  ("gala get_staeckel_fudge_delta", "gala", GALA, "D", "--")]
        for lab, k, col, m, ls in series:
            y = per_pt(d[k], ns)
            ax.loglog(ns, y, ls, color=col, marker=m, label=lab)
            label_end(ax, ns, y, lab.split(" get_")[0].replace(" (warm)", ""), col,
                      dy={"galax_eager": 7, "galax_jit": -7}.get(k, 0))
        comp = np.median([c["wall"] for c in d["galax_jit_compile"]])
        ax.text(0.97, 0.97, f"jit first-call compile ~{comp:.2f} s\n(one-off, not in curves)",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=MUTED)
        ax.set_title(name, loc="left", fontweight="bold", color=INK)
        ax.set_ylabel("wall time per point [µs]")
        ax.set_xlim(0.6, ns[-1] * 12)

        g_w, g_c = per_pt(d["gala"], ns), per_pt(d["gala"], ns, "cpu")
        for lab, k, key, col, ls, m in [
            ("jit, wall", "galax_jit", "wall", BLUE, "-", "o"),
            ("jit, per CPU-s", "galax_jit", "cpu", BLUE, ":", "^"),
            ("eager, wall", "galax_eager", "wall", ORANGE, "-", "s"),
        ]:
            ref = g_w if key == "wall" else g_c
            y = ref / per_pt(d[k], ns, key)
            bx.loglog(ns, y, ls, color=col, marker=m, label=lab)
            label_end(bx, ns, y, lab, col, dy={"jit, per CPU-s": -8, "jit, wall": 7}.get(lab, 0))
        bx.axhline(1, color=MUTED, lw=1)
        bx.text(1, 1.08, "parity", color=MUTED, fontsize=8, va="bottom")
        bx.set_ylabel("speedup vs gala  (gala time / galax time)")
        bx.set_xlabel("N points per call")
    ymax = max(a.get_ylim()[1] for a in axs[0]); ymin = min(a.get_ylim()[0] for a in axs[0])
    for a in axs[0]:
        a.set_ylim(ymin, ymax)
    fig.suptitle("PR #969: galax staeckel_fudge_delta vs gala get_staeckel_fudge_delta "
                 "(median=False), random disk points", x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, 0.935, "Lower is better (top); higher is better (bottom). gala's CPU time ~= its "
             "wall time (single thread); XLA threads across cores, so 'per CPU-s' compares "
             "throughput per core-second.", fontsize=8, color=MUTED)
    caption(fig, env, "Outputs verified to agree (max rel. diff <1e-9, identical NaN masks) before timing.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(os.path.join(HERE, "pr969-delta-vs-gala.png"), dpi=170)


def fig_stages():
    r = load("pr3a-staeckel-stages-vs-gala.json")
    env, ns = r["env"], np.array(r["Ns"])
    names = list(r["results"])
    stages = [("(a) integrals of motion", "a_iom", BLUE, "o"),
              ("(b) + turning points", "b_limits", ORANGE, "s"),
              ("(c) + tangents + Taylor", "c_tangents_taylor", AQUA, "^"),
              ("(d) = (c) value + grad", "d_value_and_grad", YELLOW, "v")]
    fig, axs = plt.subplots(2, len(names), figsize=(11.5, 7.8), sharex=True,
                            gridspec_kw={"height_ratios": [1.5, 1]})
    for j, name in enumerate(names):
        d = r["results"][name]
        ax, bx = axs[0, j], axs[1, j]
        g_w, g_c = per_pt(d["gala_full"], ns), per_pt(d["gala_full"], ns, "cpu")
        ax.loglog(ns, g_w / 1.5, "-", color=MUTED, lw=1.2)
        ax.text(0.02, 0.03, "thin grey line: wall-time ceiling for the FULL galax pipeline "
                "(gala / 1.5)", transform=ax.transAxes, color=MUTED, fontsize=7.5)
        ax.loglog(ns, g_w, "--", color=GALA, marker="D", label="gala full C++ pipeline")
        label_end(ax, ns, g_w, "gala FULL (J, θ, Ω)", GALA,
                  dy=-10 if 1 < per_pt(d["d_value_and_grad"], ns)[-1] / g_w[-1] < 2 else 10)
        for lab, k, col, m in stages:
            y = per_pt(d[k], ns)
            ax.loglog(ns, y, "-", color=col, marker=m, label=lab)
            dy = {"b_limits": -7, "c_tangents_taylor": 7}.get(k, 0)
            if k == "d_value_and_grad":
                dy = -10 if g_w[-1] > y[-1] else 10
            label_end(ax, ns, y, lab.split(" ")[0], col, dy=dy)
        ax.set_title(f"{name}   (Δ = {d['delta']:.3f} kpc, fixed)", loc="left",
                     fontweight="bold", color=INK)
        ax.set_ylabel("wall time per point [µs]")
        ax.set_xlim(60, ns[-1] * 6)

        for lab, k, col, m in stages[2:]:
            bx.loglog(ns, g_w / per_pt(d[k], ns), "-", color=col, marker=m)
            bx.loglog(ns, g_c / per_pt(d[k], ns, "cpu"), ":", color=col, marker=m, mfc="white")
            label_end(bx, ns, g_w / per_pt(d[k], ns), lab.split(" ")[0] + " wall", col, dy=6)
            label_end(bx, ns, g_c / per_pt(d[k], ns, "cpu"), lab.split(" ")[0] + " per CPU-s",
                      col, dy=-6)
        bx.axhline(1.5, color=MUTED, lw=1.2, ls="-")
        bx.axhline(0.5, color=MUTED, lw=1.2, ls=":")
        bx.text(0.02, 0.97, "floors for the FULL pipeline:\n"
                "solid grey 1.5× wall,  dotted grey 0.5× per CPU-s",
                transform=bx.transAxes, color=MUTED, fontsize=7.5, va="top")
        bx.set_ylabel("galax-stage speedup vs gala FULL\n(solid wall, dotted per CPU-s)")
        bx.set_xlabel("N points per call")
    lo = min(a.get_ylim()[0] for a in axs[0]); hi = max(a.get_ylim()[1] for a in axs[0])
    for a in axs[0]:
        a.set_ylim(lo * 0.3, hi)  # room for the floor note under the curves
    lo = min(a.get_ylim()[0] for a in axs[1]); hi = max(a.get_ylim()[1] for a in axs[1])
    for a in axs[1]:
        a.set_ylim(lo, hi * 4)  # room for the floor note above the curves
    fig.suptitle("PR 3a: galax Staeckel stages (jit, warm) vs gala's full C++ Staeckel fudge",
                 x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, 0.94, "PARTIAL PIPELINE: galax curves stop before quadrature, actions, angles and "
             "frequencies (not in PR 3a). gala curve is the FULL pipeline.",
             fontsize=9, color="#b0281a", fontweight="bold")
    fig.text(0.01, 0.92, "The floors apply to the eventual full galax pipeline, so a partial stage "
             "must sit well inside them. Same points and Δ for both codes.",
             fontsize=8, color=MUTED)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.01, 0.915), ncol=5, fontsize=8)
    caption(fig, env, "The 'after' load is inflated by this run's own XLA threads (no other "
            "heavy process was running).")
    fig.tight_layout(rect=(0, 0.03, 1, 0.885))
    fig.savefig(os.path.join(HERE, "pr3a-staeckel-stages-vs-gala.png"), dpi=170)


if __name__ == "__main__":
    fig_delta()
    if os.path.exists(os.path.join(HERE, "pr3a-staeckel-stages-vs-gala.json")):
        fig_stages()
