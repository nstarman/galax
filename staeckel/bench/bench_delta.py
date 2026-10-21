"""PR #969: galax ``staeckel_fudge_delta`` vs gala ``get_staeckel_fudge_delta``.

Run from the PR-969 checkout (galax @ 06559c18, gala 1.10.1 in its .venv):

    cd <pr964 checkout> && JAX_ENABLE_X64=1 .venv/bin/python <this file> [out.json]
"""

import datetime
import json
import os
import platform
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")  # gala/astropy deprecation chatter

import jax

jax.config.update("jax_enable_x64", True)
import astropy.units as apyu
import gala
import gala.dynamics as gad
import gala.potential as gap
import jax.numpy as jnp
import numpy as np
from gala.dynamics.actionangle import get_staeckel_fudge_delta
from gala.units import galactic

import galax.dynamics as gd
import galax.potential as gp
import unxt as u

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "pr969-delta-vs-gala.json"
)
NS = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
MIN_REPS, MAX_REPS, BUDGET = 3, 15, 2.0  # best-of-N, stop after ~BUDGET s


def uptime():
    return subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()


def best_of(fn):
    """Best wall (and the CPU time of that same call) over repeated calls."""
    best, cpu, spent, reps = np.inf, np.nan, 0.0, 0
    while reps < MIN_REPS or (reps < MAX_REPS and spent < BUDGET):
        w0, c0 = time.perf_counter(), time.process_time()
        fn()
        dw, dc = time.perf_counter() - w0, time.process_time() - c0
        spent, reps = spent + dw, reps + 1
        if dw < best:
            best, cpu = dw, dc
    return {"wall": best, "cpu": cpu, "reps": reps}


def first_call(fn):
    w0, c0 = time.perf_counter(), time.process_time()
    fn()
    return {"wall": time.perf_counter() - w0, "cpu": time.process_time() - c0}


def disk_points(n, seed=0):
    rng = np.random.default_rng(seed)
    R = rng.uniform(0.5, 15, n)
    ang = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(-1, 1, n)
    q = np.stack([R * np.cos(ang), R * np.sin(ang), z], -1)
    v = rng.normal(0, 60, (n, 3))  # km/s; only enters gala's PSP container
    v[:, 1] += 200
    return q, v


POTS = {
    "MiyamotoNagai": (
        gp.MiyamotoNagaiPotential(
            m_tot=u.Q(1e11, "Msun"), a=u.Q(3.0, "kpc"), b=u.Q(0.3, "kpc"),
            units="galactic",
        ),
        gap.MiyamotoNagaiPotential(m=1e11, a=3.0, b=0.3, units=galactic),
    ),
    "MilkyWayPotential2022": (gp.MilkyWayPotential2022(), gap.MilkyWayPotential2022()),
}

res = {
    "env": {
        "machine": f"{platform.machine()} "
        + subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                         capture_output=True, text=True).stdout.strip(),
        "ncpu": os.cpu_count(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jax_x64": bool(jax.config.jax_enable_x64),
        "gala": gala.__version__,
        "galax_dynamics_path": os.path.dirname(gd.__file__),
        "galax_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                       capture_output=True, text=True).stdout.strip(),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "uptime_before": uptime(),
    },
    "Ns": NS,
    "results": {},
}
print(res["env"], flush=True)

for name, (gx, ga) in POTS.items():
    out = {k: [] for k in ("galax_jit", "galax_eager", "gala")}
    out["galax_jit_compile"], out["galax_eager_first"], out["max_rel_diff"] = [], [], []
    jitted = jax.jit(lambda q, t, _p=gx: gd.actionangle.staeckel_fudge_delta(_p, q, t))
    for n in NS:
        qn, vn = disk_points(n)
        q, t = jnp.asarray(qn), jnp.zeros(n)
        psp = gad.PhaseSpacePosition(pos=qn.T * apyu.kpc, vel=vn.T * apyu.km / apyu.s)

        gala_call = lambda: get_staeckel_fudge_delta(ga, psp, median=False)  # noqa: E731
        jit_call = lambda: jitted(q, t).block_until_ready()  # noqa: E731
        eager_call = lambda: jax.block_until_ready(  # noqa: E731
            gd.actionangle.staeckel_fudge_delta(gx, q, t)
        )

        # eager first call at this shape (per-shape op compile caches still cold)
        out["galax_eager_first"].append(first_call(eager_call))
        # correctness
        d_gala = np.asarray(gala_call().to_value(apyu.kpc))
        d_jit = np.asarray(jitted(q, t))
        ok = np.isfinite(d_gala) & np.isfinite(d_jit)
        rel = float(np.max(np.abs(d_jit[ok] - d_gala[ok]) / np.abs(d_gala[ok]))) if ok.any() else np.nan
        out["max_rel_diff"].append({"rel": rel, "n_finite": int(ok.sum()),
                                    "n_nan_galax": int(np.isnan(d_jit).sum()),
                                    "n_nan_gala": int(np.isnan(d_gala).sum()),
                                    "nan_mask_equal": bool((np.isnan(d_jit) == np.isnan(d_gala)).all())})
        # compile cost: a fresh jit wrapper per N (shape-specialised anyway)
        fresh = jax.jit(lambda q, t, _p=gx: gd.actionangle.staeckel_fudge_delta(_p, q, t))
        out["galax_jit_compile"].append(first_call(lambda: fresh(q, t).block_until_ready()))

        out["galax_jit"].append(best_of(jit_call))
        out["galax_eager"].append(best_of(eager_call))
        out["gala"].append(best_of(gala_call))
        r = {k: out[k][-1]["wall"] / n * 1e6 for k in ("galax_jit", "galax_eager", "gala")}
        print(f"{name:22s} N={n:>8} us/pt jit {r['galax_jit']:9.3f} eager {r['galax_eager']:9.3f}"
              f" gala {r['gala']:9.3f} | compile {out['galax_jit_compile'][-1]['wall']:.2f}s"
              f" | maxrel {rel:.1e}", flush=True)
    res["results"][name] = out

res["env"]["uptime_after"] = uptime()
print(res["env"]["uptime_after"])
with open(OUT, "w") as f:
    json.dump(res, f, indent=1)
print("wrote", OUT)
