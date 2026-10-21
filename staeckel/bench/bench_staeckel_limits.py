"""PR 3a: galax Staeckel stages (partial pipeline) vs gala's full C++ Staeckel fudge.

Run from the staeckel/3a-limits worktree (galax @ 8434d869), with gala's Staeckel
C++ extension build first on the path:

    cd <galax worktree> && PYTHONPATH=<gala build/lib.macosx-11.0-arm64-cpython-312> \
        JAX_ENABLE_X64=1 uv run --frozen python <this file> [out.json]
"""

import datetime
import json
import os
import platform
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import jax

jax.config.update("jax_enable_x64", True)
import gala
import gala.potential as gap
import jax.numpy as jnp
import numpy as np
from gala.dynamics.actionangle._staeckel import staeckel_fudge_c
from gala.units import galactic

import galax.potential as gp
import unxt as u
from galax.dynamics._src.actionangle._staeckel_core import integrals_of_motion, make_F
from galax.dynamics._src.actionangle._staeckel_limits import (
    attach_tangents,
    default_phi_inf,
    find_limits,
    taylor_coeffs,
)
from galax.dynamics._src.actionangle.staeckel import _delta

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "pr3a-staeckel-stages-vs-gala.json"
)
NS = [100, 1_000, 10_000, 100_000, 1_000_000]
MIN_REPS, MAX_REPS, BUDGET = 3, 15, 2.0
REF_RZ = (8.0, 0.5)  # Delta fixed per potential: staeckel_fudge_delta here
T0 = jnp.asarray(0.0)
sg = jax.lax.stop_gradient


def uptime():
    return subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()


def best_of(fn):
    best, cpu, spent, reps = np.inf, np.nan, 0.0, 0
    while reps < MIN_REPS or (reps < MAX_REPS and spent < BUDGET):
        w0, c0 = time.perf_counter(), time.process_time()
        fn()
        dw, dc = time.perf_counter() - w0, time.process_time() - c0
        spent, reps = spent + dw, reps + 1
        if dw < best:
            best, cpu = dw, dc
    return {"wall": best, "cpu": cpu, "reps": reps}


def phase_points(n, vc, seed=0):
    """Disk-like points: R~U(0.5,15), z~U(-1,1), v ~ rotation vc + 0.3 vc dispersion."""
    rng = np.random.default_rng(seed)
    R = rng.uniform(0.5, 15, n)
    ang = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(-1, 1, n)
    v = rng.normal(0, 0.3 * vc, (n, 3))
    v[:, 1] += vc
    c, s = np.cos(ang), np.sin(ang)
    return np.ascontiguousarray(
        np.stack([R * c, R * s, z, v[:, 0] * c - v[:, 1] * s,
                  v[:, 0] * s + v[:, 1] * c, v[:, 2]], 1)
    )


def stages(pot):
    def mk(p):
        return lambda R, z: p._potential(jnp.stack([R, jnp.zeros_like(R), z], -1), T0)

    phi, phiz = mk(pot), mk(jax.tree.map(sg, pot))

    def a_iom(w, d2):
        i = integrals_of_motion(phi, w, d2)
        return i.E + i.I3 + i.Lz

    def b_lim(w, d2):
        i = integrals_of_motion(phiz, w, d2)
        l = find_limits(make_F(phiz, i), i, default_phi_inf(phiz, i))
        return l.lam_min + l.lam_max + l.nu_max

    def c_full(w, d2):
        iz = integrals_of_motion(phiz, sg(w), d2)
        Fz = make_F(phiz, iz)
        l = find_limits(Fz, iz, default_phi_inf(phiz, iz))
        i = integrals_of_motion(phi, w, d2)
        F = make_F(phi, i)
        l = attach_tangents(l, F, Fz)
        t = taylor_coeffs(F, l)
        return l.lam_min + l.lam_max + l.nu_max + t.c2 + t.c3 + t.c4 + t.F2 + t.F4

    def d_grad(w, d2):
        return jax.value_and_grad(lambda w: jnp.nansum(jax.vmap(c_full)(w, d2)))(w)

    return {
        "a_iom": jax.jit(jax.vmap(a_iom)),
        "b_limits": jax.jit(jax.vmap(b_lim)),
        "c_tangents_taylor": jax.jit(jax.vmap(c_full)),
        "d_value_and_grad": jax.jit(d_grad),
    }


POTS = {
    "MiyamotoNagai": (
        gp.MiyamotoNagaiPotential(m_tot=u.Q(1e11, "Msun"), a=u.Q(3.0, "kpc"),
                                  b=u.Q(0.3, "kpc"), units="galactic"),
        gap.MiyamotoNagaiPotential(m=1e11, a=3.0, b=0.3, units=galactic),
        0.2,  # kpc/Myr
    ),
    "MilkyWayPotential2022": (gp.MilkyWayPotential2022(), gap.MilkyWayPotential2022(), 0.22),
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
        "gala_path": os.path.dirname(gala.__file__),
        "galax_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                       capture_output=True, text=True).stdout.strip(),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "uptime_before": uptime(),
        "delta_ref_point_R_z_kpc": REF_RZ,
    },
    "Ns": NS,
    "results": {},
}
print(res["env"], flush=True)

for name, (gx, ga, vc) in POTS.items():
    delta = float(_delta(gx, jnp.asarray([[REF_RZ[0], 0.0, REF_RZ[1]]]), jnp.zeros(1))[0])
    fns = stages(gx)
    out = {k: [] for k in [*fns, "gala_full"]}
    out["delta"] = delta
    out["compile_s"] = {k: [] for k in fns}
    out["finite_frac"] = {"galax_b_limits": [], "gala_Jr": []}
    for n in NS:
        wn = phase_points(n, vc)
        dn = np.full(n, delta)
        w, D2 = jnp.asarray(wn), jnp.asarray(dn**2)
        for k, f in fns.items():
            t0 = time.perf_counter()
            c = f.lower(w, D2).compile()
            out["compile_s"][k].append(time.perf_counter() - t0)
            r = jax.block_until_ready(c(w, D2))
            if k == "b_limits":
                out["finite_frac"]["galax_b_limits"].append(float(np.isfinite(r).mean()))
            out[k].append(best_of(lambda c=c: jax.block_until_ready(c(w, D2))))
        g = staeckel_fudge_c(ga, wn, dn)
        out["finite_frac"]["gala_Jr"].append(float(np.isfinite(g[:, 0]).mean()))
        out["gala_full"].append(best_of(lambda: staeckel_fudge_c(ga, wn, dn)))
        print(f"{name:22s} N={n:>8} us/pt " + " ".join(
            f"{k}={out[k][-1]['wall'] / n * 1e6:.3f}(cpu {out[k][-1]['cpu'] / n * 1e6:.3f})"
            for k in out if isinstance(out[k], list)), flush=True)
    res["results"][name] = out

res["env"]["uptime_after"] = uptime()
print(res["env"]["uptime_after"])
with open(OUT, "w") as f:
    json.dump(res, f, indent=1)
print("wrote", OUT)
