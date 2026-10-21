# Staeckel-fudge benchmarks vs gala

Reference machine: Apple M2 Pro (10 cores), JAX x64. Each timing is best-of-N
(N ≥ 3; up to 15 or ~2 s per point), wall time from `time.perf_counter` and CPU time
from `time.process_time` (all threads) for the same best call. `uptime` is
recorded before and after each run in the JSON.

## Rerun

```sh
S=/private/tmp/claude-501/-Users-nmrs-local-galax--claude-worktrees-relaxed-feistel-18a2be/abbd83f9-4bb8-4c32-833a-7a316d0cabf2/scratchpad
B=$S/bench

# PR #969 (galax @ 06559c18, gala 1.10.1 in that checkout's .venv)
cd $S/pr964
JAX_ENABLE_X64=1 .venv/bin/python $B/bench_delta.py

# PR 3a (galax @ 8434d869); gala's Staeckel C++ build goes first on the path
cd /Users/nmrs/local/galax/.claude/worktrees/relaxed-feistel-18a2be
PYTHONPATH=/Users/nmrs/.cache/uv/git-v1/checkouts/aa61f6cc8cf2ca48/6cd50cef/build/lib.macosx-11.0-arm64-cpython-312 \
  JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 uv run --frozen python $B/bench_staeckel_limits.py

# figures (any env with matplotlib)
$S/pr964/.venv/bin/python $B/plot_bench.py
```

Each bench script takes an optional output-JSON path; the default is this directory.

## Inputs

Random disk-like points: R ~ U(0.5, 15) kpc, φ ~ U(0, 2π), z ~ U(−1, 1) kpc;
velocities are rotation plus 30 % isotropic dispersion. The potentials are
Miyamoto–Nagai (M = 1e11 M☉, a = 3, b = 0.3 kpc) and MilkyWayPotential2022, using the
galax and gala versions with the same parameters.

## Curves: `pr969-delta-vs-gala.png`

- **galax, jit**: `jax.jit(lambda q, t: staeckel_fudge_delta(pot, q, t))` on bare
  arrays, warm, `block_until_ready`. The first-call compile time is in the JSON
  (`galax_jit_compile`) and noted on the panel. It is not part of the curve.
- **galax, eager**: the same public call without `jit`, timed after its first call at
  that shape (`galax_eager_first` holds that first call).
- **gala**: `get_staeckel_fudge_delta(pot, PhaseSpacePosition, median=False)`.
- **Bottom row**: gala time / galax time. *Wall* compares wall-clock. *Per CPU-s*
  compares CPU time. gala runs on one thread, so its CPU time ≈ its wall time, while
  XLA spreads work across cores.
- Correctness is checked before timing. `max_rel_diff` holds the largest relative
  difference over finite outputs and confirms that both codes give NaN for the same
  points, i.e. where the fudge Δ² < 0.

## Curves: `pr3a-staeckel-stages-vs-gala.png`

Δ is fixed per potential at `staeckel_fudge_delta` evaluated at (R, z) = (8, 0.5) kpc,
and both codes get the same Δ. The galax stages use the private functions in
`_staeckel_core.py` and `_staeckel_limits.py`. Each stage is `jax.jit(jax.vmap(...))`,
compiled ahead of time, and timed warm. The stages are cumulative:

- **(a)** `integrals_of_motion`.
- **(b)** (a) + `make_F`, `default_phi_inf`, `find_limits`, which give the turning points.
- **(c)** (b) + `attach_tangents` + `taylor_coeffs`, value pass only.
- **(d)** `jax.value_and_grad` of `nansum` of (c) with respect to the phase-space
  positions.

**gala FULL**: `gala.dynamics.actionangle._staeckel.staeckel_fudge_c(pot, w, delta)`.
It is gala's complete C++ pipeline: actions, angles and frequencies, on one thread.

The galax curves are a **partial pipeline**. PR 3a has no quadrature, actions,
angles or frequencies yet. The grey lines are the spec's floors for the eventual
*full* galax pipeline:

- wall throughput ≥ 1.5 × gala's single-core throughput;
- throughput per CPU-second ≥ 0.5 × gala's.

A partial stage has to sit well inside these floors to leave room for the rest of
the pipeline. "Share of floor budget" in SUMMARY.md is the stage's time divided by
the largest time the floor allows (gala/1.5 for wall, 2 × gala for CPU).
