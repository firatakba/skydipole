#!/usr/bin/env python3
"""Wall-clock scaling of the main steps of the monopole-aware dipole estimator.

Produces the numbers of the "Performance" table (tab:timing) of the paper.

Steps timed (all single-threaded; thread env vars are set BEFORE numpy is imported):
  fit       design matrix + OLS monopole+dipole fit            (chirality_lib.dipole_fit)
  perm      n_perm label-permutation chi2 nulls, in blocks      (chirality_lib.fast_chi2_dipole)
  boot      n_boot bootstrap refits                              (chirality_lib.dipole_fit)
  split     n_split split-half consistency tests (2 fits each)   (chirality_lib.dipole_fit)
  inject    len(AMPS) x n_real injection-recovery refits         (simulate_labels + dipole_fit)
  reference perm + boot + split: the "reference run" quoted in the Program Summary

Catalogue: the real GZ1 positions (ra, dec) of the labels file, resampled with replacement
to size N (for N equal to the real sample size the real positions are used as they are).
Labels are synthetic (+/-1 with probability 1/2), so only the cost is measured, not any
astrophysical result.

Usage
  python benchmark_scaling.py --labels real_galaxy_labels.csv --out benchmark_scaling_results.json
  python benchmark_scaling.py --labels real_galaxy_labels.csv --sizes 10000 36267 100000   # quick
Run on the machine whose hardware you quote in the paper; copy the printed LaTeX rows.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"          # single thread, as stated in the paper

import argparse
import hashlib
import json
import platform
import sys
import time

import numpy as np
import pandas as pd

from chirality_lib import (radec_to_unit, design_matrix, dipole_fit,
                           fast_chi2_dipole, simulate_labels)

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--labels", default="data/real_galaxy_labels.csv", help="CSV with ra,dec,spin")
p.add_argument("--sizes", type=int, nargs="+", default=[10_000, 36_267, 100_000, 1_000_000])
p.add_argument("--nperm", type=int, default=10_000)
p.add_argument("--nboot", type=int, default=300)
p.add_argument("--nsplit", type=int, default=50)
p.add_argument("--amps", type=float, nargs="+", default=[0.02, 0.03, 0.05, 0.08])
p.add_argument("--nreal", type=int, default=200, help="realisations per amplitude")
p.add_argument("--block", type=int, default=250, help="permutation block size K")
p.add_argument("--block-large", type=int, default=50,
               help="block size used when N*K*8 bytes*3 would exceed --mem-gb")
p.add_argument("--mem-gb", type=float, default=4.0, help="working-memory budget for a block")
p.add_argument("--repeats", type=int, default=5, help="repeats for the (cheap) fit timing")
p.add_argument("--seed", type=int, default=20261021)
p.add_argument("--out", default="benchmark_scaling_results.json")
a = p.parse_args()

t_all = time.perf_counter()
raw = open(a.labels, "rb").read()
sha = hashlib.sha256(raw).hexdigest()
df = pd.read_csv(a.labels).drop_duplicates(["ra", "dec"])   # as in the paper: duplicate positions removed
ra0, dec0 = df["ra"].to_numpy(float), df["dec"].to_numpy(float)
N_real = len(df)
rng = np.random.default_rng(a.seed)
print(f"input {a.labels}: {N_real} rows, sha256 {sha[:12]}...  numpy {np.__version__}, "
      f"python {platform.python_version()}, machine {platform.machine()}")

# warm-up (BLAS/einsum initialisation, page faults) so the first row is not penalised
_n = radec_to_unit(ra0[:2000], dec0[:2000]); _s = simulate_labels(_n, rng)
dipole_fit(_n, _s); fast_chi2_dipole(design_matrix(_n), np.linalg.inv(design_matrix(_n).T @ design_matrix(_n)),
                                     np.column_stack([_s] * 8))

rows = []
for N in a.sizes:
    if N == N_real:
        ra, dec = ra0, dec0
    else:
        i = rng.integers(0, N_real, N)
        ra, dec = ra0[i], dec0[i]
    n = radec_to_unit(ra, dec)
    s = simulate_labels(n, rng)                       # null labels, m = 0, d = 0

    # --- fit: design matrix + inversion + OLS + chi2 (what one call to dipole_fit costs)
    tf = []
    for _ in range(a.repeats):
        t0 = time.perf_counter(); dipole_fit(n, s, True); tf.append(time.perf_counter() - t0)
    t_fit = float(np.median(tf))

    X = design_matrix(n, True)
    XtX_inv = np.linalg.inv(X.T @ X)

    # --- permutation null (nperm label permutations, blocks of K columns)
    K = a.block
    if N * K * 8 * 3 / 1e9 > a.mem_gb:                # S, X@B and residual block are alive together
        K = max(10, min(a.block_large, int(a.mem_gb * 1e9 / (N * 8 * 3))))
    t0 = time.perf_counter()
    done = 0
    while done < a.nperm:
        k = min(K, a.nperm - done)
        S = np.column_stack([rng.permutation(s) for _ in range(k)])
        fast_chi2_dipole(X, XtX_inv, S)
        done += k
    t_perm = time.perf_counter() - t0
    del S

    # --- bootstrap refits
    t0 = time.perf_counter()
    for _ in range(a.nboot):
        j = rng.integers(0, N, N)
        dipole_fit(n[j], s[j], True)
    t_boot = time.perf_counter() - t0

    # --- split-half consistency (same procedure as option2_real_analysis.py)
    t0 = time.perf_counter()
    for _ in range(a.nsplit):
        pm = rng.permutation(N); h1, h2 = pm[: N // 2], pm[N // 2:]
        f1, f2 = dipole_fit(n[h1], s[h1], True), dipole_fit(n[h2], s[h2], True)
        dd = f1["d"] - f2["d"]; dd @ np.linalg.solve(f1["cov_d"] + f2["cov_d"], dd)
    t_split = time.perf_counter() - t0

    # --- injection-recovery grid
    ax = radec_to_unit(np.array([200.0]), np.array([-60.0]))[0]
    t0 = time.perf_counter()
    for A in a.amps:
        for _ in range(a.nreal):
            si = simulate_labels(n, rng, m=0.0, dvec=A * ax)
            dipole_fit(n, si, True, XtX_inv)
    t_inj = time.perf_counter() - t0

    row = {"N": int(N), "fit_ms": 1e3 * t_fit, "perm_s": t_perm, "perm_block": int(K),
           "n_perm": a.nperm, "boot_s": t_boot, "n_boot": a.nboot, "split_s": t_split, "n_split": a.nsplit,
           "reference_run_s": t_perm + t_boot + t_split, "inject_s": t_inj,
           "inject_grid": f"{len(a.amps)} amplitudes x {a.nreal}"}
    rows.append(row)
    print(f"N={N:>9,d}  fit {row['fit_ms']:8.2f} ms | {a.nperm} perm {t_perm:8.2f} s (K={K}) | "
          f"{a.nboot} boot {t_boot:7.2f} s | {a.nsplit} split {t_split:6.2f} s | "
          f"injection {t_inj:7.2f} s | reference run {row['reference_run_s']:7.2f} s", flush=True)

out = {"seed": a.seed, "labels_sha256": sha, "n_real_rows": N_real,
       "environment": {"python": platform.python_version(), "numpy": np.__version__,
                       "machine": platform.machine(), "system": platform.platform(),
                       "cpu_count_logical": os.cpu_count(),
                       "threads_env": os.environ.get("OMP_NUM_THREADS")},
       "args": vars(a), "rows": rows, "total_runtime_s": time.perf_counter() - t_all}
with open(a.out, "w") as f:
    json.dump(out, f, indent=1)

print("\nLaTeX rows (N & fit (ms) & perm (s) & boot (s) & injection (s)):")
for r in rows:
    Ns = f"{r['N']:,}".replace(",", "{,}")
    print(f"${Ns}$ & ${r['fit_ms']:.1f}$ & ${r['perm_s']:.1f}$ & ${r['boot_s']:.1f}$ & ${r['inject_s']:.1f}$\\\\")
print(f"\nwrote {a.out}; total {out['total_runtime_s']:.0f} s")
