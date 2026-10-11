"""OPTION 2 - Dipole analysis of REAL galaxy-spin labels.

Usage:
  python option2_real_analysis.py --labels labels.csv --ra RA --dec DEC --spin SPIN \
         [--z Z] [--ebv EBV] [--nperm 10000] [--axis-ra 200 --axis-dec -60] [--out run1]

Input: one row per galaxy with a *measured* handedness (from human or validated-algorithm
classification). Accepted spin codes: +1/-1, S/Z, CW/ACW (case-insens.). Anything else
(0, NaN, 'U', ...) is dropped and counted. This script NEVER invents labels, redshifts or
dust values: if an optional column is not supplied the corresponding test is skipped
and reported as SKIPPED, not faked.
"""
import argparse, hashlib, json, sys, datetime, platform
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chirality_lib import *

p = argparse.ArgumentParser()
p.add_argument("--labels", required=True); p.add_argument("--ra", required=True)
p.add_argument("--dec", required=True); p.add_argument("--spin", required=True)
p.add_argument("--z"); p.add_argument("--ebv")
p.add_argument("--nperm", type=int, default=10000); p.add_argument("--nboot", type=int, default=300)
p.add_argument("--axis-ra", type=float, default=200.0); p.add_argument("--axis-dec", type=float, default=-60.0)
p.add_argument("--seed", type=int, default=12345); p.add_argument("--out", default="option2")
a = p.parse_args()
rng = np.random.default_rng(a.seed)
def die(msg): print("FATAL:", msg); sys.exit(2)

# ----------------------------------------------------------------- load + validate
try: df = pd.read_csv(a.labels)
except Exception as e: die(f"cannot read {a.labels}: {e}")
for c in [a.ra, a.dec, a.spin] + [x for x in (a.z, a.ebv) if x]:
    if c not in df.columns: die(f"column '{c}' not in file; columns are {list(df.columns)}")
sha = hashlib.sha256(open(a.labels, "rb").read()).hexdigest()
log = {"input": a.labels, "sha256": sha, "n_rows_raw": int(len(df)), "seed": a.seed,
       "utc": datetime.datetime.utcnow().isoformat(), "python": platform.python_version(),
       "numpy": np.__version__}
if "TEST_synthetic" in a.labels:
    print("#" * 70 + "\n# INPUT FILE IS SYNTHETIC (pipeline self-test). NOT A MEASUREMENT.\n" + "#" * 70)
    log["WARNING"] = "synthetic input - pipeline self-test only"

code = {"S": 1, "CW": 1, "+1": 1, "1": 1, "Z": -1, "ACW": -1, "CCW": -1, "-1": -1}
sp = df[a.spin].astype(str).str.strip().str.upper().map(code)
df = df.assign(_s=sp)
log["n_dropped_unrecognised_spin"] = int(df["_s"].isna().sum())
df = df.dropna(subset=["_s", a.ra, a.dec])
df = df[(df[a.dec].between(-90, 90)) & (df[a.ra].between(0, 360))]
before = len(df)
df = df.loc[~pd.DataFrame({"r": df[a.ra].round(4), "d": df[a.dec].round(4)}).duplicated()]
log["n_duplicates_removed"] = int(before - len(df))
if len(df) < 1000: die(f"only {len(df)} usable labelled galaxies - refusing to fit")
s = df["_s"].values.astype(float)
n = radec_to_unit(df[a.ra].values, df[a.dec].values)
ax = radec_to_unit(np.array([a.axis_ra]), np.array([a.axis_dec]))[0]
log["N_used"] = int(len(s)); log["frac_S"] = float((s > 0).mean())
log["footprint"] = footprint_report(df[a.ra].values, df[a.dec].values, (a.axis_ra, a.axis_dec))
print(json.dumps(log, indent=1))

X = design_matrix(n, True); XtX_inv = np.linalg.inv(X.T @ X)
ev = np.linalg.eigvalsh(XtX_inv[1:, 1:])
log["dipole_sigma_best_worst_direction"] = [float(np.sqrt(ev.min())), float(np.sqrt(ev.max()))]
if np.sqrt(ev.max()) > 5 * np.sqrt(3.0 / len(s)):
    print("WARNING: poor sky coverage - some dipole directions are barely constrained "
          f"(sigma {np.sqrt(ev.max()):.4f} vs full-sky {np.sqrt(3.0/len(s)):.4f}).")

# ----------------------------------------------------------------- main fit
R = {"log": log}
fit = dipole_fit(n, s, True, XtX_inv); fit0 = dipole_fit(n, s, False)
R["fit_dipole_plus_monopole"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in fit.items() if k != "cov_d"}
R["fit_dipole_only"] = {"amp": fit0["amp"], "ra": fit0["ra"], "dec": fit0["dec"], "p_analytic": fit0["p_analytic"]}
print(f"\nMonopole m = {fit['m']:+.5f} +/- {fit['m_err']:.5f}  (non-zero => global labelling bias)")
print(f"Dipole |d| = {fit['amp']:.5f}  axis (RA,Dec) = ({fit['ra']:.1f}, {fit['dec']:.1f})  "
      f"chi2(3dof) = {fit['chi2_dipole']:.2f}")

# ----------------------------------------------------------------- permutation null
# Statistic = chi2 of the best-fit free-axis dipole => the search over all axes is inside
# the statistic, so the look-elsewhere effect is handled automatically.
null = []
for _ in range(int(np.ceil(a.nperm / 250))):
    k = 250
    S = np.column_stack([rng.permutation(s) for _ in range(k)])
    null.append(fast_chi2_dipole(X, XtX_inv, S)[0])
null = np.concatenate(null)[: a.nperm]
cnt = int(np.sum(null >= fit["chi2_dipole"]))
p_perm = (1 + cnt) / (1 + len(null))
R["permutation"] = {"n": int(len(null)), "n_ge_observed": cnt, "p_perm": p_perm,
                    "p_floor": 1.0 / (1 + len(null)), "sigma_equiv": sigma_from_p(p_perm),
                    "null_chi2_mean": float(null.mean())}
print(f"Permutation p = {p_perm:.2e} ({cnt}/{len(null)} >= observed); floor {1/(1+len(null)):.1e}")

# ----------------------------------------------------------------- pre-specified axis (1 dof)
proj = float(fit["d"] @ ax); sig_proj = float(np.sqrt(ax @ fit["cov_d"] @ ax))
R["paper_axis_projection"] = {"d_along_axis": proj, "sigma": sig_proj, "z_score": proj / sig_proj,
    "note": "axis was chosen post hoc in the original paper; only NEW data test it fairly"}
print(f"Projection on paper axis: {proj:+.5f} +/- {sig_proj:.5f}  ({proj/sig_proj:+.2f} sigma, 1 dof)")

# ----------------------------------------------------------------- bootstrap axis uncertainty
bs = []
for _ in range(a.nboot):
    i = rng.integers(0, len(s), len(s)); bs.append(dipole_fit(n[i], s[i], True)["d"])
bs = np.array(bs); u = bs / np.linalg.norm(bs, axis=1, keepdims=True)
m_ax = u.mean(0); m_ax /= np.linalg.norm(m_ax)
R["bootstrap"] = {"amp_mean": float(np.linalg.norm(bs, axis=1).mean()),
                  "amp_16_84": np.percentile(np.linalg.norm(bs, axis=1), [16, 84]).tolist(),
                  "axis_68pct_radius_deg": float(np.percentile(np.degrees(np.arccos(np.clip(u @ m_ax, -1, 1))), 68))}

# ----------------------------------------------------------------- split-half consistency (WITH errors)
cons = []
for _ in range(50):
    perm = rng.permutation(len(s)); h1, h2 = perm[: len(s) // 2], perm[len(s) // 2:]
    f1, f2 = dipole_fit(n[h1], s[h1], True), dipole_fit(n[h2], s[h2], True)
    dd = f1["d"] - f2["d"]; cons.append(dd @ np.linalg.solve(f1["cov_d"] + f2["cov_d"], dd))
R["split_half"] = {"chi2_mean_expected_3": float(np.mean(cons)), "n_splits": 50,
                   "note": "halves share the same sky pattern; values >>3 indicate internal inconsistency"}

# ----------------------------------------------------------------- redshift slices
if a.z:
    zz = df[a.z].values.astype(float); ok = np.isfinite(zz)
    edges = np.quantile(zz[ok], np.linspace(0, 1, 5)); rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ok & (zz >= lo) & (zz <= hi); f = dipole_fit(n[m], s[m], True)
        rows.append({"z_lo": float(lo), "z_hi": float(hi), "N": int(m.sum()), "amp": f["amp"],
                     "d": f["d"].tolist(), "d_err_xyz": np.sqrt(np.diag(f["cov_d"])).tolist(),
                     "chi2_dipole": f["chi2_dipole"], "p": f["p_analytic"]})
    R["redshift_slices"] = rows
else: R["redshift_slices"] = "SKIPPED (no --z column). Use spectroscopic z, not photo-z."

# ----------------------------------------------------------------- dust (real E(B-V) required)
if a.ebv:
    e = df[a.ebv].values.astype(float); ok = np.isfinite(e)
    ez = (e - e[ok].mean()) / e[ok].std(); Xe = np.column_stack([X, ez])
    beta = np.linalg.solve(Xe[ok].T @ Xe[ok], Xe[ok].T @ s[ok])
    res = s[ok] - Xe[ok] @ beta; cov = np.mean(res ** 2) * np.linalg.inv(Xe[ok].T @ Xe[ok])
    f_noe = dipole_fit(n[ok], s[ok], True)
    R["dust"] = {"ebv_coeff": float(beta[-1]), "ebv_coeff_err": float(np.sqrt(cov[-1, -1])),
                 "dipole_amp_without": f_noe["amp"], "dipole_amp_with_ebv_term": float(np.linalg.norm(beta[1:4])),
                 "note": "dipole stable when E(B-V) regressed out => not dust-driven"}
else: R["dust"] = "SKIPPED (no --ebv column). Take E(B-V) from SFD98/Planck at each position."

json.dump(R, open(f"{a.out}_results.json", "w"), indent=2, default=float)

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
axs[0].hist(null, bins=60, density=True, alpha=.6, label="permutation null")
xs = np.linspace(0, max(null.max(), fit["chi2_dipole"]) * 1.05, 300)
axs[0].plot(xs, stats.chi2.pdf(xs, 3), "k--", label="chi2(3)")
axs[0].axvline(fit["chi2_dipole"], c="r", label="observed"); axs[0].legend()
axs[0].set(xlabel="dipole chi2 (3 dof)", title="Observed vs null")
th = np.arccos(np.clip(n @ ax, -1, 1)); edges = np.quantile(th, np.linspace(0, 1, 13))
k = np.clip(np.searchsorted(edges, th, side="right") - 1, 0, 11)
Nk = np.bincount(k, minlength=12); Ak = np.bincount(k, weights=s, minlength=12) / Nk
axs[1].errorbar(np.bincount(k, weights=th, minlength=12) / Nk, Ak, yerr=1 / np.sqrt(Nk), fmt="o")
axs[1].axhline(0, c="gray", ls="--"); axs[1].set(xlabel="theta to paper axis [rad]", ylabel="mean spin",
    title=f"Binned view (N={len(s):,}); model-free")
plt.tight_layout(); plt.savefig(f"{a.out}_summary.png", dpi=130)
print(f"\nWrote {a.out}_results.json and {a.out}_summary.png")
