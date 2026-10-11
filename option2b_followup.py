"""Follow-up on REAL Galaxy Zoo 1 labels: (1) the paper's Eq.4 pipeline on real data,
(2) upper limits, (3) sensitivity (injection into the real footprint - SIMULATION)."""
import json, sys, numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import least_squares
from chirality_lib import *
path = sys.argv[1] if len(sys.argv) > 1 else "real_galaxy_labels.csv"
rng = np.random.default_rng(2026)
df = pd.read_csv(path)
df = df.loc[~pd.DataFrame({"r": df.ra.round(4), "d": df.dec.round(4)}).duplicated()]
s = df.spin.values.astype(float); n = radec_to_unit(df.ra.values, df.dec.values)
ax = radec_to_unit(np.array([200.0]), np.array([-60.0]))[0]
th = np.arccos(np.clip(n @ ax, -1, 1)); N = len(s); R = {"N": N}

# ---- (1) paper's Eq.4 pipeline on the real labels
def eq4(t, a, C0): return C0 * a * np.sin(t) / (1 + a**2 * np.cos(t)**2)
M = 24; edges = np.quantile(th, np.linspace(0, 1, M + 1))
k = np.clip(np.searchsorted(edges, th, side="right") - 1, 0, M - 1)
Nk = np.bincount(k, minlength=M); A = np.bincount(k, weights=s, minlength=M) / Nk
Tk = np.bincount(k, weights=th, minlength=M) / Nk; sg = 1 / np.sqrt(Nk)
def fit(bounds):
    best = None
    for a0 in (0.2, 0.5, 0.9):
        r = least_squares(lambda p: (eq4(Tk, *p) - A) / sg, [a0, np.clip(0.02, bounds[0][1], bounds[1][1])], bounds=bounds)
        if best is None or r.cost < best.cost: best = r
    return best
chi2_h0 = float(np.sum((A / sg) ** 2))
b = fit(([0, 0], [1, 1])); u = fit(([0, -1], [1, 1]))
const_m = float(np.sum(A / sg**2) / np.sum(1 / sg**2)); chi2_const = float(np.sum(((A - const_m) / sg) ** 2))
R["paper_pipeline_on_real_data"] = {
    "theta_range_rad": [float(Tk.min()), float(Tk.max())], "mean_A_all_bins": float(s.mean()),
    "A_bins_min_max": [float(A.min()), float(A.max())],
    "chi2_H0_A=0": chi2_h0, "p_H0": float(stats.chi2.sf(chi2_h0, M)),
    "eq4_bounded_C0>=0": {"a": float(b.x[0]), "C0": float(b.x[1]), "chi2": 2 * b.cost, "dof": M - 2},
    "eq4_C0_allowed_negative": {"a": float(u.x[0]), "C0": float(u.x[1]), "chi2": 2 * u.cost, "dof": M - 2},
    "constant_offset_model": {"m": const_m, "chi2": chi2_const, "dof": M - 1, "p": float(stats.chi2.sf(chi2_const, M - 1))}}

# ---- (2) upper limits from the monopole-marginalised fit
X = design_matrix(n, True); XtX_inv = np.linalg.inv(X.T @ X); f = dipole_fit(n, s, True, XtX_inv)
proj, sp = float(f["d"] @ ax), float(np.sqrt(ax @ f["cov_d"] @ ax))
R["limits"] = {"d_along_paper_axis": proj, "sigma": sp,
               "95pct_interval_along_axis": [proj - 1.96 * sp, proj + 1.96 * sp],
               "95pct_one_sided_upper_limit_positive_direction": proj + 1.645 * sp,
               "paper_implied_asymmetry_scale_C0": 0.032,
               "positive_0.032_excluded_at_sigma": (0.032 - proj) / sp,
               "monopole": [f["m"], f["m_err"], f["m"] / f["m_err"]]}

# ---- (3) sensitivity on the real footprint (SIMULATION ONLY)
c3 = float(stats.chi2.isf(2 * stats.norm.sf(3), 3)); rows = []
dirs = {"paper_axis": ax, "GZ_north_pole_RA192_Dec27": radec_to_unit(np.array([192.85]), np.array([27.13]))[0]}
for name, dv in dirs.items():
    for amp in (0.02, 0.03, 0.05, 0.08):
        det = np.mean([dipole_fit(n, simulate_labels(n, rng, f["m"], amp * dv), True, XtX_inv)["chi2_dipole"] > c3
                       for _ in range(200)])
        rows.append({"axis": name, "amp": amp, "frac_detected_3sigma": float(det)})
R["sensitivity_SIMULATION"] = rows
json.dump(R, open("followup_real_results.json", "w"), indent=2, default=float)
print(json.dumps(R, indent=1, default=float))
