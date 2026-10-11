"""Shared utilities for the galaxy-spin dipole analysis.

Model: P(spin = +1 | direction n) = 0.5 * (1 + m + d . n)
  m = global S/Z imbalance (monopole; the signature of classification bias)
  d = dipole vector (|d| = amplitude, direction = axis)
Estimator: ordinary least squares of s_i = +/-1 on [1, nx, ny, nz].
Unbiased for partial sky coverage because the full 4x4 normal matrix is inverted.
"""
import io
import numpy as np
import pandas as pd
from scipy import stats


def radec_to_unit(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.column_stack([np.cos(dec) * np.cos(ra),
                            np.cos(dec) * np.sin(ra),
                            np.sin(dec)])


def unit_to_radec(v):
    v = np.asarray(v, float)
    return (np.degrees(np.arctan2(v[1], v[0])) % 360.0,
            np.degrees(np.arcsin(v[2] / np.linalg.norm(v))))


def load_vizier_votable_csv(path, expected_cols=("RA_ICRS", "DE_ICRS")):
    """Parse a VizieR VOTable/CSV (';' separated, 3 header lines). Fails loudly."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().split("\n")
    idx = [i for i, l in enumerate(lines) if l.startswith(expected_cols[0] + ";")]
    if not idx:
        raise ValueError(f"Header starting with {expected_cols[0]} not found in {path}")
    i = idx[0]
    body = []
    for l in lines[i + 3:]:
        if l.startswith("<") or l.startswith("]") or not l.strip():
            break
        body.append(l)
    df = pd.read_csv(io.StringIO("\n".join([lines[i]] + body)), sep=";")
    for c in expected_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column {c}; got {list(df.columns)}")
    return df


def footprint_report(ra, dec, axis_radec=None):
    """Plain-language summary of the sky actually covered."""
    rep = {
        "N": int(len(ra)),
        "ra_range_deg": [float(np.min(ra)), float(np.max(ra))],
        "dec_range_deg": [float(np.min(dec)), float(np.max(dec))],
    }
    n = radec_to_unit(ra, dec)
    # Fraction of the sphere covered: count occupied ~4-deg cells in (ra, sin dec)
    ra_i = np.floor(ra / 4.0).astype(int)
    sd_i = np.floor((np.sin(np.radians(dec)) + 1) / 2 * 45).astype(int)  # equal-area rows
    occupied = len(set(zip(ra_i.tolist(), sd_i.tolist())))
    rep["sky_fraction_occupied_approx"] = occupied / (90 * 45)
    if axis_radec is not None:
        ax = radec_to_unit(np.array([axis_radec[0]]), np.array([axis_radec[1]]))[0]
        th = np.degrees(np.arccos(np.clip(n @ ax, -1, 1)))
        rep["theta_to_axis_deg_min_max"] = [float(th.min()), float(th.max())]
    return rep


def design_matrix(n, monopole=True):
    return np.column_stack([np.ones(len(n)), n]) if monopole else n


def dipole_fit(n, s, monopole=True, XtX_inv=None):
    """OLS dipole fit. Returns dict with m, d vector, amplitude, axis, cov, chi2 (3 dof)."""
    X = design_matrix(n, monopole)
    if XtX_inv is None:
        XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ s)
    resid = s - X @ beta
    sigma2 = np.mean(resid ** 2)
    cov = sigma2 * XtX_inv
    k = 1 if monopole else 0
    d, cov_d = beta[k:], cov[k:, k:]
    chi2_dip = float(d @ np.linalg.solve(cov_d, d))
    ra, dec = unit_to_radec(d)
    return {"m": float(beta[0]) if monopole else 0.0,
            "m_err": float(np.sqrt(cov[0, 0])) if monopole else 0.0,
            "d": d, "cov_d": cov_d, "amp": float(np.linalg.norm(d)),
            "ra": ra, "dec": dec, "chi2_dipole": chi2_dip,
            "p_analytic": float(stats.chi2.sf(chi2_dip, 3))}


def fast_chi2_dipole(X, XtX_inv, S, monopole=True):
    """Vectorised dipole chi2 and amplitude for many label vectors S (N x K)."""
    B = XtX_inv @ (X.T @ S)                     # (p x K)
    k = 1 if monopole else 0
    d = B[k:, :]
    R = S - X @ B
    sig2 = np.mean(R ** 2, axis=0)
    cov_d_unit = XtX_inv[k:, k:]
    inv = np.linalg.inv(cov_d_unit)
    chi2 = np.einsum("ik,ij,jk->k", d, inv, d) / sig2
    return chi2, np.linalg.norm(d, axis=0)


def sigma_from_p(p):
    return float(stats.norm.isf(p / 2.0)) if p > 0 else float("inf")


def simulate_labels(n, rng, m=0.0, dvec=(0, 0, 0)):
    p = np.clip(0.5 * (1 + m + n @ np.asarray(dvec, float)), 0, 1)
    return np.where(rng.random(len(n)) < p, 1.0, -1.0)
