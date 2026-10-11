"""Robustness of the monopole-aware dipole fit to label definition and vote threshold (GZ1).

Fixes relative to the earlier script:
  * parse: units/dash lines are removed by content; NO data rows are skipped
  * exact-duplicate positions (rounded to 1e-4 deg) are removed, as in the main analysis
  * objects satisfying BOTH vote criteria (ties) are excluded, not assigned +1
  * columns are labelled as the catalogue defines them: pcS/paS = standard vote fractions,
    pcSm/paSm = MONOCHROME vote fractions (bias-study data; identical to standard for
    objects without monochrome votes). They are not 'debiased' votes.
  * output is written as CSV and as LaTeX rows (no manual transcription)
Usage: python run_robustness_checks_fixed.py vizier_votable.tsv
"""
import io, sys, json
import numpy as np, pandas as pd
from scipy import stats

path = sys.argv[1] if len(sys.argv) > 1 else "vizier_votable.tsv"
def unit(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.column_stack([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])

def parse(path):
    L = open(path, encoding="utf-8", errors="ignore").read().split("\n")
    h = [i for i, l in enumerate(L) if l.startswith("_RAJ2000;")]
    if not h: raise ValueError("header '_RAJ2000;' not found")
    body = [L[h[0]]]
    for l in L[h[0] + 1:]:
        if l.startswith("<") or l.startswith("]"): break
        if l.startswith("deg;") or l.startswith("---") or not l.strip(): continue
        body.append(l)
    df = pd.read_csv(io.StringIO("\n".join(body)), sep=";", low_memory=False)
    df.columns = df.columns.str.strip()
    return df

def fit(n, s):
    X = np.column_stack([np.ones(len(n)), n]); XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ (X.T @ s); sig2 = np.mean((s - X @ b) ** 2); cov = sig2 * XtXi
    d, cd = b[1:], cov[1:, 1:]; chi2 = float(d @ np.linalg.solve(cd, d))
    return dict(m=b[0], m_err=np.sqrt(cov[0, 0]), amp=np.linalg.norm(d), chi2=chi2, p=float(stats.chi2.sf(chi2, 3)))

df = parse(path)
for c in ["_RAJ2000", "_DEJ2000", "pcS", "paS", "pcSm", "paSm", "Ntm"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
meta = {"rows_parsed": int(len(df)), "rows_with_monochrome_votes": int((df.Ntm > 0).sum())}
labels = {"Standard votes (pcS/paS)": ("pcS", "paS"), "Monochrome votes (pcSm/paSm)": ("pcSm", "paSm")}
rows = []
for name, (cw, acw) in labels.items():
    for th in (0.50, 0.60, 0.80):
        c, a = df[cw] >= th, df[acw] >= th
        ties = int((c & a).sum()); m = (c ^ a)
        ra, dec = df._RAJ2000[m].values, df._DEJ2000[m].values; s = np.where(c[m], 1.0, -1.0)
        ok = np.isfinite(ra) & np.isfinite(dec); ra, dec, s = ra[ok], dec[ok], s[ok]
        keep = ~pd.DataFrame({"r": ra.round(4), "d": dec.round(4)}).duplicated().values
        ra, dec, s = ra[keep], dec[keep], s[keep]
        f = fit(unit(ra, dec), s)
        rows.append(dict(vote=name, threshold=th, N=len(s), ties_excluded=ties, m=f["m"], m_err=f["m_err"],
                         amp=f["amp"], chi2=f["chi2"], p=f["p"], frac_clockwise=float((s > 0).mean())))
res = pd.DataFrame(rows); res.to_csv("robustness_results_fixed.csv", index=False)
with open("table4_rows.tex", "w") as fh:
    for r in rows:
        v = "Standard" if r["vote"].startswith("Standard") else "Monochrome"
        fh.write(f"{v} & ${r['threshold']:.2f}$ & ${r['N']:,}$ & ${r['m']:+.4f}\\pm{r['m_err']:.4f}$ & ${r['amp']:.4f}$ & ${r['p']:.4f}$\\\\\n".replace(",", "{,}").replace("+", "+"))
json.dump({"meta": meta, "rows": rows}, open("robustness_results_fixed.json", "w"), indent=1, default=float)
print(json.dumps(meta)); print(res.round(4).to_string(index=False))
ref = res[(res.vote.str.startswith("Standard")) & (res.threshold == 0.80)].iloc[0]
assert ref.N == 36267 and abs(ref.p - 0.7915) < 1e-3, "standard 0.80 row does not reproduce the main analysis"
print("CHECK OK: standard 0.80 row reproduces the main analysis (N=36,267, p=0.7915)")
