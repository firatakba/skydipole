"""Sample-composition checks on the GZ1 spiral sample: Main vs Additional sample (flag S), vote-count cuts.
Usage: python sample_composition_checks.py vizier_votable.tsv   (writes final_control.json)"""
import io, json, numpy as np, pandas as pd
from scipy import stats
import sys; sys.path.insert(0, ".")
from chirality_lib import radec_to_unit, dipole_fit
L = open(sys.argv[1] if len(sys.argv) > 1 else "vizier_votable.tsv", encoding="utf-8", errors="ignore").read().split("\n")
h = [i for i, l in enumerate(L) if l.startswith("_RAJ2000;")][0]
body = [L[h]] + [l for l in L[h+1:] if not (l.startswith("deg;") or l.startswith("---") or not l.strip() or l.startswith("<") or l.startswith("]"))]
df = pd.read_csv(io.StringIO("\n".join(body)), sep=";", low_memory=False)
df.columns = df.columns.str.strip(); df["S"] = df["S"].astype(str).str.strip()
for c in ["_RAJ2000","_DEJ2000","pcS","paS","N"]: df[c] = pd.to_numeric(df[c], errors="coerce")
c, a = df.pcS >= 0.8, df.paS >= 0.8
sel = df[c ^ a].copy(); sel["s"] = np.where(c[c ^ a], 1.0, -1.0)
keep = ~pd.DataFrame({"r": sel._RAJ2000.round(4), "d": sel._DEJ2000.round(4)}).duplicated().values
sel = sel[keep]
out = {"N_sample": len(sel), "S_counts_catalogue": df.S.value_counts().to_dict(), "S_counts_sample": sel.S.value_counts().to_dict(),
       "votes_N_sample": {"min": int(sel.N.min()), "p1": float(sel.N.quantile(.01)), "p5": float(sel.N.quantile(.05)), "median": float(sel.N.median()),
                          "frac_N_lt_20": float((sel.N < 20).mean()), "frac_N_lt_10": float((sel.N < 10).mean())}}
def fit(sub):
    n = radec_to_unit(sub._RAJ2000.values, sub._DEJ2000.values); f = dipole_fit(n, sub.s.values, True)
    return {"N": len(sub), "m": f["m"], "m_err": f["m_err"], "amp": f["amp"], "chi2": f["chi2_dipole"], "p": f["p_analytic"], "frac_cw": float((sub.s > 0).mean())}
out["fit_all"] = fit(sel)
out["fit_S=M"] = fit(sel[sel.S == "M"])
if (sel.S == "A").sum() > 500: out["fit_S=A"] = fit(sel[sel.S == "A"])
for k in (20, 30): out[f"fit_N>={k}"] = fit(sel[sel.N >= k])
out["fit_M_and_N>=20"] = fit(sel[(sel.S == "M") & (sel.N >= 20)])
json.dump(out, open("final_control.json", "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float))
