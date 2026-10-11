"""REAL GZ1 labels: (a) exact naive-model numbers, (b) projection on published axes,
(c) check that the monopole-induced false dipole points away from the footprint."""
import json, sys, numpy as np, pandas as pd
from scipy import stats
from chirality_lib import *
path = sys.argv[1] if len(sys.argv) > 1 else "real_galaxy_labels.csv"
df = pd.read_csv(path)
df = df.loc[~pd.DataFrame({"r": df.ra.round(4), "d": df.dec.round(4)}).duplicated()]
s = df.spin.values.astype(float); n = radec_to_unit(df.ra.values, df.dec.values); N = len(s)
R = {"N": N}
Rg = np.array([[-0.0548755604,-0.8734370902,-0.4838350155],[0.4941094279,-0.4448296300,0.7469822445],
               [-0.8676661490,-0.1980763734,0.4559837762]])
def gal(l, b):
    l, b = np.radians(l), np.radians(b)
    return Rg.T @ np.array([np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)])
axes = {"authors_earlier_manuscript (200,-60) [post hoc]": radec_to_unit(np.array([200.]), np.array([-60.]))[0],
        "Longo2011 (l,b)=(52,68.5)": gal(52, 68.5),
        "Shamir2012 (RA,Dec)=(132,+32)": radec_to_unit(np.array([132.]), np.array([32.]))[0],
        "CMB kinematic dipole (l,b)=(264.02,48.25)": gal(264.021, 48.253),
        "Axis of Evil (l,b)~(260,60) [secondary source]": gal(260, 60)}
X = design_matrix(n, True); XtX_inv = np.linalg.inv(X.T @ X)
f = dipole_fit(n, s, True, XtX_inv); f0 = dipole_fit(n, s, False)
R["unbiased"] = {"amp": f["amp"], "chi2": f["chi2_dipole"], "p": f["p_analytic"], "m": f["m"], "m_err": f["m_err"]}
R["naive"] = {"amp": f0["amp"], "chi2": f0["chi2_dipole"], "p": f0["p_analytic"],
              "sigma_two_sided": float(stats.norm.isf(f0["p_analytic"]/2)), "ra": f0["ra"], "dec": f0["dec"]}
mean_s = s.mean(); R["simple_mean_spin"] = [float(mean_s), float(1/np.sqrt(N)), float(mean_s*np.sqrt(N))]
# naive dipole direction vs. anti-centroid of the footprint
nbar = n.mean(0); nbar_u = nbar/np.linalg.norm(nbar)
dn = f0["d"]/np.linalg.norm(f0["d"])
R["footprint_centroid_radec"] = unit_to_radec(nbar_u)
R["angle_naive_dipole_vs_anticentroid_deg"] = float(np.degrees(np.arccos(np.clip(dn @ -nbar_u, -1, 1))))
rows = []
for name, a in axes.items():
    pr = float(f["d"] @ a); sg = float(np.sqrt(a @ f["cov_d"] @ a)); z = pr/sg
    rows.append({"axis": name, "projection": pr, "sigma": sg, "z": z, "p_two_sided": float(2*stats.norm.sf(abs(z))),
                 "95pct_interval": [pr-1.96*sg, pr+1.96*sg]})
R["axis_projections"] = rows; R["n_axes_tested"] = len(rows)
R["bonferroni_p_min"] = min(1.0, min(r["p_two_sided"] for r in rows)*len(rows))
json.dump(R, open("literature_axes_results.json", "w"), indent=2, default=float)
print(json.dumps(R, indent=1, default=float))
