"""Test suite for the skydipole package (standard-library unittest; run: python -m unittest discover -s tests -v).

Run from the repository root.  Uses only the shipped label file and a synthetic mini catalogue.
"""
import os, shutil, subprocess, sys, tempfile, unittest
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from chirality_lib import (radec_to_unit, design_matrix, dipole_fit, fast_chi2_dipole, simulate_labels)

LABELS = os.path.join(ROOT, "data", "real_galaxy_labels.csv")


def load_positions():
    df = pd.read_csv(LABELS)
    df = df.loc[~pd.DataFrame({"r": df.ra.round(4), "d": df.dec.round(4)}).duplicated()]
    return df, radec_to_unit(df.ra.values, df.dec.values)


class TestEstimator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.n = load_positions()

    def test_sample_size(self):
        self.assertEqual(len(self.df), 36267)

    def test_paper_numbers_main_fit(self):
        """Monopole-aware and naive fits on the real GZ1 labels (Table tab:main_results)."""
        s = self.df.spin.values.astype(float)
        f = dipole_fit(self.n, s, True)
        self.assertAlmostEqual(f["m"], -0.0468, places=4)
        self.assertAlmostEqual(f["m_err"], 0.0106, places=4)
        self.assertAlmostEqual(f["amp"], 0.0155, places=4)
        self.assertAlmostEqual(f["chi2_dipole"], 1.04, places=2)
        g = dipole_fit(self.n, s, False)
        self.assertAlmostEqual(g["amp"], 0.0446, places=4)
        self.assertAlmostEqual(g["chi2_dipole"], 33.29, places=2)

    def test_injection_recovery_within_errors(self):
        """Injected (m, d) are recovered within 4 sigma; the realisation is fixed by the seed."""
        rng = np.random.default_rng(12345)
        ax = radec_to_unit(np.array([200.0]), np.array([-60.0]))[0]
        m_in, d_in = -0.05, 0.08 * ax
        s = simulate_labels(self.n, rng, m=m_in, dvec=d_in)
        f = dipole_fit(self.n, s, True)
        self.assertLess(abs(f["m"] - m_in), 4 * f["m_err"])
        sig = np.sqrt(np.diag(f["cov_d"]))
        self.assertTrue(np.all(np.abs(f["d"] - d_in) < 4 * sig))

    def test_omitted_monopole_creates_false_dipole(self):
        """Pure monopole, no dipole: the naive fit reports a large chi2, the monopole-aware fit does not."""
        rng = np.random.default_rng(777)
        s = simulate_labels(self.n, rng, m=-0.047, dvec=(0, 0, 0))
        naive = dipole_fit(self.n, s, False)["chi2_dipole"]
        aware = dipole_fit(self.n, s, True)["p_analytic"]
        self.assertGreater(naive, 20.0)       # > ~4 sigma for 3 d.o.f.
        self.assertGreater(aware, 0.01)

    def test_fast_matches_slow(self):
        rng = np.random.default_rng(1)
        X = design_matrix(self.n, True); XtX_inv = np.linalg.inv(X.T @ X)
        S = np.column_stack([simulate_labels(self.n, rng) for _ in range(6)])
        chi2, amp = fast_chi2_dipole(X, XtX_inv, S)
        for k in range(6):
            f = dipole_fit(self.n, S[:, k], True)
            self.assertAlmostEqual(chi2[k], f["chi2_dipole"], places=8)
            self.assertAlmostEqual(amp[k], f["amp"], places=10)

    def test_permutation_null_follows_chi2_3(self):
        rng = np.random.default_rng(2026)
        s = self.df.spin.values.astype(float)
        X = design_matrix(self.n, True); XtX_inv = np.linalg.inv(X.T @ X)
        S = np.column_stack([rng.permutation(s) for _ in range(1000)])
        chi2, _ = fast_chi2_dipole(X, XtX_inv, S)
        self.assertAlmostEqual(chi2.mean(), 3.0, delta=0.25)     # 1000 draws: s.e. of the mean = 0.08


class TestExtractor(unittest.TestCase):
    def test_extractor_on_mini_catalogue(self):
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(os.path.join(ROOT, "tests", "data", "mini_vizier.tsv"), os.path.join(tmp, "vizier_votable.tsv"))
            r = subprocess.run([sys.executable, os.path.join(ROOT, "Galaxy_Data_Extractor.py")],
                               cwd=tmp, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            out = pd.read_csv(os.path.join(tmp, "real_galaxy_labels.csv"))
            # rows kept at threshold 0.80: (pcS 0.90), (paS 0.90), (pcS 0.80), (paS 0.80); the 0.50/0.45 and 0.79 rows are dropped
            self.assertEqual(out.spin.tolist(), [1, -1, 1, -1])
            self.assertEqual(out.ra.tolist(), [10.0, 11.0, 13.0, 14.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
