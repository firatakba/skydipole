# skydipole

Monopole-aware dipole estimator for binary-labelled sky catalogues, with an application to galaxy-spin
handedness in Galaxy Zoo 1 (GZ1).  Accompanies the *Computer Physics Communications* (CPiP) article
"skydipole: a monopole-aware dipole estimator for binary-labelled sky catalogues, with an application to galaxy spin handedness in Galaxy Zoo 1".

* Model: `E[s | n] = m + d·n`, fitted by ordinary least squares with the full 4×4 normal matrix.
* Free-axis χ²(3 d.o.f.) calibrated by label permutation; fixed-axis projections; bootstrap, split halves; injection–recovery on the real positions.
* Licence: MIT (see `LICENSE`).  Python ≥ 3.10; NumPy, SciPy, pandas, Matplotlib (`requirements.txt`).

## Contents

| File | Purpose (article Table `tab:files`) |
|---|---|
| `chirality_lib.py` | shared library: unit vectors, design matrix, OLS fit, vectorised permutation χ², label simulation |
| `Galaxy_Data_Extractor.py` | builds the ±1 label file from a VizieR GZ1 export (vote-fraction threshold 0.80) |
| `option2_real_analysis.py` | main fit, permutation null, bootstrap, split halves, optional z and E(B−V) tests |
| `option2b_followup.py` | binned diagnostic (Appendix A), fixed-axis limits, injection–recovery |
| `option2c_literature_axes.py` | projections on published axes, naive-model numbers, Bonferroni correction |
| `run_robustness_checks_fixed.py` | threshold and label-definition robustness (Table `tab:robustness`) |
| `sample_composition_checks.py` | sub-sample fits (Table `tab:subsets`) |
| `benchmark_scaling.py` | timing versus N and n_perm (Table `tab:timing`) |
| `data/real_galaxy_labels.csv` | GZ1 handedness labels used in the article (36,268 rows; 36,267 after removal of one duplicate position) |
| `results/` | reference outputs (see below) |
| `figures/fig_footprint.png` | footprint figure of the article |
| `tests/` | test suite and a synthetic mini catalogue for the extractor |

## Install and test

```
python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v                # 7 tests, about 2 s
```

The tests check, among other things, that the shipped label file reproduces the main numbers of the article
(m = −0.0468 ± 0.0106, |d| = 0.0155, χ²_dip = 1.04; naive fit χ²_dip = 33.29), that injected (m, d) are recovered,
that omitting the monopole creates a false dipole, and that the permutation null has mean 3.

## Run order of the example application

All commands are run from the repository root; outputs are written to the current directory
(reference copies are in `results/`).  Seeds are fixed inside each script.

| Step | Command | Output |
|---|---|---|
| 1. Labels (optional; needs the VizieR export, see below) | `python Galaxy_Data_Extractor.py` | `real_galaxy_labels.csv` |
| 2. Main fit, permutation null, bootstrap, split halves | `python option2_real_analysis.py --labels data/real_galaxy_labels.csv --ra ra --dec dec --spin spin --nperm 10000 --out run_real_allsky` | `run_real_allsky_results.json`, `run_real_allsky_summary.png` |
| 3. Appendix A diagnostic, limits, sensitivity (injection, simulation) | `python option2b_followup.py data/real_galaxy_labels.csv` | `followup_real_results.json` |
| 4. Published-axis projections, naive-model numbers | `python option2c_literature_axes.py data/real_galaxy_labels.csv` | `literature_axes_results.json` |
| 5. Threshold / label-definition robustness | `python run_robustness_checks_fixed.py vizier_votable.tsv` | `robustness_results_fixed.csv`, `table4_rows.tex` |
| 6. Sample composition | `python sample_composition_checks.py vizier_votable.tsv` | `final_control.json` |
| 7. Timing benchmark (single thread; about 4 min) | `python benchmark_scaling.py --labels data/real_galaxy_labels.csv --out benchmark_scaling_results.json` | `benchmark_scaling_results.json` |

Steps 2–4 and 7 need only the shipped label file.  Steps 1, 5 and 6 need the GZ1 table from VizieR
(`J/MNRAS/410/166`, all columns, no row limit, 893,212 rows, saved as `vizier_votable.tsv`); it is not redistributed here.

Step 2 on the shipped labels prints |d| = 0.01547, axis (RA, Dec) = (165.9°, +62.7°), χ²(3 d.o.f.) = 1.04 and
p_perm = 0.7923 (7,923 of 10⁴).  A typical call on your own catalogue is

```
python option2_real_analysis.py --labels labels.csv --ra ra --dec dec --spin spin --nperm 10000 --out run1
```

(label codes accepted: ±1, S/Z, CW/ACW; other values are dropped and counted; optional `--z`, `--ebv` columns enable
the redshift and dust tests, otherwise they are reported as *skipped*).

## Reference outputs in `results/`

| File | Produced by | Provenance |
|---|---|---|
| `run_real_allsky_results.json`, `run_real_allsky_summary.png`, `option2_real_analysis_stdout.log` | step 2 | re-run for this release on the shipped labels |
| `followup_real_results.json` | step 3 | re-run for this release |
| `literature_axes_results.json` | step 4 | re-run for this release |
| `robustness_results_fixed.csv`, `table4_rows.tex` | step 5 | produced from the VizieR export (not re-run in this release) |
| `final_control.json` | step 6 | produced from the VizieR export (not re-run in this release) |
| `benchmark_scaling_results.json` | step 7 | measured by the authors: AMD Ryzen 7 9850X3D, Windows 11, Python 3.14.4, NumPy 2.5.3, 1 thread |

Timings depend on the machine (run-to-run variation of several per cent); the numbers in the article
are those of `results/benchmark_scaling_results.json`.

## Data provenance

`data/real_galaxy_labels.csv` is derived from the Galaxy Zoo 1 data-release catalogue of Lintott et al. (2011),
VizieR `J/MNRAS/410/166`: `spin = +1` (clockwise, Z-wise) if `pcS >= 0.80`, `-1` (anticlockwise, S-wise) if `paS >= 0.80`.
SHA-256 of the file: `49bb62a52fbb3240e71780bcff22efaf63698c47a960a7c2c27bcad965b9756c`.
Please cite the original catalogue when using the labels.

## Citation

See `CITATION.cff`.  Archive: Zenodo DOI 10.5281/zenodo.23298960.
