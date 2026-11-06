# Structural regularization and spectral risk in quantum subspace reconstruction

Computational data and code for the manuscript by **Ahmet Kamer Çivi**, Department of Physics, Middle East Technical University, 06800 Ankara, Turkey. Correspondence: kamer.civi@metu.edu.tr.

This is a computational research archive, not a claim of journal acceptance. All reported quantum measurements are classically simulated finite-shot outcomes. State preparation and gates are ideal in the reported studies. The scope is stated in the manuscript; the data include unfavorable outcomes as well as improvements.

## Primary data

| Archive | Role | Conditions | Repeats per parameter cell |
|---|---|---:|---:|
| `data/L6_mechanism.zip` | Six-site, U/t=7; two product-formula steps, four budgets | 160 | 20 |
| `data/L8_validation.zip` | Eight-site, U/t=7; two steps, two budgets | 80 | 20 |

A condition is a parameter/budget/noise-seed combination, **not** an independent Hamiltonian. Each archive includes raw branch measurements, model checkpoints, spectra, selection records, the computational protocol, and original scientific source modules. Operational logs and private machine paths are omitted from this public export. The scientific members and original input hashes are described in each `PUBLIC_EXPORT_MANIFEST.json`. A repacked archive has a new ZIP hash; it must not be confused with the original archive digest.

`data/exploratory_latent_L6.zip`, `data/exploratory_optimization_L6.zip`, and `data/exploratory_allocation_L6.zip` retain earlier development and diagnostic records. They are not additional independent validation of the final estimator. In particular, the earlier U/t=5 development study is not represented as a new frozen-estimator U/t=4 robustness experiment.

## Reproduce the released tables without retraining

Python 3.10 or later is required. The quick audit uses only the Python standard library:

```bash
python3 verify_release.py
```

It verifies archive integrity, per-member hashes, counts, and the main eight-site medians. To repeat the deeper independent analyses, install the analysis dependencies in a separate environment and run:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-analysis.txt
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python analysis/l8/analyze_validation9.py   --input data/L8_validation.zip --out reproduced/l8
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python analysis/l8/verify_physics9.py   --input data/L8_validation.zip --out reproduced/l8
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python analysis/l8/verify_time9.py   --input data/L8_validation.zip --out reproduced/l8
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python analysis/l6/analyze_mechanism8.py   --input data/L6_mechanism.zip --out reproduced/l6
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python analysis/l6/audit_time_comparator.py   --input data/L6_mechanism.zip --reference reproduced/l6/reference_arrays.npz --out reproduced/l6
```

These commands analyze recorded data; they do not rerun model training. The `verify_time9.py` check reconstructs historical time-only samples with the documented sampler, which is different from producing new independent trials. Reconstructed time-only counts are explicitly provided at `results/l8/time_full_budget_reconstructed_counts.json`. They were not saved contemporaneously in the original producer archive.

The `results/` directory contains the previously computed tables and audits used by the manuscript. The `source/` directory contains the original producer modules. They are retained byte-for-byte where their hashes are part of the recorded scientific protocol. Tests in those modules are execution checks, not scientific-accuracy guarantees. Their historical output-folder names are implementation labels, not extra studies.

## Interpretation

The main target is the physical spectral function at fixed broadening eta=0.2. Physical-window relative L1 errors and periodized L1 errors are separate metrics and must not be interchanged. Equal shot counts are not equal compiled gate counts or total runtime. The data show both useful and harmful structural regularization, rather than a universally superior estimator.

The exploratory pooling decision rule is provided under `analysis/exploratory/` and `results/pooling_predictor_pilot/`. It was designed after the validation data had been examined and is not a validated main contribution.

## AI assistance and author responsibility

ChatGPT (OpenAI) assisted code development, analytical exploration, literature organization, analysis, and drafting. Separate numerical implementations were used for recorded checks. This computational cross-checking is not an independent laboratory replication. Session-specific model versions were not consistently archived; none are invented in this release. See `documentation/AI_ASSISTANCE.md`.

## Licensing and citation

A software/data license has not been selected by the author. Public visibility alone should not be described as an open-source license. Third-party material retains its existing terms. `CITATION.cff` identifies the author and the computational archive; no journal DOI or acceptance is claimed.

## Release status

This package was assembled on 5 November 2026. Repository creation and upload are separate operations. The supplied publication script only reports success after a real GitHub push and remote visibility check. No repository URL is predeclared as live.
