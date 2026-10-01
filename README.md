# Probability-only open-set rules for 5G intrusion detection

Reproducibility artifact: the pipeline, the per-cell result records, the unit
tests, and a script that re-derives every numeric claim in the accompanying
manuscript from those records.

An intrusion detector meets attacks outside its training catalog. The usual
remedy is to wrap the trained classifier in an out-of-distribution scoring rule
that can answer *unknown*. Such a wrapper normally sees only class
probabilities: pre-softmax logits and penultimate activations, which the
originating literature assumes, are not exposed across a service boundary. This
study measures what that constraint does to four published rules.

## What is measured

Four base classifiers, four rejection rules plus a no-rejection control, on
5G-NIDD with three of the eight attack types held out of training.

| Base classifier | Seeds |
|---|---|
| LightGBM | 10 |
| XGBoost | 10 |
| TabNet | 5 |
| FT-Transformer | 5 |

| Rule | Defined on | Under probability-only access |
|---|---|---|
| Maximum softmax probability | class probabilities | native |
| Energy | logits | reformulated in probability space |
| Mahalanobis | penultimate features | fitted on the probability simplex, ridge 1e-3 |
| KNN-OOD | penultimate features | distance to the k-th nearest training probability vector, k=50 |

The deep models run over five seeds and the boosted ensembles over ten because
a single FT-Transformer fit averages 88 minutes against 41 seconds for
LightGBM.

## Findings, in one line each

1. **The wrapper buys triage, not detection.** Without a rule a classifier
   cannot emit *unknown* at all, so novel-recall is zero by construction. What
   is measured is where those flows go: 77-95% receive a known *attack* label
   rather than a benign one, at 0.94-0.98 mean confidence.
2. **Under probability-only access the energy score is the MSP rule.** The two
   are strictly monotone transforms of one another, so at a quantile-calibrated
   threshold they flag identical sets. At unit temperature the textbook
   substitution is algebraically constant and the detector built on it
   thresholds floating-point residue while still reporting a nonzero
   novel-recall.
3. **Refitting the base classifier once per rule confounds the rule with the
   fit.** Under that design two mathematically identical rules differed on six
   of ten seeds, once by 0.28 novel-recall. Scoring every rule from one fit
   removes the confound by construction and costs one fifth as much.
4. **Rejectability tracks the base classifier's confidence, not the rule.**
   Slow-rate denial of service is 71% of the novel flows and resists every
   rule, because the classifier absorbs it into one known attack family at
   0.994 confidence; the two types the rules do catch sit at 0.90.
5. **The operating point is not load-bearing, and its budget is not honored.**
   A quantile sweep from 0.90 to 0.99 moves open-set macro-F1 by at most 0.045,
   while the realized benign false-alarm rate misses its nominal budget by more
   than a factor of two in both directions.

## Dataset

5G-NIDD, captured on the 5G Test Network at the University of Oulu: 1,215,890
flows, 9 classes, two physically separate Pico base stations. Licensed CC BY 4.0
and distributed by its authors. **This repository does not redistribute it.**

Experiments consume the authors' published ML-ready file (`Encoded.csv`, per
their `dataload.txt`), which yields 89 features after the column drops they
specify. The loader asserts the row and per-class counts on load, so a
divergence in the input fails rather than being silently absorbed.

The split holds out three attack types. Training sees SYN scan, UDP scan, ICMP
flood, UDP flood and HTTP flood plus half the benign traffic; the test partition
adds TCP connect scan, SYN flood and slow-rate denial of service, which the
model has never seen.

## Getting the data

Download `Encoded.csv` from the dataset's own distribution and place it where
`configs/paths.yaml` expects:

```
data/raw/combined_encoded/Encoded.csv
```

Everything else in `configs/paths.yaml` is relative to the repository root and
needs no editing. Adjust that one path if your layout differs.

## Install

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.11. CPU only; no GPU is used or required.

## Reproduce

Check the shipped numbers without running anything:

```bash
python -m pytest tests/ -q                   # 70 tests over the scoring rules
python scripts/verify_openset_numbers.py     # re-derives every number from results/
```

The verifier reads only `results/` and reports one line per claim. It is the
honest test of this artifact: if a number in the manuscript does not follow from
a shipped record, it fails loudly.

Re-run the experiments from the dataset:

```bash
python scripts/run_campaign.py --stage openset              --workers 3
python scripts/run_campaign.py --stage openset_detail       --workers 3
python scripts/run_campaign.py --stage openset_temperature  --workers 3
python scripts/run_campaign.py --stage openset_destinations --workers 3
```

Stages are independent and resumable; each skips jobs whose output already
exists. The four campaigns total about 27.9 hours of job time across 128 jobs,
roughly 9 to 14 hours of wall clock at three workers, dominated by
FT-Transformer (slowest single job 99 minutes). All runs were executed on one
CPU workstation.

## Layout

```
src/
  data_loader.py                 load the author-provided encoded file
  preprocess.py, features.py     column handling and feature assembly
  splits.py                      the held-out-attack split
  models.py, deep_models.py      the four base classifiers
  open_set.py                    the four rejection rules, all in probability space
  run_open_set_all.py            main campaign: one fit, every rule scored from it
  run_open_set_detail.py         per-attack rejectability and the quantile sweep
  run_open_set_temperature.py    the temperature family of the energy score
  run_open_set_destinations.py   where unflagged novel flows actually go
  aggregate_openset*.py          reduce per-cell records to the reported tables
  openset_stats.py               paired Wilcoxon signed-rank and Cohen's d_z
  make_openset_tables*.py        the manuscript's tables
  make_openset_figures.py        the manuscript's figures
scripts/
  run_campaign.py                stage driver
  verify_openset_numbers.py      re-derives every numeric claim from results/
tests/                           unit tests, 26 of them over the scoring rules
configs/                         paths, model hyperparameters, seeds
results/                         per-cell records and aggregated tables
```

## Results layout

| Directory | Records | Behind |
|---|---|---|
| `results/openset_v2/` | 150 | the main campaign, one fit per base and seed |
| `results/openset_detail/` | 30 | per-attack rejectability and the quantile sweep |
| `results/openset_temperature/` | 30 | the temperature family |
| `results/openset_destinations/` | 30 | where unflagged novel flows go |
| `results/metrics/openset_*.json` | 130 | the superseded per-rule-refit design, kept as the evidence for finding 3 |

Aggregated tables sit at the top level of `results/` as `openset_*.csv` and
`openset_*.json`, and the campaign job logs as `campaign_openset*.jsonl`.

## A note on the shared code base

The loader, the splits and the model factories are shared with a separate
cross-station study by the same group, which lives in its own repository. The
two studies report no result that depends on the other and share no table or
figure. This repository carries the full pipeline so that it stands alone, but
only the result records behind the open-set manuscript.

## Authors

Trong-Thua Huynh (Posts and Telecommunications Institute of Technology),
Van-Quynh Trinh (Posts and Telecommunications Institute of Technology,
corresponding author), De-Thu Huynh (The Saigon International University),
Phuc Nguyen (University of Economics and Law, and Vietnam National University,
Ho Chi Minh City).

Funded by the Posts and Telecommunications Institute of Technology, Vietnam.

## License

MIT, see `LICENSE`. The 5G-NIDD dataset is not covered by it and remains under
its own CC BY 4.0 terms.
