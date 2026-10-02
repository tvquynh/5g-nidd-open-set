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
| KNN-OOD | penultimate features | distance to the k-th nearest training probability vector, k=50; reference set subsampled to 30,000 vectors with a fixed sampling seed when the training set is larger |

The deep models run over five seeds and the boosted ensembles over ten because
a single FT-Transformer fit averages 88 minutes against 41 seconds for
LightGBM.

## Findings, in one line each

1. **The wrapper buys triage, not detection.** Without a rule a classifier
   cannot emit *unknown* at all, so novel-recall is zero by construction. What
   is measured is where those flows go: 87.8-95.3% receive a known *attack*
   label rather than a benign one, at 0.94-0.98 mean confidence.
2. **MSP and the evaluated log-MSP surrogate share a ranking.** The scores
   `1 - p_max` and `-log p_max` are strictly monotone transforms of one
   another. Decision equivalence holds under an order-statistic quantile on a
   common calibration set; the reported runs used linear interpolation and
   produced identical reported metrics, and sample-level identity of the
   rejection masks was not tested. Separately, substituting log-probabilities
   into energy at unit temperature is algebraically constant, and the detector
   built on it thresholds floating-point residue while still reporting a
   nonzero novel-recall.
3. **Separate runs can confound the scoring rule with the experimental
   configuration.** Our earlier design produced differences in six of ten
   paired seeds, reaching 0.28 novel-recall. The largest discrepancy involved
   different training configurations, including 71,891 versus 215,675 training
   samples, so it does not isolate the effect of refitting under otherwise
   identical conditions. Scoring every rule on a shared fitted model removes
   this source of between-run variation and costs one fifth as much.
4. **Rejection difficulty is attack-type-specific.** Slow-rate denial of
   service is 71.1% of the novel flows and remains the hardest held-out type
   in the evaluated configurations: its recall ranges from 0.113 to 0.463
   across the rules, while under MSP and KNN-OOD the two types the rules do
   catch range from 0.912 to 0.984. The classifier absorbs it into one known
   attack family at 0.994 mean confidence, which is consistent with the
   difficulty but does not establish that no probability-space score could
   improve it.
5. **The operating point affects the comparison, and its benign rate must be
   measured.** Across the evaluated quantiles from 0.90 to 0.99 the largest
   range of mean open-set macro-F1 is 0.0454 on the seven-label scale, which
   is 15.9 percentage points under the two-label rescaling the paper defines,
   and some rule rankings change within that range. Under MSP at the
   pooled-training 95th-percentile threshold the realized benign false-unknown
   rate ranges from 0.0158 to 0.1018 across base classifiers, so a quantile of
   the pooled score distribution is not a guarantee of the rejection rate on
   benign traffic.

KNN-OOD has the highest mean open-set macro-F1 on all four base
classifiers, but its advantage over MSP reaches an uncorrected `p < 0.05` only
on LightGBM; the other three comparisons do not, and the mean difference is
particularly small on XGBoost. KNN-OOD does exceed Mahalanobis on every
reported seed.

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
python -m pytest tests/ -q                   # 35 tests over the scoring rules
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
tests/                           unit tests over the scoring rules and splits
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

The loader, the splits and the model factories were developed alongside a
separate cross-station study by the same group, which lives in its own
repository. The two studies report no result that depends on the other and share
no table or figure. This repository is self-contained: it carries only the
modules the open-set entry points reach and only the result records behind this
study, so nothing here has to be read against the other repository.

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
