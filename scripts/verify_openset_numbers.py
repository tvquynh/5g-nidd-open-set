"""Re-derive every numeric claim in the open-set manuscript from the results.

The manuscript's prose carries numbers that the generated tables do not. This
script recomputes each of them from the record files and compares against what
the text says, so a stale claim fails loudly instead of shipping.

Usage:
    python scripts/verify_openset_numbers.py
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.run_experiment import load_paths  # noqa: E402

TOL = 5e-5


class Check:
    """One claim: what the text says, and what the data says."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, bool]] = []

    def equal(self, name: str, claimed, actual, tol: float = TOL) -> None:
        if isinstance(claimed, (int, float)) and isinstance(actual, (int, float)):
            ok = abs(float(claimed) - float(actual)) <= tol
            self.rows.append((name, f"{claimed}", f"{actual:.6g}", ok))
        else:
            ok = str(claimed) == str(actual)
            self.rows.append((name, str(claimed), str(actual), ok))

    def report(self) -> int:
        width = max(len(r[0]) for r in self.rows)
        failures = 0
        for name, claimed, actual, ok in self.rows:
            mark = "ok  " if ok else "FAIL"
            print(f"  {mark} {name:<{width}}  text={claimed:<14} data={actual}")
            failures += not ok
        print(f"\n{len(self.rows) - failures} of {len(self.rows)} checks pass")
        return failures


def main() -> int:
    paths = load_paths()
    results = Path(paths["results"])
    check = Check()

    grid = pd.read_csv(results / "openset_summary_per_seed.csv")
    composition = json.loads((results / "openset_composition.json").read_text())
    cell = composition["per_seed"]["42"]
    train, test = cell["train_counts"], cell["test_counts"]

    print("Dataset and split")
    check.equal("full flows", 1_215_890, composition["full_rows"], tol=0)
    check.equal("n_train", 215_675, cell["n_train"], tol=0)
    check.equal("n_test", 84_325, cell["n_test"], tol=0)
    novel = sum(v for k, v in test.items() if k != "Benign")
    check.equal("novel flows", 25_388, novel, tol=0)
    check.equal("slow-rate share of novel", 0.711,
                test["SlowrateDoS"] / novel, tol=5e-4)
    check.equal("other types share of novel", 0.289,
                1 - test["SlowrateDoS"] / novel, tol=5e-4)
    check.equal("UDP flood share of train", 0.523,
                train["UDPFlood"] / cell["n_train"], tol=5e-4)
    check.equal("benign share of train", 0.273,
                train["Benign"] / cell["n_train"], tol=5e-4)

    print("\nC1: novel-recall without a rejection rule is zero by construction")
    none = grid[grid.method == "none"]
    check.equal("max novel-recall, no rule", 0.0, none.novel_recall.max(), tol=0)
    check.equal("max std, no rule", 0.0,
                none.groupby("base_model").novel_recall.std().max(), tol=0)
    check.equal("base fits", 30, grid.groupby(["base_model", "seed"]).ngroups, tol=0)
    check.equal("scored records", 150, len(grid), tol=0)

    print("\nC2: MSP equals energy on every base and seed")
    metrics = ["macro_f1_open", "novel_recall", "false_unknown_rate"]
    worst = 0.0
    for base in grid.base_model.unique():
        block = grid[grid.base_model == base]
        msp = block[block.method == "msp"].set_index("seed")[metrics]
        energy = block[block.method == "energy"].set_index("seed")[metrics]
        common = msp.index.intersection(energy.index)
        worst = max(worst, float((msp.loc[common] - energy.loc[common]).abs().max().max()))
    check.equal("max |MSP - energy|, any base/seed/metric", 0.0, worst, tol=0)

    print("\nC3: the per-rule-refit confound")
    legacy = pd.DataFrame([json.loads(Path(f).read_text())
                           for f in glob.glob(str(results / "metrics" / "openset_*.json"))])
    lgb = legacy[legacy.base_model == "lightgbm"]
    msp = lgb[lgb.method == "msp"].set_index("seed").novel_recall.sort_index()
    energy = lgb[lgb.method == "energy"].set_index("seed").novel_recall.sort_index()
    delta = (msp - energy).abs()
    check.equal("legacy LightGBM MSP mean", 0.3522, msp.mean())
    check.equal("legacy LightGBM energy mean", 0.3241, energy.mean())
    check.equal("legacy mean difference", 0.0281, msp.mean() - energy.mean())
    check.equal("legacy seeds differing", 6, int((delta > 0).sum()), tol=0)
    check.equal("legacy max difference", 0.2818, delta.max())
    check.equal("legacy differences below max", 0.0008,
                delta[delta.index != 42].max())
    check.equal("legacy seeds under 0.0008", 5,
                int(((delta > 0) & (delta <= 0.0008)).sum()), tol=0)
    check.equal("legacy MSP mean excluding seed 42", 0.3270, msp.drop(42).mean())
    check.equal("legacy energy mean excluding seed 42", 0.3271, energy.drop(42).mean())
    xgb = legacy[legacy.base_model == "xgboost"]
    xm = xgb[xgb.method == "msp"].set_index("seed").novel_recall.sort_index()
    xe = xgb[xgb.method == "energy"].set_index("seed").novel_recall.sort_index()
    check.equal("legacy XGBoost seeds differing", 0, int(((xm - xe).abs() > 0).sum()), tol=0)
    seed42 = lgb[(lgb.seed == 42) & (lgb.method.isin(["msp", "energy"]))]
    check.equal("seed 42 n_train values", "[71891, 215675]",
                str(sorted(seed42.n_train.unique().tolist())))
    shared = grid[(grid.base_model == "lightgbm") & (grid.method == "msp")]
    check.equal("shared-fit LightGBM MSP mean", 0.3270, shared.novel_recall.mean())
    check.equal("shared-fit LightGBM MSP std", 0.0188, shared.novel_recall.std())

    print("\nC4: per-attack rejectability")
    per_attack = pd.read_csv(results / "openset_detail_summary_per_attack_per_seed.csv")
    # Base classifiers weighted equally, as Section 5.3 states: average over
    # seeds within a base, then over bases.
    pooled = (per_attack[per_attack.rule != "none"]
              .groupby(["attack", "rule", "base_model"])["recall"].mean()
              .groupby(["attack", "rule"]).mean().unstack())
    record_weighted = (per_attack[per_attack.rule != "none"]
                       .groupby(["attack", "rule"])["recall"].mean().unstack())
    check.equal("the two poolings differ, so the convention matters", True,
                abs(pooled.loc["SlowrateDoS", "knn"]
                    - record_weighted.loc["SlowrateDoS", "knn"]) > 1e-3)
    check.equal("SYN flood, MSP", 0.957, pooled.loc["SYNFlood", "msp"], tol=5e-4)
    check.equal("SYN flood, KNN-OOD", 0.984, pooled.loc["SYNFlood", "knn"], tol=5e-4)
    check.equal("TCP connect scan, MSP", 0.912, pooled.loc["TCPConnectScan", "msp"], tol=5e-4)
    check.equal("TCP connect scan, KNN-OOD", 0.980, pooled.loc["TCPConnectScan", "knn"], tol=5e-4)
    check.equal("slow-rate DoS, MSP", 0.292, pooled.loc["SlowrateDoS", "msp"], tol=5e-4)
    check.equal("slow-rate DoS, Mahalanobis", 0.113, pooled.loc["SlowrateDoS", "mahalanobis"], tol=5e-4)
    check.equal("slow-rate DoS, KNN-OOD", 0.463, pooled.loc["SlowrateDoS", "knn"], tol=5e-4)
    msp_pooled = grid[grid.method == "msp"].groupby("base_model").novel_recall.mean()
    check.equal("MSP aggregate novel-recall over the four bases", 0.476,
                msp_pooled.mean(), tol=5e-4)
    benign = pooled.loc["Benign (false unknown)"]
    check.equal("benign false alarm, lowest rule", 0.062, benign.min(), tol=5e-4)
    check.equal("benign false alarm, highest rule", 0.081, benign.max(), tol=5e-4)

    print("\nRule ranking")
    means = grid.groupby(["base_model", "method"], observed=True).macro_f1_open.mean()
    ranking = [("lightgbm", 0.2124, 0.0711, 0.743), ("xgboost", 0.2420, 0.1007, 0.847),
               ("tabnet", 0.2334, 0.0942, 0.817), ("ftt", 0.2447, 0.1033, 0.857)]
    for base, f1, gain, restricted in ranking:
        check.equal(f"{base}: KNN-OOD macro-F1-open", f1, means[(base, "knn")])
        check.equal(f"{base}: gain over no rejection", gain,
                    means[(base, "knn")] - means[(base, "none")])
        check.equal(f"{base}: restricted two-label macro-F1", restricted,
                    round(means[(base, "knn")] * 3.5, 3), tol=0)
        best = (grid[(grid.base_model == base) & (grid.method != "none")]
                .groupby("method", observed=True).macro_f1_open.mean())
        check.equal(f"{base}: best rule is KNN-OOD", "knn", best.idxmax())
        check.equal(f"{base}: worst rule is Mahalanobis", "mahalanobis", best.idxmin())
    stds = grid[grid.method == "knn"].groupby("base_model").novel_recall.std()
    for base, value in [("tabnet", 0.1832), ("lightgbm", 0.1469),
                        ("ftt", 0.0878), ("xgboost", 0.0452)]:
        check.equal(f"{base}: KNN-OOD novel-recall std", value, stds[base])
    non_none = grid[grid.method != "none"].groupby(
        ["base_model", "method"], observed=True).novel_recall.mean()
    check.equal("aggregate novel-recall minimum", 0.2558, non_none.min())
    check.equal("aggregate novel-recall maximum", 0.7052, non_none.max())

    print("\nWhere unflagged novel flows go, and how sure the base is")
    TRAINED = ["HTTPFlood", "SYNScan", "UDPScan", "UDPFlood", "ICMPFlood"]
    HELD_OUT = ["SlowrateDoS", "TCPConnectScan", "SYNFlood"]

    def split(counts):
        total = sum(counts.values())
        benign = counts.get("Benign", 0)
        untrained = sum(counts.get(a, 0) for a in HELD_OUT)
        return (benign / total, (total - benign - untrained) / total,
                untrained / total)

    dest_records = [json.loads(Path(f).read_text()) for f in
                    sorted(glob.glob(str(results / "openset_destinations" / "*.json")))]
    check.equal("destination base fits", 30, len(dest_records), tol=0)
    check.equal("novel-recall without a rule, recomputed", 0.0,
                max(r["novel_recall_no_rejection"] for r in dest_records), tol=0)

    dest_rows = []
    for r in dest_records:
        benign, trained, untrained = split(r["novel_destination_counts"])
        dest_rows.append({
            "base": r["base_model"], "seed": r["seed"], "benign": benign,
            "trained": trained, "untrained": untrained,
            "conf_novel": r["confidence_novel"]["mean"],
            "conf_known": r["confidence_known"]["mean"],
            "novel_above": r["confidence_novel"]["fraction_above_0.99"]})
    dest = pd.DataFrame(dest_rows)
    by_base = dest.groupby("base")[["benign", "trained", "untrained",
                                    "conf_novel", "conf_known",
                                    "novel_above"]].mean()

    check.equal("novel flows given a known attack label, lowest base", 0.878,
                by_base.trained.min(), tol=5e-4)
    check.equal("novel flows given a known attack label, highest base", 0.953,
                by_base.trained.max(), tol=5e-4)
    check.equal("novel flows called benign, lowest base", 0.047,
                by_base.benign.min(), tol=5e-4)
    check.equal("novel flows called benign, highest base", 0.122,
                by_base.benign.max(), tol=5e-4)
    # No base places a novel flow on an output slot that carried no training
    # data. An earlier version of the pipeline appeared to show TabNet doing so;
    # that was a column-index-as-class-code defect, fixed, and pinned by
    # tests/test_class_order.py. These checks exist so it cannot come back.
    check.equal("no base uses the untrained slots, worst base", 0.0,
                by_base.untrained.max(), tol=1e-9)
    check.equal("no base uses the untrained slots, worst seed", 0.0,
                dest.untrained.max(), tol=1e-9)

    check.equal("confidence on novel flows, lowest base", 0.943,
                by_base.conf_novel.min(), tol=5e-4)
    check.equal("confidence on novel flows, highest base", 0.980,
                by_base.conf_novel.max(), tol=5e-4)
    check.equal("confidence on known flows, lowest base", 0.994,
                by_base.conf_known.min(), tol=5e-4)
    check.equal("confidence on known flows, highest base", 0.999,
                by_base.conf_known.max(), tol=5e-4)
    check.equal("novel flows still above 0.99, lowest base", 0.71,
                by_base.novel_above.min(), tol=5e-3)
    check.equal("novel flows still above 0.99, highest base", 0.80,
                by_base.novel_above.max(), tol=5e-3)
    gaps = (by_base.conf_known - by_base.conf_novel)
    check.equal("largest confidence gap is FT-Transformer", "ftt", gaps.idxmax())
    check.equal("that gap", 0.051, gaps.max(), tol=5e-4)
    check.equal("TabNet has the second smallest gap", 2,
                int(gaps.rank().loc["tabnet"]), tol=0)

    print("\nPer held-out attack type: destination and confidence")
    attack_rows = []
    for r in dest_records:
        for attack, block in r["per_attack_destinations"].items():
            benign, trained, _ = split(block["counts"])
            total = sum(block["counts"].values())
            attack_rows.append({
                "base": r["base_model"], "attack": attack, "benign": benign,
                "trained": trained,
                "to_http": block["counts"].get("HTTPFlood", 0) / total,
                "confidence": block["confidence"]["mean"]})
    per_attack_dest = (pd.DataFrame(attack_rows)
                       .groupby(["attack", "base"]).mean(numeric_only=True)
                       .groupby("attack").mean())
    check.equal("slow-rate DoS labeled a known attack", 0.998,
                per_attack_dest.loc["SlowrateDoS", "trained"], tol=5e-4)
    check.equal("slow-rate DoS labeled benign", 0.002,
                per_attack_dest.loc["SlowrateDoS", "benign"], tol=5e-4)
    check.equal("slow-rate DoS labeled HTTP flood", 0.995,
                per_attack_dest.loc["SlowrateDoS", "to_http"], tol=5e-4)
    check.equal("slow-rate DoS confidence", 0.994,
                per_attack_dest.loc["SlowrateDoS", "confidence"], tol=5e-4)
    check.equal("TCP connect scan labeled benign", 0.200,
                per_attack_dest.loc["TCPConnectScan", "benign"], tol=5e-4)
    check.equal("SYN flood labeled benign", 0.276,
                per_attack_dest.loc["SYNFlood", "benign"], tol=5e-4)
    check.equal("TCP connect scan confidence", 0.899,
                per_attack_dest.loc["TCPConnectScan", "confidence"], tol=5e-4)
    check.equal("SYN flood confidence", 0.900,
                per_attack_dest.loc["SYNFlood", "confidence"], tol=5e-4)
    check.equal("the unrejectable type is the most confident one", "SlowrateDoS",
                per_attack_dest.confidence.idxmax())
    check.equal("and the least often called benign", "SlowrateDoS",
                per_attack_dest.benign.idxmin())

    print("\nDispersion the operational claims rest on")
    fu_stats = grid.groupby(["base_model", "method"], observed=True).false_unknown_rate
    fu_sd = fu_stats.std()
    check.equal("LightGBM Mahalanobis false-unknown SD", 0.0543,
                fu_sd[("lightgbm", "mahalanobis")], tol=5e-5)
    check.equal("per-seed budget ratio, LightGBM worst rule, lowest", 0.61,
                fu_stats.min()[("lightgbm", "mahalanobis")] / 0.05, tol=5e-3)
    check.equal("per-seed budget ratio, LightGBM worst rule, highest", 3.63,
                fu_stats.max()[("lightgbm", "mahalanobis")] / 0.05, tol=5e-3)
    check.equal("one LightGBM seed underspends the nominal budget", True,
                bool(fu_stats.min()[("lightgbm", "mahalanobis")] < 0.05))

    print("\nThe T=1 residue is the wrapper's arithmetic, not summation noise")
    temp_records = [json.loads(Path(f).read_text()) for f in
                    sorted(glob.glob(str(results / "openset_temperature" / "*.json")))]
    lgb_extremes = set()
    for r in temp_records:
        if r["base_model"] != "lightgbm":
            continue
        low, high = r["rules"]["energy_T1"]["score_train_range"]
        lgb_extremes.add(round(max(abs(low), abs(high)), 18))
    check.equal("LightGBM T=1 extreme is one value across all seeds", 1,
                len(lgb_extremes), tol=0)
    # The score is -log(sum of clipped probabilities). With eight of the nine
    # class probabilities lifted to the floor, that sum is 1 + 8*eps, so the
    # magnitude is log1p(8e-12) exactly.
    extreme = next(iter(lgb_extremes))
    check.equal("LightGBM T=1 extreme is 8 times the clip floor",
                8e-12, extreme, tol=1e-24)
    check.equal("that is 8 of the 9 class probabilities lifted to the floor",
                8, round(extreme / 1e-12), tol=0)
    check.equal("it is three orders above float64 summation noise", True,
                extreme > 1000 * 9 * 2**-53)
    binary32 = 2.0 ** -23
    for base, low, high in [("xgboost", 0.7, 1.1), ("tabnet", 2.0, 2.5),
                            ("ftt", 3.0, 3.4)]:
        units = [max(abs(a), abs(b)) / binary32
                 for r in temp_records if r["base_model"] == base
                 for a, b in [r["rules"]["energy_T1"]["score_train_range"]]]
        check.equal(f"{base} T=1 residue in units of 2^-23, within [{low}, {high}]",
                    True, low <= min(units) and max(units) <= high)

    print("\nRule ranking: what the paired tests support")
    stats = pd.read_csv(results / "openset_stats.csv")
    f1 = stats[stats.metric == "macro_f1_open"]
    against_none = f1[f1.alternative == "none"]
    check.equal("KNN-OOD ahead of no rejection on every seed", 30,
                int(against_none.wins.sum()), tol=0)
    check.equal("seeds compared against no rejection", 30,
                int(against_none.seeds.sum()), tol=0)
    check.equal("smallest paired d_z against no rejection", 4.05,
                against_none.cohens_dz.min(), tol=0.05)
    check.equal("largest paired d_z against no rejection", 22.36,
                against_none.cohens_dz.max(), tol=0.05)
    check.equal("effect sizes are the paired estimator", True,
                "cohens_dz" in stats.columns and "cohens_d" not in stats.columns)
    boosted = ["lightgbm", "xgboost"]
    check.equal("p against no rejection, boosted bases", 0.0020,
                against_none[against_none.base_model.isin(boosted)].p_value.max(),
                tol=5e-5)
    check.equal("p against no rejection, deep bases", 0.0625,
                against_none[~against_none.base_model.isin(boosted)].p_value.max(),
                tol=5e-5)

    against_mahalanobis = f1[f1.alternative == "mahalanobis"]
    check.equal("KNN-OOD ahead of Mahalanobis on every seed", 30,
                int(against_mahalanobis.wins.sum()), tol=0)
    check.equal("smallest paired d_z against Mahalanobis", 1.60,
                against_mahalanobis.cohens_dz.min(), tol=0.05)
    check.equal("largest paired d_z against Mahalanobis", 3.33,
                against_mahalanobis.cohens_dz.max(), tol=0.05)

    against_msp = f1[f1.alternative == "msp"].set_index("base_model")
    for base, delta, p_value, wins in [("lightgbm", 0.0118, 0.0137, 8),
                                       ("tabnet", 0.0241, 0.1250, 4),
                                       ("ftt", 0.0179, 0.1875, 4),
                                       ("xgboost", 0.0006, 0.6250, 3)]:
        check.equal(f"{base}: KNN-OOD over MSP, delta", delta,
                    against_msp.loc[base, "delta"])
        check.equal(f"{base}: KNN-OOD over MSP, p", p_value,
                    against_msp.loc[base, "p_value"], tol=5e-5)
        check.equal(f"{base}: KNN-OOD over MSP, wins", wins,
                    int(against_msp.loc[base, "wins"]), tol=0)
    check.equal("xgboost: paired d_z for KNN-OOD over MSP", 0.11,
                against_msp.loc["xgboost", "cohens_dz"], tol=5e-3)
    check.equal("no KNN-OOD-over-MSP comparison reaches p<0.01", True,
                bool((against_msp.p_value >= 0.01).all()))
    recall = stats[(stats.metric == "novel_recall") & (stats.alternative == "msp")
                   ].set_index("base_model")
    check.equal("lightgbm: KNN-OOD over MSP, novel-recall delta", 0.1406,
                recall.loc["lightgbm", "delta"])
    check.equal("lightgbm: KNN-OOD over MSP, novel-recall p", 0.0020,
                recall.loc["lightgbm", "p_value"], tol=5e-5)

    print("\nC5: operating point and realized budget")
    detail = [json.loads(Path(f).read_text())
              for f in sorted(glob.glob(str(results / "openset_detail" / "*.json")))]
    rows = [{"base": r["base_model"], "seed": r["seed"], "rule": rule,
             "q": float(q), "f1": m["macro_f1_open"]}
            for r in detail for rule, s in r["threshold_sweep"].items()
            for q, m in s.items()]
    sweep = pd.DataFrame(rows).pivot_table(index=["base", "rule"], columns="q",
                                           values="f1", aggfunc="mean")
    spread = sweep.max(axis=1) - sweep.min(axis=1)
    check.equal("sweep cells", 16, len(spread), tol=0)
    check.equal("largest sweep spread on seed means", 0.0454, spread.max())
    per_seed = (pd.DataFrame(rows)
                .pivot_table(index=["base", "rule", "seed"], columns="q", values="f1")
                .pipe(lambda t: t.max(axis=1) - t.min(axis=1)))
    check.equal("largest sweep spread on a single seed", 0.0726, per_seed.max())
    check.equal("cells moving under 0.036", 14, int((spread < 0.036).sum()), tol=0)
    check.equal("detail base fits", 30, len(detail), tol=0)
    temperature_fits = len(list((results / "openset_temperature").glob("*.json")))
    check.equal("temperature base fits", 30, temperature_fits, tol=0)

    fu = grid.groupby(["base_model", "method"], observed=True).false_unknown_rate.mean()
    for base, value in [("lightgbm", 0.0687), ("xgboost", 0.0604),
                        ("tabnet", 0.1018), ("ftt", 0.0158)]:
        check.equal(f"{base}: realized benign rate, MSP", value, fu[(base, "msp")])
    realized = fu[fu.index.get_level_values("method") != "none"]
    check.equal("realized rate minimum, any rule", 0.0030, realized.min(), tol=5e-5)
    check.equal("realized rate maximum, any rule", 0.1347, realized.max(), tol=5e-5)
    check.equal("worst overspend ratio", 2.69, realized.max() / 0.05, tol=5e-3)
    check.equal("worst underspend ratio", 0.06, realized.min() / 0.05, tol=5e-3)
    check.equal("alerts per 1M benign flows, FT-Transformer MSP", 15_800,
                round(fu[("ftt", "msp")] * 1e6, -2), tol=50)
    check.equal("alerts per 1M benign flows, TabNet MSP", 101_800,
                round(fu[("tabnet", "msp")] * 1e6, -2), tol=50)

    print("\nTemperature campaign")
    temp_rows = []
    for f in sorted(glob.glob(str(results / "openset_temperature" / "*.json"))):
        record = json.loads(Path(f).read_text())
        for rule, m in record["rules"].items():
            low, high = m["score_train_range"]
            temp_rows.append({"base": record["base_model"], "seed": record["seed"],
                              "rule": rule, "nr": m["novel_recall"],
                              "flag": m["flag_rate"], "f1": m["macro_f1_open"],
                              "uniq": m["score_train_unique"],
                              "threshold": m["threshold"],
                              "max_abs": max(abs(low), abs(high))})
    temp = pd.DataFrame(temp_rows)
    if temp.empty:
        print("  (no temperature records yet)")
    else:
        agg = temp.groupby(["base", "rule"]).agg(
            nr=("nr", "mean"), flag=("flag", "mean"), f1=("f1", "mean"),
            uniq=("uniq", "min"), max_abs=("max_abs", "max"))
        def deviation(predicate) -> float:
            worst = 0.0
            for base in temp.base.unique():
                reference = agg.loc[(base, "msp"), "nr"]
                for rule in temp.rule.unique():
                    if not rule.startswith("energy_T"):
                        continue
                    if not predicate(float(rule.split("T")[1])):
                        continue
                    worst = max(worst, abs(agg.loc[(base, rule), "nr"] - reference))
            return worst

        check.equal("max |energy(T<=0.25) - MSP| novel-recall", 0.0,
                    deviation(lambda t: t <= 0.25), tol=0)
        check.equal("max |energy(T=0.5) - MSP| novel-recall, under 8e-4", True,
                    deviation(lambda t: t == 0.5) < 8e-4)
        check.equal("that deviation is at most twenty novel flows", True,
                    deviation(lambda t: t == 0.5) <= 20 / 25_388 + 1e-9)
        # The MSP threshold is the q-quantile of 1 - max(p), so 1 - threshold is
        # the maximum class probability at that quantile: the peakedness that
        # makes the finite-temperature energy score collapse onto MSP.
        peakedness = 1.0 - temp[temp.rule == "msp"].groupby("base").threshold.mean()
        check.equal("max class probability at q=0.95, least peaked base, over 0.993",
                    True, bool(peakedness.min() > 0.993))
        check.equal("three of four bases exceed 0.998", 3,
                    int((peakedness > 0.998).sum()), tol=0)
        check.equal("least peaked base is FT-Transformer", "ftt", peakedness.idxmin())

        # The residual disagreement at T = 0.5, counted in novel flows, should
        # track peakedness: the flattest base departs from MSP the most.
        paired = temp.pivot_table(index=["base", "seed"], columns="rule",
                                  values="nr")
        departures = ((paired["energy_T0.5"] - paired["msp"]).abs()
                      .groupby("base").max() * 25_388).round()
        check.equal("T=0.5 departure, FT-Transformer, in novel flows", 20,
                    departures["ftt"], tol=0)
        check.equal("T=0.5 departure, TabNet, in novel flows", 1,
                    departures["tabnet"], tol=0)
        check.equal("T=0.5 departure, boosted ensembles, in novel flows", 2,
                    max(departures["lightgbm"], departures["xgboost"]), tol=0)
        check.equal("the flattest base departs the most", "ftt",
                    departures.idxmax())
        degenerate = agg.loc[(slice(None), "energy_T1"), :]
        check.equal("T=1 largest max|s| over bases", 3.9e-7,
                    degenerate.max_abs.max(), tol=5e-8)
        check.equal("T=1 smallest max|s| over bases", 8.0e-12,
                    degenerate.max_abs.min(), tol=5e-13)

        # The manuscript states the collapse as a range of orders of magnitude
        # between each base classifier's informative scale and its residue.
        msp_scale = agg.loc[(slice(None), "msp"), "max_abs"].droplevel(1)
        residue = degenerate.max_abs.droplevel(1)
        gaps = [math.log10(msp_scale[b] / residue[b]) for b in residue.index]
        check.equal("MSP score scale, lowest base", 0.59, msp_scale.min(), tol=5e-3)
        check.equal("MSP score scale, highest base", 0.79, msp_scale.max(), tol=5e-3)
        check.equal("smallest collapse, at least six orders", True, min(gaps) >= 6.0)
        check.equal("largest collapse, at most eleven orders", True, max(gaps) <= 11.0)

        ftt_uniq = temp[(temp.base == "ftt") & (temp.rule == "energy_T1")].uniq
        check.equal("T=1 FT-Transformer distinct scores, about 1300", True,
                    1200 <= ftt_uniq.min() and ftt_uniq.max() <= 1400)
        big = agg.loc[(["xgboost", "tabnet"], "energy_T1"), "uniq"]
        check.equal("T=1 XGBoost and TabNet distinct scores in the tens of thousands",
                    True, bool((big >= 10_000).all()))
        check.equal("T=1 flag rate, lowest base", 0.071, degenerate.flag.min(), tol=5e-4)
        check.equal("T=1 flag rate, highest base", 0.119, degenerate.flag.max(), tol=5e-4)
        lgb_uniq = temp[(temp.base == "lightgbm") & (temp.rule == "energy_T1")].uniq
        check.equal("T=1 LightGBM distinct scores, fewest", 2, lgb_uniq.min(), tol=0)
        check.equal("T=1 LightGBM distinct scores, most", 6, lgb_uniq.max(), tol=0)
        for base, nr, f1 in [("lightgbm", 0.2247, 0.1825), ("tabnet", 0.1215, 0.1635),
                             ("xgboost", 0.0962, 0.1594), ("ftt", 0.0900, 0.1526)]:
            if (base, "energy_T1") not in agg.index:
                continue
            check.equal(f"T=1 {base} novel-recall", nr, agg.loc[(base, "energy_T1"), "nr"])
            check.equal(f"T=1 {base} macro-F1-open", f1, agg.loc[(base, "energy_T1"), "f1"])
        for base, f1 in [("lightgbm", 0.1413), ("tabnet", 0.1392),
                         ("xgboost", 0.1413), ("ftt", 0.1414)]:
            check.equal(f"no-rejection macro-F1-open, {base}", f1,
                        grid[(grid.base_model == base)
                             & (grid.method == "none")].macro_f1_open.mean())
        print(f"  temperature records so far: {temp.groupby('base').seed.nunique().to_dict()}")

    print()
    return check.report()


if __name__ == "__main__":
    # main() returns the number of failed checks, so success is zero.
    failures = main()
    raise SystemExit(0 if failures == 0 else 1)
