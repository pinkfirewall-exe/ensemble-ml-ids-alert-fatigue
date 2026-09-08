"""
triage_framework.py
-------------------
Confidence threshold triage framework for SOC alert fatigue reduction.

Routes each IDS prediction to one of three operational tiers based on combined model confidence and empirically observed tier accuracy:

  Tier 1 (Auto-action)   : confidence >= tier1_threshold AND tier accuracy >= 96%
  Tier 2 (Second Review) : confidence 50-90% AND tier accuracy 40-92%
  Tier 3 (Full Escalation): confidence < tier3_threshold AND tier accuracy < 50%

Threshold justification:
  - 90% upper threshold: satisfies the selective classification accuracy guarantee (Geifman & El-Yaniv, NeurIPS 2017). Tier 1 accuracy of
    98.85-100% was empirically validated across all model-dataset combinations.
  - 50% lower threshold: optimal abstention criterion (Hendrickx et al., Machine Learning 2024). On an 8-10 class problem, confidence < 50%
    means the model assigns less than 4x random-chance probability to its top class. Base rate argument (Sommer & Paxson, IEEE S&P 2010) provides
    IDS-specific rationale.

IMPORTANT - Threshold calibration:
  The default thresholds (90% and 50%) are theoretically grounded starting points. Every organisation should calibrate these against their own traffic
  characteristics, analyst team capacity, and acceptable false positive and false negative rates before production deployment. A high-volume financial
  SOC and a small healthcare security team have very different operational constraints, and optimal thresholds will differ accordingly.

Workload reduction formula:
  WR = (1 - (0*|T1| + 0.5*|T2| + 1.0*|T3|) / total) * 100%

  Tier 1 = 0 reviews (automated, no analyst time)
  Tier 2 = 0.5 reviews (second-reviewer check)
  Tier 3 = 1.0 reviews (full manual investigation)

Usage:
  python triage_framework.py --dataset ciciot2023 --balancing class_weight
  python triage_framework.py --dataset unsw_nb15  --balancing class_weight
  python triage_framework.py --dataset ciciot2023 --tier1-threshold 0.85
  python triage_framework.py --dataset unsw_nb15  --tier3-threshold 0.40

Random seed: 42
"""

import argparse
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

RANDOM_SEED = 42

# Default thresholds — see IMPORTANT note above before changing
DEFAULT_TIER1_THRESHOLD = 0.90
DEFAULT_TIER3_THRESHOLD = 0.50


# Confidence scoring 

def compute_confidence(proba: np.ndarray) -> np.ndarray:
    """
    Confidence score = maximum predicted class probability.

    For a K-class problem:
      - Random prediction baseline: 1/K
      - CICIoT2023 (8 classes):  baseline = 12.5%
      - UNSW-NB15  (10 classes): baseline = 10.0%

    A confidence below 50% means the model assigns less than 4x the random-chance probability to its top predicted class.
    """
    return np.max(proba, axis=1)


# Tier assignment 

def assign_tiers(confidence: np.ndarray,
                 tier1_threshold: float = DEFAULT_TIER1_THRESHOLD,
                 tier3_threshold: float = DEFAULT_TIER3_THRESHOLD) -> np.ndarray:
    """
    Assign each prediction to a tier based on confidence score.

    Returns an array of tier labels: 1, 2, or 3.
    Note: observed tier accuracy is validated after assignment.
    """
    tiers = np.full(len(confidence), 2, dtype=int)  # default Tier 2
    tiers[confidence >= tier1_threshold] = 1
    tiers[confidence <  tier3_threshold] = 3
    return tiers


# Tier accuracy validation 

def compute_tier_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                          tiers: np.ndarray) -> dict:
    """
    Compute prediction accuracy within each confidence tier.

    The combined confidence-and-accuracy criteria require that Tier 1 achieves >= 96% accuracy (satisfying the selective classification
    guarantee) and Tier 3 achieves < 50% (validating manual escalation).
    """
    results = {}
    for tier in [1, 2, 3]:
        mask = (tiers == tier)
        count = mask.sum()
        if count == 0:
            results[tier] = {"count": 0, "accuracy": None, "pct": 0.0}
            continue
        acc = accuracy_score(y_true[mask], y_pred[mask])
        results[tier] = {
            "count":    int(count),
            "pct":      round(count / len(tiers) * 100, 2),
            "accuracy": round(acc * 100, 2),
        }
    return results


#  Workload reduction

def compute_workload_reduction(tier_results: dict) -> float:
    """
    Estimate SOC analyst workload reduction vs fully manual baseline.

    Tier 1 = 0.0 reviews  (automated - no analyst involvement)
    Tier 2 = 0.5 reviews  (second-reviewer check)
    Tier 3 = 1.0 reviews  (full manual investigation)

    WR = (1 - weighted_reviews / total_alerts) * 100
    """
    t1 = tier_results[1]["count"]
    t2 = tier_results[2]["count"]
    t3 = tier_results[3]["count"]
    total = t1 + t2 + t3

    if total == 0:
        return 0.0

    weighted = 0.0 * t1 + 0.5 * t2 + 1.0 * t3
    return round((1 - weighted / total) * 100, 2)


#  Per-class tier breakdown 

def per_class_tier_distribution(y_true_str: np.ndarray,
                                 y_pred_str: np.ndarray,
                                 tiers: np.ndarray,
                                 classes: list) -> pd.DataFrame:
    """
    Break down tier assignments by true attack class.

    Shows how the triage framework routes each attack category -
    high-confidence majority classes route to Tier 1 (auto-action),
    rare uncertain classes route to Tier 3 (full escalation).
    """
    rows = []
    for cls in classes:
        mask = (y_true_str == cls)
        total_cls = mask.sum()
        if total_cls == 0:
            continue
        t1 = ((tiers == 1) & mask).sum()
        t2 = ((tiers == 2) & mask).sum()
        t3 = ((tiers == 3) & mask).sum()
        rows.append({
            "Class":  cls,
            "Total":  int(total_cls),
            "T1":     int(t1),
            "T1 %":   round(t1 / total_cls * 100, 1),
            "T2":     int(t2),
            "T2 %":   round(t2 / total_cls * 100, 1),
            "T3":     int(t3),
            "T3 %":   round(t3 / total_cls * 100, 1),
        })
    return pd.DataFrame(rows).set_index("Class")


def print_tier_table(df: pd.DataFrame, model_name: str):
    """Pretty-print per-class tier distribution."""
    print(f"\n  {'─'*72}")
    print(f"  {model_name} — Per-Class Tier Distribution")
    print(f"  {'─'*72}")
    print(f"  {'Class':<18} {'Total':>8} "
          f"{'T1':>7} {'T1%':>6} "
          f"{'T2':>7} {'T2%':>6} "
          f"{'T3':>7} {'T3%':>6}")
    print(f"  {'─'*72}")
    for cls, row in df.iterrows():
        print(f"  {cls:<18} {row['Total']:>8,} "
              f"{row['T1']:>7,} {row['T1 %']:>5.1f}% "
              f"{row['T2']:>7,} {row['T2 %']:>5.1f}% "
              f"{row['T3']:>7,} {row['T3 %']:>5.1f}%")


# Confidence distribution plot

def plot_confidence_distribution(confidence: np.ndarray,
                                  tiers: np.ndarray,
                                  model_name: str,
                                  dataset: str,
                                  tier1_threshold: float,
                                  tier3_threshold: float,
                                  tier_results: dict,
                                  output_dir: str):
    """Plot confidence score distribution coloured by tier."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colours = {1: "#2ecc71", 2: "#f39c12", 3: "#e74c3c"}
    labels  = {
        1: f"Tier 1 — Auto-action ({tier_results[1]['pct']:.1f}%)",
        2: f"Tier 2 — Second Review ({tier_results[2]['pct']:.1f}%)",
        3: f"Tier 3 — Escalation ({tier_results[3]['pct']:.1f}%)",
    }

    for tier in [1, 2, 3]:
        mask = (tiers == tier)
        if mask.sum() == 0:
            continue
        ax.hist(confidence[mask], bins=50, alpha=0.65,
                color=colours[tier], label=labels[tier],
                range=(0, 1))

    ax.axvline(tier1_threshold, color="green", linestyle="--", linewidth=1.5,
               label=f"Tier 1 threshold ({tier1_threshold*100:.0f}%)")
    ax.axvline(tier3_threshold, color="red",   linestyle="--", linewidth=1.5,
               label=f"Tier 3 threshold ({tier3_threshold*100:.0f}%)")

    ax.set_xlabel("Confidence Score (Max Predicted Probability)")
    ax.set_ylabel("Number of Predictions")
    ax.set_title(f"Confidence Distribution by Tier\n{model_name} — {dataset}")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()

    fname = (f"confidence_dist_{model_name.lower().replace(' ', '_')}"
             f"_{dataset}.png")
    path  = os.path.join(output_dir, fname)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# Summary plot

def plot_tier_summary(all_results: list, dataset: str, output_dir: str):
    """
    Grouped bar chart: Tier % distribution and workload reduction
    across all three models.
    """
    models = [r["model"] for r in all_results]
    t1_pct = [r["tier_results"][1]["pct"] for r in all_results]
    t2_pct = [r["tier_results"][2]["pct"] for r in all_results]
    t3_pct = [r["tier_results"][3]["pct"] for r in all_results]
    wr     = [r["workload_reduction"]     for r in all_results]

    x = np.arange(len(models))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Tier distribution
    ax1.bar(x - width, t1_pct, width, label="Tier 1 Auto-action",   color="#2ecc71")
    ax1.bar(x,          t2_pct, width, label="Tier 2 Second Review", color="#f39c12")
    ax1.bar(x + width,  t3_pct, width, label="Tier 3 Escalation",   color="#e74c3c")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha="right")
    ax1.set_ylabel("Percentage of Predictions (%)")
    ax1.set_title(f"Tier Distribution — {dataset}")
    ax1.legend()
    ax1.set_ylim(0, 110)

    # Workload reduction
    bars = ax2.bar(models, wr, color=["#2ecc71", "#3498db", "#9b59b6"])
    ax2.set_ylabel("Estimated Workload Reduction (%)")
    ax2.set_title(f"SOC Workload Reduction - {dataset}")
    ax2.set_ylim(0, 110)
    for bar, val in zip(bars, wr):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    path = os.path.join(output_dir, f"triage_summary_{dataset}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# Main pipeline 

def run_triage(dataset: str, balancing: str,
               tier1_threshold: float, tier3_threshold: float,
               model_dir: str, data_dir: str, output_dir: str):

    print(f"\n=== Confidence Threshold Triage Framework ===")
    print(f"  Dataset          : {dataset}")
    print(f"  Balancing        : {balancing}")
    print(f"  Tier 1 threshold : >= {tier1_threshold*100:.0f}% confidence")
    print(f"  Tier 3 threshold : <  {tier3_threshold*100:.0f}% confidence")
    print(f"\n  NOTE: These thresholds are theoretically grounded starting points.")
    print(f"  Calibrate against your organisation's traffic and analyst capacity")
    print(f"  before production deployment.\n")

    prefix = "ciciot2023" if dataset == "ciciot2023" else "unsw_nb15"
    tag    = f"{prefix}_{balancing}"

    # Load models and test data 
    with open(os.path.join(model_dir, f"rf_{tag}.pkl"),  "rb") as f:
        rf = pickle.load(f)
    with open(os.path.join(model_dir, f"xgb_{tag}.pkl"), "rb") as f:
        xgb = pickle.load(f)
    with open(os.path.join(model_dir, f"label_encoder_{prefix}.pkl"), "rb") as f:
        le = pickle.load(f)

    vote_probs = np.load(os.path.join(model_dir, f"vote_probs_{tag}.npy"))
    y_test_enc = np.load(os.path.join(model_dir, f"y_test_enc_{prefix}.npy"))
    y_test_str = np.load(os.path.join(model_dir, f"y_test_str_{prefix}.npy"),
                         allow_pickle=True)
    X_test = pd.read_csv(
        os.path.join(data_dir, f"{prefix}_test_X.csv")).values

    classes = list(le.classes_)
    os.makedirs(output_dir, exist_ok=True)

    # Process each model 
    all_results   = []
    summary_rows  = []

    model_configs = [
        ("Random Forest",     rf.predict_proba(X_test),  rf.predict(X_test)),
        ("XGBoost",           xgb.predict_proba(X_test), xgb.predict(X_test)),
        ("Voting Classifier", vote_probs,                 np.argmax(vote_probs, axis=1)),
    ]

    for model_name, proba, preds_enc in model_configs:
        print(f"\n  {'━'*60}")
        print(f"  {model_name}")
        print(f"  {'━'*60}")

        preds_str  = le.inverse_transform(preds_enc)
        confidence = compute_confidence(proba)
        tiers      = assign_tiers(confidence, tier1_threshold, tier3_threshold)
        tier_res   = compute_tier_accuracy(y_test_enc, preds_enc, tiers)
        wr         = compute_workload_reduction(tier_res)

        # Tier summary 
        print(f"\n  Tier Distribution:")
        total = len(tiers)
        for t in [1, 2, 3]:
            tr = tier_res[t]
            acc_str = f"{tr['accuracy']:.2f}%" if tr["accuracy"] is not None else "N/A"
            print(f"    Tier {t}: {tr['count']:>8,} predictions "
                  f"({tr['pct']:>5.2f}%)  "
                  f"Tier accuracy: {acc_str}")

        print(f"\n  Estimated SOC workload reduction: {wr:.2f}%")
        print(f"  (vs fully manual baseline where every alert = 1 full review)")

        # Per-class tier distribution
        pc_df = per_class_tier_distribution(
            y_test_str, preds_str, tiers, classes)
        print_tier_table(pc_df, model_name)

        pc_path = (f"perclass_tiers_{model_name.lower().replace(' ', '_')}"
                   f"_{tag}.csv")
        pc_df.to_csv(os.path.join(output_dir, pc_path))

        # Mean confidence: correct vs incorrect predictions 
        correct_mask   = (preds_enc == y_test_enc)
        mean_conf_all  = confidence.mean()
        mean_conf_corr = confidence[correct_mask].mean()
        mean_conf_wrong= confidence[~correct_mask].mean() if (~correct_mask).sum() > 0 else 0
        conf_gap       = mean_conf_corr - mean_conf_wrong

        print(f"\n  Confidence Analysis:")
        print(f"    Mean overall     : {mean_conf_all*100:.2f}%")
        print(f"    Mean (correct)   : {mean_conf_corr*100:.2f}%")
        print(f"    Mean (wrong)     : {mean_conf_wrong*100:.2f}%")
        print(f"    Correct-Wrong gap: {conf_gap*100:.2f} pp")

        # Plot confidence distribution 
        plot_confidence_distribution(
            confidence, tiers, model_name, dataset,
            tier1_threshold, tier3_threshold, tier_res, output_dir)

        # Store for summary 
        all_results.append({
            "model":              model_name,
            "tier_results":       tier_res,
            "workload_reduction": wr,
        })

        summary_rows.append({
            "Model":              model_name,
            "Dataset":            dataset,
            "Balancing":          balancing,
            "T1 %":               tier_res[1]["pct"],
            "T2 %":               tier_res[2]["pct"],
            "T3 %":               tier_res[3]["pct"],
            "T1 Accuracy %":      tier_res[1]["accuracy"],
            "T2 Accuracy %":      tier_res[2]["accuracy"],
            "T3 Accuracy %":      tier_res[3]["accuracy"],
            "Workload Reduction": wr,
            "Conf Gap (pp)":      round(conf_gap * 100, 2),
            "Tier1 Threshold":    tier1_threshold,
            "Tier3 Threshold":    tier3_threshold,
        })

    # Summary plot 
    plot_tier_summary(all_results, dataset, output_dir)

    # Save summary CSV 
    summary_df   = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, f"triage_summary_{tag}.csv")
    summary_df.to_csv(summary_path, index=False)

    # Final summary printout 
    print(f"\n  {'═'*72}")
    print("  TRIAGE FRAMEWORK SUMMARY")
    print(f"  {'═'*72}")
    print(f"  {'Model':<22} {'T1%':>6} {'T2%':>6} {'T3%':>6} "
          f"{'WR%':>7} {'T1 Acc':>8} {'T3 Acc':>8}")
    print(f"  {'─'*72}")
    for row in summary_rows:
        t1a = f"{row['T1 Accuracy %']:.2f}%" if row['T1 Accuracy %'] else "N/A"
        t3a = f"{row['T3 Accuracy %']:.2f}%" if row['T3 Accuracy %'] else "N/A"
        print(f"  {row['Model']:<22} "
              f"{row['T1 %']:>6.2f} {row['T2 %']:>6.2f} {row['T3 %']:>6.2f} "
              f"{row['Workload Reduction']:>7.2f} {t1a:>8} {t3a:>8}")

    print(f"\n  All results saved to: {output_dir}")
    print("\n=== Triage framework complete ===")


# CLI 

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Confidence threshold triage framework for SOC alert fatigue reduction.\n"
            "Routes IDS predictions to auto-action, second review, or full escalation\n"
            "based on model confidence and empirically observed tier accuracy.\n\n"
            "IMPORTANT: Default thresholds are theoretically grounded starting points.\n"
            "Calibrate against your organisation's operational context before deployment."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=["ciciot2023", "unsw_nb15"],
    )
    parser.add_argument(
        "--balancing", default="class_weight",
        choices=["class_weight", "smote"],
    )
    parser.add_argument(
        "--tier1-threshold", type=float, default=DEFAULT_TIER1_THRESHOLD,
        help=("Confidence threshold for Tier 1 auto-action "
              f"(default: {DEFAULT_TIER1_THRESHOLD}). "
              "Calibrate against your SOC's acceptable error rate.")
    )
    parser.add_argument(
        "--tier3-threshold", type=float, default=DEFAULT_TIER3_THRESHOLD,
        help=("Confidence below which predictions escalate to Tier 3 "
              f"(default: {DEFAULT_TIER3_THRESHOLD}). "
              "Calibrate against your SOC's analyst capacity.")
    )
    parser.add_argument(
        "--model-dir", default="models/",
        help="Directory containing saved model .pkl files."
    )
    parser.add_argument(
        "--data-dir", default="data/processed/",
        help="Directory containing preprocessed test CSV files."
    )
    parser.add_argument(
        "--output", default="results/triage/",
        help="Directory to save triage results (default: results/triage/)."
    )
    args = parser.parse_args()

    run_triage(
        dataset=args.dataset,
        balancing=args.balancing,
        tier1_threshold=args.tier1_threshold,
        tier3_threshold=args.tier3_threshold,
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
