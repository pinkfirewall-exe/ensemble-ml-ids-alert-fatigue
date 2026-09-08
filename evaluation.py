"""
evaluation.py
-------------
Per-class evaluation for trained IDS ensemble classifiers.

Produces:
  - Per-class Precision, Recall, F1-score for every attack category
  - Per-class False Positive Rate (FPR) and False Negative Rate (FNR)
  - Macro-averaged F1 and overall accuracy
  - Confusion matrix (saved as PNG)
  - Binary classification results (all attacks vs normal/benign)
  - Summary CSV for downstream analysis

Usage:
  python evaluation.py --dataset ciciot2023 --balancing class_weight
  python evaluation.py --dataset unsw_nb15  --balancing class_weight
  python evaluation.py --dataset unsw_nb15  --balancing smote

Why per-class FPR and FNR matter:
  Aggregate accuracy is dominated by majority classes. A 99.6% accurate
  classifier on CICIoT2023 simultaneously misses 61% of Web attacks.
  Per-class FPR and FNR are the metrics SOC deployment decisions require.

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
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

RANDOM_SEED = 42


# ─── Load helpers ─────────────────────────────────────────────────────────────

def load_artifacts(dataset: str, balancing: str,
                   model_dir: str, data_dir: str):
    """Load trained models, test features, and labels."""
    prefix = "ciciot2023" if dataset == "ciciot2023" else "unsw_nb15"
    tag    = f"{prefix}_{balancing}"

    with open(os.path.join(model_dir, f"rf_{tag}.pkl"),  "rb") as f:
        rf = pickle.load(f)
    with open(os.path.join(model_dir, f"xgb_{tag}.pkl"), "rb") as f:
        xgb = pickle.load(f)
    with open(os.path.join(model_dir, f"label_encoder_{prefix}.pkl"), "rb") as f:
        le = pickle.load(f)

    vote_probs = np.load(
        os.path.join(model_dir, f"vote_probs_{tag}.npy"))
    y_test_enc = np.load(
        os.path.join(model_dir, f"y_test_enc_{prefix}.npy"))
    y_test_str = np.load(
        os.path.join(model_dir, f"y_test_str_{prefix}.npy"), allow_pickle=True)

    X_test = pd.read_csv(
        os.path.join(data_dir, f"{prefix}_test_X.csv")).values

    return rf, xgb, le, vote_probs, y_test_enc, y_test_str, X_test


# ─── Per-class FPR / FNR ──────────────────────────────────────────────────────

def per_class_fpr_fnr(y_true: np.ndarray, y_pred: np.ndarray,
                      classes: list) -> pd.DataFrame:
    """
    Compute per-class FPR and FNR from a multi-class confusion matrix.

    For class i:
      TP_i = cm[i, i]
      FN_i = row_sum_i - TP_i          (missed positives)
      FP_i = col_sum_i - TP_i          (false alarms)
      TN_i = total - TP_i - FN_i - FP_i

      FPR_i = FP_i / (FP_i + TN_i)    (false alarm rate)
      FNR_i = FN_i / (FN_i + TP_i)    (missed detection rate)
    """
    cm       = confusion_matrix(y_true, y_pred)
    n        = cm.sum()
    rows     = []

    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp   # missed in row i
        fp = cm[:, i].sum() - tp   # false alarms in column i
        tn = n - tp - fn - fp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr       = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        rows.append({
            "Class":     cls,
            "TP":        int(tp),
            "FP":        int(fp),
            "FN":        int(fn),
            "TN":        int(tn),
            "Precision": round(precision * 100, 2),
            "Recall":    round(recall    * 100, 2),
            "F1":        round(f1        * 100, 2),
            "FPR":       round(fpr       * 100, 2),
            "FNR":       round(fnr       * 100, 2),
        })

    return pd.DataFrame(rows).set_index("Class")


def print_per_class_table(df: pd.DataFrame, model_name: str):
    """Pretty-print the per-class evaluation table."""
    print(f"\n  {'─'*70}")
    print(f"  {model_name} — Per-Class Results")
    print(f"  {'─'*70}")
    print(f"  {'Class':<18} {'P%':>7} {'R%':>7} {'F1%':>7} "
          f"{'FPR%':>7} {'FNR%':>7}")
    print(f"  {'─'*70}")
    for cls, row in df.iterrows():
        print(f"  {cls:<18} {row['Precision']:>7.2f} {row['Recall']:>7.2f} "
              f"{row['F1']:>7.2f} {row['FPR']:>7.2f} {row['FNR']:>7.2f}")
    print(f"  {'─'*70}")
    macro_f1 = df["F1"].mean()
    print(f"  {'Macro F1':<18} {macro_f1:>7.2f}%")


# ─── Confusion matrix plot ────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          classes: list, model_name: str,
                          dataset: str, output_dir: str):
    """Save a normalised confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        ax=ax, cbar_kws={"label": "Proportion"}
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"{model_name} Confusion Matrix — {dataset}")
    plt.tight_layout()

    fname = f"cm_{model_name.lower().replace(' ', '_')}_{dataset}.png"
    path  = os.path.join(output_dir, fname)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix: {path}")


# ─── Binary classification ────────────────────────────────────────────────────

def binary_results(y_true_str: np.ndarray, y_pred_str: np.ndarray,
                   benign_label: str, model_name: str) -> dict:
    """
    Collapse multi-class predictions to binary (benign vs attack)
    and compute accuracy, precision, recall, F1.
    """
    y_true_bin = (y_true_str != benign_label).astype(int)
    y_pred_bin = (y_pred_str != benign_label).astype(int)

    acc  = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec  = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1   = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    print(f"  {model_name} Binary  —  Acc: {acc*100:.2f}%  "
          f"P: {prec*100:.2f}%  R: {rec*100:.2f}%  F1: {f1*100:.2f}%")

    return {
        "Model":              model_name,
        "Binary Accuracy %":  round(acc  * 100, 2),
        "Binary Precision %": round(prec * 100, 2),
        "Binary Recall %":    round(rec  * 100, 2),
        "Binary F1 %":        round(f1   * 100, 2),
    }


# ─── Main evaluation pipeline ─────────────────────────────────────────────────

def evaluate(dataset: str, balancing: str,
             model_dir: str, data_dir: str, output_dir: str):

    print(f"\n=== Evaluation ===")
    print(f"  Dataset   : {dataset}")
    print(f"  Balancing : {balancing}\n")

    # ── Load ──────────────────────────────────────────────────────────────────
    rf, xgb, le, vote_probs, y_test_enc, y_test_str, X_test = \
        load_artifacts(dataset, balancing, model_dir, data_dir)

    classes     = list(le.classes_)
    benign_lbl  = "Benign" if dataset == "ciciot2023" else "Normal"

    # ── Predictions ───────────────────────────────────────────────────────────
    rf_preds_enc   = rf.predict(X_test)
    xgb_preds_enc  = xgb.predict(X_test)
    vote_preds_enc = np.argmax(vote_probs, axis=1)

    rf_preds_str   = le.inverse_transform(rf_preds_enc)
    xgb_preds_str  = le.inverse_transform(xgb_preds_enc)
    vote_preds_str = le.inverse_transform(vote_preds_enc)

    # ── Aggregate accuracy ────────────────────────────────────────────────────
    print("  Overall Accuracy")
    print(f"    RF    : {accuracy_score(y_test_enc, rf_preds_enc)*100:.2f}%")
    print(f"    XGB   : {accuracy_score(y_test_enc, xgb_preds_enc)*100:.2f}%")
    print(f"    Vote  : {accuracy_score(y_test_enc, vote_preds_enc)*100:.2f}%")

    # ── Per-class FPR / FNR ───────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    prefix = "ciciot2023" if dataset == "ciciot2023" else "unsw_nb15"
    tag    = f"{prefix}_{balancing}"

    all_tables = {}
    for model_name, preds_enc, preds_str in [
        ("Random Forest",     rf_preds_enc,   rf_preds_str),
        ("XGBoost",           xgb_preds_enc,  xgb_preds_str),
        ("Voting Classifier", vote_preds_enc, vote_preds_str),
    ]:
        table = per_class_fpr_fnr(y_test_enc, preds_enc, classes)
        print_per_class_table(table, model_name)
        all_tables[model_name] = table

        # Save per-class CSV
        csv_name = (f"perclass_{model_name.lower().replace(' ', '_')}"
                    f"_{tag}.csv")
        table.to_csv(os.path.join(output_dir, csv_name))

        # Confusion matrix
        plot_confusion_matrix(y_test_enc, preds_enc, classes,
                              model_name, dataset, output_dir)

    # ── Binary results ────────────────────────────────────────────────────────
    print(f"\n  {'─'*70}")
    print("  Binary Classification (attack vs benign/normal)")
    print(f"  {'─'*70}")

    binary_rows = []
    for model_name, preds_str in [
        ("Random Forest",     rf_preds_str),
        ("XGBoost",           xgb_preds_str),
        ("Voting Classifier", vote_preds_str),
    ]:
        binary_rows.append(
            binary_results(y_test_str, preds_str, benign_lbl, model_name)
        )

    binary_df = pd.DataFrame(binary_rows).set_index("Model")
    binary_df.to_csv(os.path.join(output_dir, f"binary_results_{tag}.csv"))
    print(f"\n  Saved binary results: binary_results_{tag}.csv")

    # ── Combined summary ──────────────────────────────────────────────────────
    summary_rows = []
    for model_name, table in all_tables.items():
        summary_rows.append({
            "Model":            model_name,
            "Dataset":          dataset,
            "Balancing":        balancing,
            "Test Accuracy %":  round(accuracy_score(
                                    y_test_enc,
                                    rf_preds_enc if "Forest" in model_name
                                    else (xgb_preds_enc if "XGB" in model_name
                                          else vote_preds_enc)
                                ) * 100, 2),
            "Macro F1 %":       round(table["F1"].mean(), 2),
            "Max FNR %":        round(table["FNR"].max(), 2),
            "Max FNR Class":    table["FNR"].idxmax(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, f"summary_{tag}.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\n  {'═'*70}")
    print("  EVALUATION SUMMARY")
    print(f"  {'═'*70}")
    for _, row in summary_df.iterrows():
        print(f"  {row['Model']:<22}  Acc: {row['Test Accuracy %']:>6.2f}%  "
              f"MacroF1: {row['Macro F1 %']:>6.2f}%  "
              f"MaxFNR: {row['Max FNR %']:>6.2f}% ({row['Max FNR Class']})")

    print(f"\n  All results saved to: {output_dir}")
    print("\n=== Evaluation complete ===")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Per-class FPR/FNR evaluation for trained IDS classifiers."
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
        "--model-dir", default="models/",
        help="Directory containing saved model .pkl files."
    )
    parser.add_argument(
        "--data-dir", default="data/processed/",
        help="Directory containing preprocessed test CSV files."
    )
    parser.add_argument(
        "--output", default="results/",
        help="Directory to save evaluation outputs (default: results/)."
    )
    args = parser.parse_args()

    evaluate(
        dataset=args.dataset,
        balancing=args.balancing,
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
