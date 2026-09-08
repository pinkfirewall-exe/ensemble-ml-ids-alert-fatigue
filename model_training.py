"""
model_training.py
-----------------
Trains Random Forest, XGBoost, and a soft Voting Classifier on preprocessed
CICIoT2023 or UNSW-NB15 data using class weighting or SMOTE for imbalance handling.

Features:
  - Five-fold stratified cross-validation with mean +/- std reporting
  - Class weighting (primary strategy, works at any scale)
  - SMOTE (optional, UNSW-NB15 only -- CICIoT2023 is too large)
  - Soft Voting Classifier averaging RF and XGBoost probability outputs
  - Saved models for downstream evaluation, triage, and SHAP analysis

Usage:
  python model_training.py --dataset ciciot2023 --balancing class_weight
  python model_training.py --dataset unsw_nb15  --balancing class_weight
  python model_training.py --dataset unsw_nb15  --balancing smote

  For CICIoT2023 cross-validation uses a stratified 20% subsample due to memory constraints. Final models are trained on the full training set.

Random seed: 42 (fixed throughout)
"""

import argparse
import os
import pickle
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, accuracy_score

RANDOM_SEED  = 42
CV_FOLDS     = 5
N_ESTIMATORS = 100
CICIOT_CV_SUBSAMPLE = 0.20   # 20% subsample for CICIoT2023 cross-validation
SMOTE_K_NEIGHBOURS  = 5


# Model builders 

def build_rf(class_weight="balanced"):
    """Random Forest with class weighting."""
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=RANDOM_SEED
    )


def build_xgb(num_classes: int):
    """
    XGBoost with multi-class log-loss objective.
    Class weighting is applied via sample_weight at fit time.
    """
    return XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        use_label_encoder=False,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbosity=0
    )


# Cross-validation

def run_cross_validation(X: np.ndarray, y: np.ndarray,
                         model, model_name: str,
                         sample_weight=None,
                         is_large_dataset: bool = False):
    """
    Five-fold stratified cross-validation.
    For large datasets (CICIoT2023) a stratified 20% subsample is used.
    """
    if is_large_dataset:
        print(f"  [{model_name}] Subsampling {CICIOT_CV_SUBSAMPLE*100:.0f}% "
              f"of training data for cross-validation ...")
        from sklearn.model_selection import train_test_split
        X_cv, _, y_cv, _ = train_test_split(
            X, y,
            train_size=CICIOT_CV_SUBSAMPLE,
            stratify=y,
            random_state=RANDOM_SEED
        )
        if sample_weight is not None:
            sw_cv = compute_sample_weight("balanced", y_cv)
        else:
            sw_cv = None
    else:
        X_cv, y_cv = X, y
        sw_cv = sample_weight

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                          random_state=RANDOM_SEED)

    cv_acc, cv_f1 = [], []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_cv, y_cv), 1):
        X_tr, X_val = X_cv[train_idx], X_cv[val_idx]
        y_tr, y_val = y_cv[train_idx], y_cv[val_idx]

        sw_fold = sw_cv[train_idx] if sw_cv is not None else None

        if sw_fold is not None:
            model.fit(X_tr, y_tr, sample_weight=sw_fold)
        else:
            model.fit(X_tr, y_tr)

        preds = model.predict(X_val)
        cv_acc.append(accuracy_score(y_val, preds))
        cv_f1.append(f1_score(y_val, preds, average="macro", zero_division=0))
        print(f"    Fold {fold}/{CV_FOLDS}  acc={cv_acc[-1]*100:.2f}%  "
              f"macro_f1={cv_f1[-1]*100:.2f}%")

    print(f"  [{model_name}] CV Accuracy : "
          f"{np.mean(cv_acc)*100:.2f}% +/- {np.std(cv_acc)*100:.2f}%")
    print(f"  [{model_name}] CV Macro F1 : "
          f"{np.mean(cv_f1)*100:.2f}% +/- {np.std(cv_f1)*100:.2f}%")

    return np.mean(cv_acc), np.mean(cv_f1)


#Training pipeline 

def train_models(dataset: str, balancing: str,
                 data_dir: str, output_dir: str):

    print(f"\n=== Model Training ===")
    print(f"  Dataset   : {dataset}")
    print(f"  Balancing : {balancing}\n")

    # Load preprocessed data 
    prefix = "ciciot2023" if dataset == "ciciot2023" else "unsw_nb15"
    X_train = pd.read_csv(
        os.path.join(data_dir, f"{prefix}_train_X.csv")).values
    y_train = pd.read_csv(
        os.path.join(data_dir, f"{prefix}_train_y.csv")).squeeze().values
    X_test  = pd.read_csv(
        os.path.join(data_dir, f"{prefix}_test_X.csv")).values
    y_test  = pd.read_csv(
        os.path.join(data_dir, f"{prefix}_test_y.csv")).squeeze().values

    print(f"  Training set : {X_train.shape[0]:,} rows, "
          f"{X_train.shape[1]} features")
    print(f"  Test set     : {X_test.shape[0]:,} rows\n")

    # Encode string labels to integers for XGBoost
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)
    num_classes = len(le.classes_)
    print(f"  Classes ({num_classes}): {list(le.classes_)}\n")

    is_large = (dataset == "ciciot2023")

    # Apply SMOTE if requested 
    if balancing == "smote":
        if dataset == "ciciot2023":
            print("  WARNING: SMOTE is not supported for CICIoT2023 "
                  "(dataset too large for available memory).")
            print("  Falling back to class weighting.")
            balancing = "class_weight"
        else:
            print(f"  Applying SMOTE (k={SMOTE_K_NEIGHBOURS}) ...")
            t0 = time.time()
            sm = SMOTE(k_neighbors=SMOTE_K_NEIGHBOURS,
                       random_state=RANDOM_SEED, n_jobs=-1)
            X_train, y_train_enc = sm.fit_resample(X_train, y_train_enc)
            print(f"  SMOTE complete in {time.time()-t0:.1f}s. "
                  f"New training size: {X_train.shape[0]:,} rows.")
            # No class weighting needed after SMOTE
            sample_weight_train = None
    
    # Compute sample weights for class weighting
    if balancing == "class_weight":
        sample_weight_train = compute_sample_weight("balanced", y_train_enc)
    else:
        sample_weight_train = None

    #  RF
    print("─" * 50)
    print("  Random Forest")
    print("─" * 50)
    rf_cw = "balanced" if balancing == "class_weight" else None
    rf = build_rf(class_weight=rf_cw)

    print("  Cross-validation ...")
    run_cross_validation(X_train, y_train_enc, rf, "RF",
                         sample_weight=None,   # RF uses class_weight param
                         is_large_dataset=is_large)

    print("  Training on full training set ...")
    t0 = time.time()
    rf.fit(X_train, y_train_enc)
    print(f"  RF training complete in {time.time()-t0:.1f}s.")

    rf_preds = rf.predict(X_test)
    rf_acc   = accuracy_score(y_test_enc, rf_preds)
    rf_f1    = f1_score(y_test_enc, rf_preds, average="macro", zero_division=0)
    print(f"  Test Accuracy : {rf_acc*100:.2f}%")
    print(f"  Test Macro F1 : {rf_f1*100:.2f}%")

    # XGBoost
    print("\n" + "─" * 50)
    print("  XGBoost")
    print("─" * 50)
    xgb = build_xgb(num_classes)

    print("  Cross-validation ...")
    run_cross_validation(X_train, y_train_enc, xgb, "XGB",
                         sample_weight=sample_weight_train,
                         is_large_dataset=is_large)

    print("  Training on full training set ...")
    t0 = time.time()
    if sample_weight_train is not None:
        xgb.fit(X_train, y_train_enc, sample_weight=sample_weight_train)
    else:
        xgb.fit(X_train, y_train_enc)
    print(f"  XGBoost training complete in {time.time()-t0:.1f}s.")

    xgb_preds = xgb.predict(X_test)
    xgb_acc   = accuracy_score(y_test_enc, xgb_preds)
    xgb_f1    = f1_score(y_test_enc, xgb_preds, average="macro", zero_division=0)
    print(f"  Test Accuracy : {xgb_acc*100:.2f}%")
    print(f"  Test Macro F1 : {xgb_f1*100:.2f}%")

    # Voting Classifier
    # Implemented by averaging RF and XGBoost probability vectors directly,
    # rather than using sklearn's VotingClassifier wrapper, to avoid the double training cost on large datasets.
    print("\n" + "─" * 50)
    print("  Voting Classifier (soft vote: RF + XGBoost average)")
    print("─" * 50)

    rf_probs  = rf.predict_proba(X_test)
    xgb_probs = xgb.predict_proba(X_test)
    vote_probs = (rf_probs + xgb_probs) / 2.0
    vote_preds = np.argmax(vote_probs, axis=1)

    vote_acc = accuracy_score(y_test_enc, vote_preds)
    vote_f1  = f1_score(y_test_enc, vote_preds, average="macro", zero_division=0)
    print(f"  Test Accuracy : {vote_acc*100:.2f}%")
    print(f"  Test Macro F1 : {vote_f1*100:.2f}%")

    # Summary
    print("\n" + "=" * 50)
    print("  RESULTS SUMMARY")
    print("=" * 50)
    print(f"  {'Model':<22} {'Accuracy':>10}  {'Macro F1':>10}")
    print(f"  {'-'*44}")
    print(f"  {'Random Forest':<22} {rf_acc*100:>9.2f}%  {rf_f1*100:>9.2f}%")
    print(f"  {'XGBoost':<22} {xgb_acc*100:>9.2f}%  {xgb_f1*100:>9.2f}%")
    print(f"  {'Voting Classifier':<22} {vote_acc*100:>9.2f}%  {vote_f1*100:>9.2f}%")

    # Save models
    os.makedirs(output_dir, exist_ok=True)
    tag = f"{prefix}_{balancing}"

    _save_model(rf,         output_dir, f"rf_{tag}.pkl")
    _save_model(xgb,        output_dir, f"xgb_{tag}.pkl")
    _save_model(le,         output_dir, f"label_encoder_{prefix}.pkl")

    # Save Voting Classifier probabilities for downstream use
    probs_path = os.path.join(output_dir, f"vote_probs_{tag}.npy")
    np.save(probs_path, vote_probs)
    print(f"\n  Saved Voting Classifier probabilities: {probs_path}")

    # Save test labels (encoded) for evaluation scripts
    np.save(os.path.join(output_dir, f"y_test_enc_{prefix}.npy"), y_test_enc)
    np.save(os.path.join(output_dir, f"y_test_str_{prefix}.npy"), y_test)

    print(f"\n  All models saved to: {output_dir}")
    print("\n=== Training complete ===")


def _save_model(obj, output_dir: str, filename: str):
    path = os.path.join(output_dir, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved: {path}")


#CLI 
def main():
    parser = argparse.ArgumentParser(
        description="Train RF, XGBoost, and Voting Classifier on IDS benchmark data."
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=["ciciot2023", "unsw_nb15"],
        help="Which dataset to train on."
    )
    parser.add_argument(
        "--balancing", default="class_weight",
        choices=["class_weight", "smote"],
        help="Imbalance handling strategy (default: class_weight)."
    )
    parser.add_argument(
        "--data-dir", default="data/processed/",
        help="Directory containing preprocessed CSV files."
    )
    parser.add_argument(
        "--output", default="models/",
        help="Directory to save trained models (default: models/)."
    )
    args = parser.parse_args()

    train_models(
        dataset=args.dataset,
        balancing=args.balancing,
        data_dir=args.data_dir,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
