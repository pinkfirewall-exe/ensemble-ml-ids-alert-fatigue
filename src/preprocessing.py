"""
preprocessing.py
----------------
Data cleaning, feature encoding, and Min-Max scaling for CICIoT2023 and UNSW-NB15.

Handles:
  - Constant feature removal (CICIoT2023)
  - Exact duplicate removal
  - Label mapping (34 raw CICIoT2023 labels -> 8 operational categories)
  - One-hot encoding of categorical features (UNSW-NB15)
  - Rare protocol grouping (UNSW-NB15)
  - Min-Max scaling fitted on training partition only (no data leakage)

Usage:
  python preprocessing.py --dataset ciciot2023 --train data/raw/ciciot_train.csv
                           --test data/raw/ciciot_test.csv --output data/processed/

  python preprocessing.py --dataset unsw_nb15 --train data/raw/unsw_train.csv
                           --test data/raw/unsw_test.csv --output data/processed/

Random seed: 42 (fixed throughout for reproducibility)
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 42

# CICIoT2023 label mapping
# Maps 34 raw attack type labels to 8 operational categories.
CICIOT_LABEL_MAP = {
    # DDoS
    "DDoS-ICMP_Flood":            "DDoS",
    "DDoS-UDP_Flood":             "DDoS",
    "DDoS-TCP_Flood":             "DDoS",
    "DDoS-PSHACK_Flood":          "DDoS",
    "DDoS-SYN_Flood":             "DDoS",
    "DDoS-RSTFINFlood":           "DDoS",
    "DDoS-SynonymousIP_Flood":    "DDoS",
    "DDoS-ICMP_Fragmentation":    "DDoS",
    "DDoS-UDP_Fragmentation":     "DDoS",
    "DDoS-ACK_Fragmentation":     "DDoS",
    "DDoS-HTTP_Flood":            "DDoS",
    "DDoS-SlowLoris":             "DDoS",
    # DoS
    "DoS-UDP_Flood":              "DoS",
    "DoS-TCP_Flood":              "DoS",
    "DoS-SYN_Flood":              "DoS",
    "DoS-HTTP_Flood":             "DoS",
    # Mirai
    "Mirai-greeth_flood":         "Mirai",
    "Mirai-udpplain":             "Mirai",
    "Mirai-greip_flood":          "Mirai",
    # Recon
    "Recon-HostDiscovery":        "Recon",
    "Recon-OSScan":               "Recon",
    "Recon-PortScan":             "Recon",
    "Recon-PingSweep":            "Recon",
    "VulnerabilityScan":          "Recon",
    # Spoofing
    "MITM-ArpSpoofing":           "Spoofing",
    "DNS_Spoofing":               "Spoofing",
    # Web
    "BrowserHijacking":           "Web",
    "CommandInjection":           "Web",
    "SqlInjection":               "Web",
    "XSS":                        "Web",
    "Backdoor_Malware":           "Web",
    "Uploading_Attack":           "Web",
    # BruteForce
    "DictionaryBruteForce":       "BruteForce",
    # Benign
    "BenignTraffic":              "Benign",
}

# Constant features to drop from CICIoT2023
CICIOT_CONSTANT_FEATURES = ["Telnet", "IRC"]

# Categorical features to one-hot encode in UNSW-NB15
UNSW_CATEGORICAL_FEATURES = ["proto", "service", "state"]

# Minimum occurrence threshold for UNSW-NB15 protocol grouping
UNSW_PROTO_MIN_COUNT = 50


def load_csv(path: str, dataset: str) -> pd.DataFrame:
    """Load a CSV file with appropriate dtype hints."""
    print(f"  Loading {path} ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns.")
    return df


# CICIoT2023 preprocessing

def preprocess_ciciot(train_path: str, test_path: str, output_dir: str):
    """Full preprocessing pipeline for CICIoT2023."""

    print("\n=== CICIoT2023 Preprocessing ===\n")

    train = load_csv(train_path, "ciciot2023")
    test  = load_csv(test_path,  "ciciot2023")

    # 1. Remove constant features
    cols_to_drop = [c for c in CICIOT_CONSTANT_FEATURES if c in train.columns]
    if cols_to_drop:
        train.drop(columns=cols_to_drop, inplace=True)
        test.drop(columns=cols_to_drop, inplace=True)
        print(f"  Dropped constant features: {cols_to_drop}")

    # 2. Deduplicate 
    before = len(train)
    train.drop_duplicates(inplace=True)
    print(f"  Train deduplication: {before:,} -> {len(train):,} "
          f"(-{before - len(train):,} rows, "
          f"{(before - len(train)) / before * 100:.2f}%)")

    before = len(test)
    test.drop_duplicates(inplace=True)
    print(f"  Test  deduplication: {before:,} -> {len(test):,} "
          f"(-{before - len(test):,} rows)")

    # 3. Map labels
    label_col = _detect_label_column(train)
    print(f"  Label column: '{label_col}'")

    train[label_col] = train[label_col].map(CICIOT_LABEL_MAP)
    test[label_col]  = test[label_col].map(CICIOT_LABEL_MAP)

    unmapped = train[label_col].isna().sum()
    if unmapped > 0:
        print(f"  WARNING: {unmapped:,} unmapped labels in training set — dropping.")
        train.dropna(subset=[label_col], inplace=True)

    print(f"  Class distribution (train):")
    _print_class_distribution(train, label_col)

    # 4. Separate features and labels
    X_train = train.drop(columns=[label_col])
    y_train = train[label_col]
    X_test  = test.drop(columns=[label_col])
    y_test  = test[label_col]

    # 5. Min-Max scaling (fit on train only)
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    print(f"  Min-Max scaling applied. Feature range: [0, 1]")

    # 6. Save outputs
    os.makedirs(output_dir, exist_ok=True)
    _save(X_train_scaled, y_train, output_dir, "ciciot2023_train")
    _save(X_test_scaled,  y_test,  output_dir, "ciciot2023_test")

    print(f"\n  Saved processed files to: {output_dir}")
    print(f"  Training: {len(X_train_scaled):,} rows, {X_train_scaled.shape[1]} features")
    print(f"  Test:     {len(X_test_scaled):,} rows, {X_test_scaled.shape[1]} features")
    print("\n=== CICIoT2023 preprocessing complete ===")


# UNSW-NB15 preprocessing

def preprocess_unsw(train_path: str, test_path: str, output_dir: str):
    """Full preprocessing pipeline for UNSW-NB15."""

    print("\n=== UNSW-NB15 Preprocessing ===\n")

    train = load_csv(train_path, "unsw_nb15")
    test  = load_csv(test_path,  "unsw_nb15")

    #1. Deduplicate
    before = len(train)
    train.drop_duplicates(inplace=True)
    print(f"  Train deduplication: {before:,} -> {len(train):,} "
          f"(-{before - len(train):,} rows, "
          f"{(before - len(train)) / before * 100:.2f}%)")

    before = len(test)
    test.drop_duplicates(inplace=True)
    print(f"  Test  deduplication: {before:,} -> {len(test):,}")

    # 2. Detect label column
    label_col = _detect_label_column(train)
    print(f"  Label column: '{label_col}'")

    print(f"  Class distribution (train):")
    _print_class_distribution(train, label_col)

    # 3. Group rare protocols
    # Protocols appearing < UNSW_PROTO_MIN_COUNT times in training are grouped
    # into 'Other'. Grouping derived from training only, applied to test.
    if "proto" in train.columns:
        proto_counts = train["proto"].value_counts()
        rare_protos  = proto_counts[proto_counts < UNSW_PROTO_MIN_COUNT].index.tolist()
        train["proto"] = train["proto"].apply(
            lambda x: "Other" if x in rare_protos else x
        )
        test["proto"]  = test["proto"].apply(
            lambda x: "Other" if x in rare_protos else x
        )
        print(f"  Grouped {len(rare_protos)} rare protocols into 'Other' "
              f"(threshold: {UNSW_PROTO_MIN_COUNT} occurrences).")

    # 4. One-hot encode categorical features (fit on train only)
    cat_cols = [c for c in UNSW_CATEGORICAL_FEATURES if c in train.columns]
    print(f"  One-hot encoding: {cat_cols}")

    train_encoded = pd.get_dummies(train, columns=cat_cols)
    test_encoded  = pd.get_dummies(test,  columns=cat_cols)

    # Align columns — test may be missing some dummies from rare training values
    train_encoded, test_encoded = train_encoded.align(
        test_encoded, join="left", axis=1, fill_value=0
    )
    print(f"  After encoding: {train_encoded.shape[1]} columns "
          f"(train), {test_encoded.shape[1]} columns (test — aligned to train).")

    # 5. Separate features and labels
    X_train = train_encoded.drop(columns=[label_col])
    y_train = train_encoded[label_col]
    X_test  = test_encoded.drop(columns=[label_col])
    y_test  = test_encoded[label_col]

    # 6. Min-Max scaling (fit on train only)
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    print(f"  Min-Max scaling applied. Feature range: [0, 1]")

    # 7. Save outputs
    os.makedirs(output_dir, exist_ok=True)
    _save(X_train_scaled, y_train, output_dir, "unsw_nb15_train")
    _save(X_test_scaled,  y_test,  output_dir, "unsw_nb15_test")

    print(f"\n  Saved processed files to: {output_dir}")
    print(f"  Training: {len(X_train_scaled):,} rows, {X_train_scaled.shape[1]} features")
    print(f"  Test:     {len(X_test_scaled):,} rows, {X_test_scaled.shape[1]} features")
    print("\n=== UNSW-NB15 preprocessing complete ===")


# Helpers 

def _detect_label_column(df: pd.DataFrame) -> str:
    """Detect the label column by checking common names."""
    candidates = ["label", "Label", "attack_cat", "attack_type",
                  "class", "Class", "category"]
    for c in candidates:
        if c in df.columns:
            return c
    # Fall back to last column
    return df.columns[-1]


def _print_class_distribution(df: pd.DataFrame, label_col: str):
    counts = df[label_col].value_counts()
    total  = len(df)
    min_c  = counts.min()
    for cls, cnt in counts.items():
        ratio = cnt / min_c
        print(f"    {cls:<20} {cnt:>10,}  ({cnt/total*100:5.2f}%)  "
              f"ratio: {ratio:,.0f}:1")


def _save(X: pd.DataFrame, y: pd.Series, output_dir: str, prefix: str):
    """Save features and labels as separate CSV files."""
    X_path = os.path.join(output_dir, f"{prefix}_X.csv")
    y_path = os.path.join(output_dir, f"{prefix}_y.csv")
    X.to_csv(X_path, index=False)
    y.to_csv(y_path, index=False, header=True)
    print(f"  Saved: {X_path}")
    print(f"  Saved: {y_path}")


# CLI 

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess CICIoT2023 or UNSW-NB15 for IDS experiments."
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=["ciciot2023", "unsw_nb15"],
        help="Which dataset to preprocess."
    )
    parser.add_argument(
        "--train", required=True,
        help="Path to the raw training CSV file."
    )
    parser.add_argument(
        "--test", required=True,
        help="Path to the raw test CSV file."
    )
    parser.add_argument(
        "--output", default="data/processed/",
        help="Directory to write processed files (default: data/processed/)."
    )
    args = parser.parse_args()

    if args.dataset == "ciciot2023":
        preprocess_ciciot(args.train, args.test, args.output)
    else:
        preprocess_unsw(args.train, args.test, args.output)


if __name__ == "__main__":
    main()
