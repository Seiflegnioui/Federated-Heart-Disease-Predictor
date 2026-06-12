"""
data_preparation.py
===================
Task 1 – Data Preparation & Hospital Simulation
Federated Heart Disease Project

Steps:
  1. Load heart.csv
  2. Report basic statistics
  3. Handle missing values (none expected, but we check)
  4. One-hot encode categorical columns
  5. StandardScaler on continuous columns
  6. Save X.npy, y.npy, feature_names.txt to outputs/processed/
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "heart.csv")
OUT_DIR    = os.path.join(BASE_DIR, "outputs", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Column definitions ─────────────────────────────────────────────────────────
# Categorical: need one-hot encoding
CATEGORICAL_COLS = ["cp", "restecg", "slope", "thal"]

# Continuous: need StandardScaler
CONTINUOUS_COLS  = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Binary (0/1): left as-is
BINARY_COLS      = ["sex", "fbs", "exang", "ca"]

TARGET_COL       = "target"


def load_data(path: str) -> pd.DataFrame:
    """Load CSV and perform basic validation."""
    df = pd.read_csv(path)
    print(f"\n{'='*60}")
    print(f"  Heart Disease Dataset — Loaded")
    print(f"{'='*60}")
    print(f"  Shape          : {df.shape[0]} samples × {df.shape[1]} columns")
    print(f"  Missing values : {df.isnull().sum().sum()}")
    print(f"  Target counts  :")
    vc = df[TARGET_COL].value_counts()
    for cls, cnt in vc.items():
        label = "Sick (1)" if cls == 1 else "Healthy (0)"
        print(f"    {label} -> {cnt} ({cnt/len(df)*100:.1f}%)")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values if any.
    - Continuous → median
    - Continuous -> median
    - Categorical / binary -> mode
    """
    missing = df.isnull().sum()
    cols_with_missing = missing[missing > 0].index.tolist()

    if not cols_with_missing:
        print(f"  [OK] No missing values detected - skipping imputation.")
        return df

    print(f"  [WARN] Missing values found in: {cols_with_missing}")
    for col in cols_with_missing:
        if col in CONTINUOUS_COLS:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"    {col}: filled with median={df[col].median():.2f}")
        else:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"    {col}: filled with mode={mode_val}")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns (drop_first=False to keep all info)."""
    print("\n  Encoding categorical columns:")
    for col in CATEGORICAL_COLS:
        n_unique = df[col].nunique()
        print(f"    {col}: {n_unique} unique values -> one-hot encoded")
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False, dtype=int)
    return df


def normalize_continuous(df: pd.DataFrame, scaler: StandardScaler = None):
    """
    Apply StandardScaler to continuous columns.
    Returns (df, fitted_scaler).
    """
    print("\n  Normalizing continuous columns:")
    for col in CONTINUOUS_COLS:
        print(f"    {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}")

    if scaler is None:
        scaler = StandardScaler()
        df[CONTINUOUS_COLS] = scaler.fit_transform(df[CONTINUOUS_COLS])
    else:
        df[CONTINUOUS_COLS] = scaler.transform(df[CONTINUOUS_COLS])

    print("  [OK] StandardScaler applied (mean=0, std=1)")
    return df, scaler


def prepare_dataset(save: bool = True):
    """
    Full preparation pipeline.

    Returns
    -------
    X : np.ndarray  shape (n_samples, n_features)
    y : np.ndarray  shape (n_samples,)
    feature_names : list[str]
    scaler : StandardScaler  (fitted on this dataset)
    """
    # 1. Load
    df = load_data(DATA_PATH)

    # 2. Handle missing values
    df = handle_missing(df)

    # 3. Separate target before encoding
    y = df[TARGET_COL].values
    df = df.drop(columns=[TARGET_COL])

    # 4. One-hot encode categoricals
    df = encode_categoricals(df)

    # 5. Normalize continuous features
    df, scaler = normalize_continuous(df)

    # 6. Build feature matrix
    X = df.values.astype(np.float32)
    feature_names = list(df.columns)

    print(f"\n  Final feature matrix : {X.shape[0]} samples x {X.shape[1]} features")
    print(f"  Features             : {feature_names}")

    # 7. Save
    if save:
        np.save(os.path.join(OUT_DIR, "X.npy"), X)
        np.save(os.path.join(OUT_DIR, "y.npy"), y)
        with open(os.path.join(OUT_DIR, "feature_names.txt"), "w") as f:
            f.write("\n".join(feature_names))
        joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))
        print(f"  [OK] Saved to {OUT_DIR}")
        print(f"    X.npy  -> {X.shape}")
        print(f"    y.npy  -> {y.shape}")
        print(f"    feature_names.txt -> {len(feature_names)} features")

    print(f"\n{'='*60}\n")
    return X, y, feature_names, scaler


if __name__ == "__main__":
    prepare_dataset(save=True)
