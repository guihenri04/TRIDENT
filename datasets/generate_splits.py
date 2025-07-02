# generate_splits.py 
import pandas as pd
import numpy as np
import os, glob, logging, json
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.preprocessing import LabelEncoder
from typing import Any
from collections import Counter

# ──────────────────────────── CONFIG ──────────────────────────── #
INPUT_DIR       = Path("datasets_raw")            # ← edit here
OUTPUT_DIR      = Path("processed_datasets")  # ← edit here
TARGET_COLUMN   = "class"                     # ← target column name
NAN_PERCENTAGES = [0.20, 0.40, 0.60, 0.80]    # 0% is already saved as _00nan
RANDOM_SEED     = 42                          # global seed
VAL_SIZE        = 0.10                        # fraction of total
TEST_SIZE       = 0.10                        # fraction of total
# ───────────────────────────────────────────────────────────────── #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def inject_nans(df: pd.DataFrame, pct: float, target: str, *, seed: int) -> pd.DataFrame | None:
    """Injects NaNs into pct (%) of rows for each feature."""
    if pct <= 0:
        return df.copy()
    rng = np.random.default_rng(seed)
    df_nan = df.copy()
    features = [c for c in df.columns if c != target]

    n_rows      = len(df_nan)
    n_inject    = int(round(n_rows * pct))

    if n_inject == 0:
        logging.warning(
            f"Insufficient size ({n_rows} rows) to inject {pct*100:.1f}% NaNs."
        )
        return None

    for col in features:
        non_nan_idx = df_nan.index[df_nan[col].notna()]
        n_here = min(n_inject, len(non_nan_idx))
        if n_here:
            to_nan = rng.choice(non_nan_idx, size=n_here, replace=False)
            df_nan.loc[to_nan, col] = np.nan
    return df_nan

# ───────────────────────────── UTILS ──────────────────────────── #

def _json_key(x: Any) -> str | int | float | bool | None:
    """Converts numpy/object keys to JSON-serializable values."""
    if isinstance(x, (np.generic, np.bool_)):
        return x.item()          # int/float/bool
    return str(x)                # safe fallback

# ────────────────────────── STRATIFICATION ───────────────────────── #

def stratified_split_indices(df: pd.DataFrame, target: str) -> dict | None:
    if target not in df.columns:
        logging.error(f"Target column '{target}' not found.")
        return None

    X = df.drop(columns=[target]).values
    y = df[target].values
    
    # Print class distribution 
    y_counts = Counter(y)
    logging.info(f"Class distribution: {dict(y_counts)}")
    
    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)
    
    # Check if any class has fewer than 2 samples
    class_counts = Counter(y_enc)
    min_samples = min(class_counts.values())
    
    if min_samples < 2:
        logging.warning(
            f"The least populated class has only {min_samples} member(s), which is too few for stratification. "
            f"Using non-stratified split instead."
        )
        # Use non-stratified split instead
        return _non_stratified_split(X, y, y_enc, le)
    
    if len(np.unique(y_enc)) < 2:
        logging.error("At least 2 classes are required for stratification.")
        return None

    try:
        sss1 = StratifiedShuffleSplit(
            n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )
        train_val_idx, test_idx = next(sss1.split(X, y_enc))

        val_fraction = VAL_SIZE / (1.0 - TEST_SIZE)
        sss2 = StratifiedShuffleSplit(
            n_splits=1, test_size=val_fraction, random_state=RANDOM_SEED
        )
        train_idx, val_idx = next(sss2.split(X[train_val_idx], y_enc[train_val_idx]))
        train_idx = train_val_idx[train_idx]
        val_idx   = train_val_idx[val_idx]
        
        return _create_split_dict(train_idx, val_idx, test_idx, y_enc, le)
        
    except ValueError as e:
        logging.warning(f"Stratification failed with error: {str(e)}. Using non-stratified split instead.")
        return _non_stratified_split(X, y, y_enc, le)

def _non_stratified_split(X, y, y_enc, le):
    """Perform a non-stratified split when stratification is not possible."""
    ss1 = ShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    train_val_idx, test_idx = next(ss1.split(X))
    
    val_fraction = VAL_SIZE / (1.0 - TEST_SIZE)
    ss2 = ShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=RANDOM_SEED
    )
    train_idx, val_idx = next(ss2.split(X[train_val_idx]))
    train_idx = train_val_idx[train_idx]
    val_idx   = train_val_idx[val_idx]
    
    return _create_split_dict(train_idx, val_idx, test_idx, y_enc, le)

def _create_split_dict(train_idx, val_idx, test_idx, y_enc, le):
    """Create the split dictionary with indices and class proportions."""
    def proportions(idx):
        if len(idx) == 0:
            return {}
        vals, counts = np.unique(y_enc[idx], return_counts=True)
        total = len(idx)
        return {
            _json_key(le.inverse_transform([v])[0]): (counts[i] / total)
            for i, v in enumerate(vals)
        }

    return {
        "train_indices": train_idx.tolist(),
        "val_indices"  : val_idx.tolist(),
        "test_indices" : test_idx.tolist(),
        "class_proportions": {
            "train": proportions(train_idx),
            "val"  : proportions(val_idx),
            "test" : proportions(test_idx),
        }
    }

# ──────────────────────────── MAIN LOOP ────────────────────────── #
def main() -> None:
    if not INPUT_DIR.is_dir():
        logging.error(f"Input directory not found: {INPUT_DIR}")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_paths = glob.glob(str(INPUT_DIR / "*.csv"))
    if not csv_paths:
        logging.error(f"No CSV files found in {INPUT_DIR}")
        return

    logging.info(f"Processing {len(csv_paths)} dataset(s) from {INPUT_DIR}...")

    # Track file names to handle duplicates
    processed_names = set()

    for csv_path in csv_paths:
        name = Path(csv_path).stem
        logging.info(f"\n─── {name} ─────────────────────────────────────────")
        
        # Always create a directory for each dataset
        dataset_dir = OUTPUT_DIR / name
        dataset_dir.mkdir(exist_ok=True)
        
        df = pd.read_csv(csv_path)
        if df.empty:
            logging.warning("Empty dataset; skipping.")
            continue

        split = stratified_split_indices(df, TARGET_COLUMN)
        if split is None:
            logging.error("Split failed; skipping NaN injection.")
            continue

        # Save indices + distributions
        splits_dir = OUTPUT_DIR / "splits"
        splits_dir.mkdir(exist_ok=True)
        split_path = splits_dir / f"{name}_split.json"
        with open(split_path, "w") as fp:
            json.dump(split, fp, indent=4)
        logging.info(f"Indices and proportions → {split_path}")

        # Save 0% NaN version
        base_out = dataset_dir / f"{name}_00nan.csv"
        df.to_csv(base_out, index=False)
        logging.info(f"Original file → {base_out}")

        # Other percentages
        for pct in NAN_PERCENTAGES:
            df_nan = inject_nans(df, pct, TARGET_COLUMN, seed=RANDOM_SEED + int(pct*100))
            if df_nan is None:
                continue
            pct_tag = f"{int(pct*100):02d}nan"
            out_path = dataset_dir / f"{name}_{pct_tag}.csv"
            df_nan.to_csv(out_path, index=False)
            logging.info(f"{pct*100:.0f}% NaN → {out_path}")

    logging.info("\n✔ Preprocessing completed.")

# ----------------------------------------------------------------- #
if __name__ == "__main__":
    main()
