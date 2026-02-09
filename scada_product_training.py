import os
import json
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import RobustScaler

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='[PRODUCT_TRAIN] %(message)s')
logger = logging.getLogger("product_train")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _safe_name(x: str) -> str:
    x = str(x or "").strip()
    x = x.replace(" ", "_").replace("-", "_").replace("/", "_")
    return "".join(ch for ch in x if ch.isalnum() or ch == "_")


def _ensure_datetime_utc(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    return dt


def build_wide_batch_frame(
    df_long: pd.DataFrame,
    resample_seconds: int = 60,
    resample_method: str = "last",
    inactive_strategy: str = "FFILL",
) -> pd.DataFrame:
    """
    df_long must contain: ts, equipment_name, counter_reading
    Returns wide DF indexed by ts with one col per equipment_name.
    """
    if df_long.empty:
        return pd.DataFrame()

    wide = df_long.pivot_table(
        index="ts",
        columns="equipment_name",
        values="counter_reading",
        aggfunc="mean",
    )

    wide = wide.sort_index()
    wide = wide[~wide.index.duplicated(keep="last")]

    # Resample
    rule = f"{int(resample_seconds)}S"
    if resample_method.lower() == "mean":
        wide = wide.resample(rule).mean()
    else:
        # default: last
        wide = wide.resample(rule).last()

    # Fill strategy
    if inactive_strategy.upper() == "ZERO":
        wide = wide.fillna(0.0)
    else:
        # default: forward-fill then remaining to 0
        wide = wide.ffill().fillna(0.0)

    # Ensure numeric
    wide = wide.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return wide


def make_windows(
    wide: pd.DataFrame,
    lookback: int,
    use_time_elapsed: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build (X,y) for next-step multivariate forecasting.

    X shape: (N, lookback, F)
    y shape: (N, F)
    """
    if wide is None or wide.empty:
        return np.empty((0, lookback, 0)), np.empty((0, 0)), []

    df = wide.copy()
    cols = list(df.columns)

    # Optional time feature
    if use_time_elapsed:
        t0 = df.index.min()
        df["meta_time_elapsed_sec"] = (df.index - t0).total_seconds().astype(float)
        cols = cols + ["meta_time_elapsed_sec"]

    arr = df[cols].to_numpy(dtype=float)

    if len(arr) <= lookback:
        return np.empty((0, lookback, len(cols))), np.empty((0, len(cols))), cols

    X_list, y_list = [], []
    for i in range(lookback, len(arr)):
        X_list.append(arr[i - lookback : i])
        y_list.append(arr[i])

    X = np.asarray(X_list, dtype=float)
    y = np.asarray(y_list, dtype=float)
    return X, y, cols


def train_rf_multioutput(X: np.ndarray, y: np.ndarray, n_estimators: int = 300, random_state: int = 42):
    """
    Train MultiOutputRegressor(RandomForestRegressor) on flattened windows.
    """
    # Flatten (N, T, F) -> (N, T*F)
    X_flat = X.reshape(X.shape[0], -1)

    base = RandomForestRegressor(
        n_estimators=int(n_estimators),
        random_state=int(random_state),
        n_jobs=-1,
    )
    model = MultiOutputRegressor(base)
    model.fit(X_flat, y)
    return model


# -----------------------------------------------------------------------------
# Main training pipeline
# -----------------------------------------------------------------------------
def train_all_drugs(
    csv_path: str,
    out_dir: str,
    lookback: int = 20,
    resample_seconds: int = 60,
    resample_method: str = "last",
    inactive_strategy: str = "FFILL",
    use_time_elapsed: bool = True,
    min_points_per_drug: int = 300,
):
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    required = [
        "prod_order_reference_no",
        "produced_stock_name",
        "equipment_name",
        "counter_reading",
        "create_date",
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["ts"] = _ensure_datetime_utc(df["create_date"])
    df = df.dropna(subset=["ts"])
    df = df.sort_values("ts")

    # Ensure strings
    df["prod_order_reference_no"] = df["prod_order_reference_no"].astype(str)
    df["produced_stock_name"] = df["produced_stock_name"].astype(str)
    df["equipment_name"] = df["equipment_name"].astype(str)

    drugs = sorted(df["produced_stock_name"].dropna().unique().tolist())
    logger.info(f"Found {len(drugs)} drugs: {drugs}")

    for drug in drugs:
        df_d = df[df["produced_stock_name"] == drug].copy()
        if df_d.empty:
            continue

        # Build batch-wise wide frames and concatenate
        batches = sorted(df_d["prod_order_reference_no"].unique().tolist())
        all_X, all_y = [], []
        cols_ref = None

        logger.info(f"Training drug='{drug}' batches={len(batches)}")

        for b in batches:
            df_b = df_d[df_d["prod_order_reference_no"] == b].copy()
            df_b = df_b[["ts", "equipment_name", "counter_reading"]]

            wide = build_wide_batch_frame(
                df_b,
                resample_seconds=resample_seconds,
                resample_method=resample_method,
                inactive_strategy=inactive_strategy,
            )
            if wide.empty:
                continue

            X, y, cols = make_windows(wide, lookback=lookback, use_time_elapsed=use_time_elapsed)
            if X.shape[0] == 0:
                continue

            if cols_ref is None:
                cols_ref = cols
            else:
                # Align columns to first batch’s columns (stable feature space)
                # If a batch is missing a sensor col, add it with zeros.
                missing = [c for c in cols_ref if c not in wide.columns and c != "meta_time_elapsed_sec"]
                for c in missing:
                    wide[c] = 0.0
                # Ensure same order
                if use_time_elapsed and "meta_time_elapsed_sec" not in wide.columns:
                    t0 = wide.index.min()
                    wide["meta_time_elapsed_sec"] = (wide.index - t0).total_seconds().astype(float)
                wide = wide[[c for c in cols_ref if c in wide.columns]]

                X, y, _ = make_windows(wide, lookback=lookback, use_time_elapsed=False)  # already included if needed

            all_X.append(X)
            all_y.append(y)

        if not all_X:
            logger.info(f"Skipping drug='{drug}': no windows produced.")
            continue

        X_all = np.concatenate(all_X, axis=0)
        y_all = np.concatenate(all_y, axis=0)

        if X_all.shape[0] < int(min_points_per_drug):
            logger.info(f"Skipping drug='{drug}': only {X_all.shape[0]} samples (<{min_points_per_drug}).")
            continue

        # Scale features for RF stability (robust)
        scaler = RobustScaler()
        X_flat = X_all.reshape(X_all.shape[0], -1)
        X_scaled = scaler.fit_transform(X_flat)

        # Train RF
        model = train_rf_multioutput(
            X=X_scaled.reshape(X_all.shape[0], X_all.shape[1], X_all.shape[2]),
            y=y_all,
            n_estimators=300,
            random_state=42,
        )

        safe_drug = _safe_name(drug)
        base_name = f"DRUG_{safe_drug}_OUTPUT__RANDOM_FOREST"

        model_path = os.path.join(out_dir, f"{base_name}.pkl")
        scaler_path = os.path.join(out_dir, f"{base_name}_scaler.pkl")
        meta_path = os.path.join(out_dir, f"{base_name}_meta.json")

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        meta = {
            "created_utc": datetime.utcnow().isoformat() + "Z",
            "drug_name": drug,
            "lookback": int(lookback),
            "resample_seconds": int(resample_seconds),
            "resample_method": str(resample_method),
            "inactive_strategy": str(inactive_strategy),
            "use_time_elapsed": bool(use_time_elapsed),
            "cols": cols_ref if cols_ref is not None else [],
            "n_samples": int(X_all.shape[0]),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved: {model_path}")
        logger.info(f"Saved: {scaler_path}")
        logger.info(f"Saved: {meta_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to dw_tbl_raw_data csv (sample OK)")
    ap.add_argument("--outdir", default="models", help="Output dir for models")
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--resample_seconds", type=int, default=60)
    ap.add_argument("--resample_method", type=str, default="last", choices=["last", "mean"])
    ap.add_argument("--inactive_strategy", type=str, default="FFILL", choices=["FFILL", "ZERO"])
    ap.add_argument("--use_time_elapsed", action="store_true")
    ap.add_argument("--no_time_elapsed", action="store_true")
    ap.add_argument("--min_points_per_drug", type=int, default=300)
    args = ap.parse_args()

    use_time = True
    if args.no_time_elapsed:
        use_time = False
    if args.use_time_elapsed:
        use_time = True

    train_all_drugs(
        csv_path=args.csv,
        out_dir=args.outdir,
        lookback=args.lookback,
        resample_seconds=args.resample_seconds,
        resample_method=args.resample_method,
        inactive_strategy=args.inactive_strategy,
        use_time_elapsed=use_time,
        min_points_per_drug=args.min_points_per_drug,
    )


if __name__ == "__main__":
    main()
