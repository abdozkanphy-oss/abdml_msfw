import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timezone

# ML Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler

# Setup Logging
logging.basicConfig(level=logging.INFO, format='[PRODUCT_TRAIN] %(message)s')
logger = logging.getLogger()

# =============================================================================
# 1. IMPORT CORE PREPROCESSING
# =============================================================================
try:
    from thread.phase_3_correlation._3_3_predictions import (
        _rt_prepare_df, 
        _make_sequences,
        _save_model_any,
        _save_meta,
        _model_paths,
        MODELS_DIR,
        _norm_str
    )
except ImportError:
    logger.error("Could not import _3_3_predictions. Check project root.")
    sys.exit(1)

# =============================================================================
# 2. CONFIGURATION & TOGGLES
# =============================================================================
CONFIG = {
    # TOGGLE 1: Add "Time Since Batch Start" as a feature?
    "USE_TIME_ELAPSED": True, 

    # TOGGLE 2: How to handle empty data (e.g., Mixer off while Dryer runs)?
    "INACTIVE_STRATEGY": "FFILL", # "ZERO" or "FFILL"
    
    # Model Params
    "LOOKBACK": 20,
    "ALGORITHM": "RANDOM_FOREST" 
}

# =============================================================================
# 3. DATA PARSING (STRATEGY 2: WIDE DRUG MATRIX)
# =============================================================================
def parse_and_merge_by_drug(csv_path, target_drug_name=None):
    if not os.path.exists(csv_path): return {}

    logger.info(f"Loading CSV: {csv_path}...")
    df = pd.read_csv(csv_path)
    df['ts'] = pd.to_datetime(df['create_date'])
    df = df.sort_values('ts')
    
    # Value Extraction
    df['is_input'] = df['equipment_type'].apply(lambda x: str(x).lower() in ['true', '1', 't', 'yes'])
    def get_val(row):
        return row['gen_read_val'] if (row['is_input'] and pd.notna(row.get('gen_read_val'))) else row.get('counter_reading')
    df['value'] = pd.to_numeric(df.apply(get_val, axis=1), errors='coerce')
    df = df.dropna(subset=['value'])
    
    # Group by Stock No (Drug ID)
    grouped_drugs = {}
    group_cols = ['produced_stock_no', 'produced_stock_name']
    
    for (st_no, st_name), drug_group in df.groupby(group_cols):
        
        if target_drug_name and target_drug_name.lower() not in str(st_name).lower():
            continue
            
        drug_key = f"DRUG_{_norm_str(st_name)}"
        logger.info(f"--- Building Matrix for: {st_name} ({st_no}) ---")
        
        batch_dfs = []
        for batch_id, batch_data in drug_group.groupby('prod_order_reference_no'):
            
            # 1. Pivot Batch
            wide_batch = batch_data.pivot_table(
                index='ts', columns='equipment_name', values='value', aggfunc='mean'
            )
            
            # 2. Resample
            wide_batch = wide_batch.resample('1min').mean()
            
            # 3. Handle Inactive
            if CONFIG["INACTIVE_STRATEGY"] == "ZERO":
                wide_batch = wide_batch.fillna(0.0)
            elif CONFIG["INACTIVE_STRATEGY"] == "FFILL":
                wide_batch = wide_batch.ffill().fillna(0.0)
            
            # 4. Add Time Elapsed
            if CONFIG["USE_TIME_ELAPSED"] and len(wide_batch) > 0:
                start_time = wide_batch.index[0]
                time_deltas = (wide_batch.index - start_time).total_seconds() / 60.0
                wide_batch['meta_time_elapsed'] = time_deltas
            
            if len(wide_batch) > CONFIG["LOOKBACK"]:
                batch_dfs.append(wide_batch)

        grouped_drugs[drug_key] = batch_dfs
        
    return grouped_drugs

# =============================================================================
# 4. TRAINING & EVALUATION
# =============================================================================
def train_product_model(target_drug="Antares"):
    
    # 1. Parse Data
    drug_matrices = parse_and_merge_by_drug("dw_tbl_raw_data_MARIFARM.csv", target_drug)
    
    if not drug_matrices:
        logger.warning("No data found for target drug.")
        return

    for key, batches in drug_matrices.items():
        logger.info(f"Training Model for {key} | Batches: {len(batches)}")
        
        if len(batches) < 2:
            logger.warning("Not enough batches to split.")
            continue

        # --- A. Align Columns ---
        all_cols = set()
        for b in batches: all_cols.update(b.columns)
        all_cols = sorted(list(all_cols))
        
        aligned_batches = []
        for b in batches:
            b_aligned = b.reindex(columns=all_cols, fill_value=0).sort_index()
            aligned_batches.append(b_aligned)
            
        # --- B. Split & Scale ---
        split_idx = int(len(aligned_batches) * 0.8)
        train_dfs = aligned_batches[:split_idx]
        val_dfs = aligned_batches[split_idx:]
        
        scaler = RobustScaler(quantile_range=(5.0, 95.0))
        full_train = pd.concat(train_dfs)
        scaler.fit(full_train.values)
        
        # --- C. Sequence Generation ---
        def get_seqs(dfs):
            X, y = [], []
            for df in dfs:
                vals = scaler.transform(df.values)
                vals = np.clip(vals, -10.0, 10.0)
                xs, ys = _make_sequences(vals, CONFIG["LOOKBACK"])
                if len(xs) > 0:
                    X.append(xs)
                    y.append(ys)
            if not X: return None, None
            return np.vstack(X), np.vstack(y)

        X_train, y_train = get_seqs(train_dfs)
        X_val, y_val = get_seqs(val_dfs)
        
        if X_train is None: continue

        # --- D. Train ---
        logger.info(f"   Input Shape: {X_train.shape} (Samples, Time, Sensors)")
        
        # Flatten for Tree Models
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        
        model = MultiOutputRegressor(RandomForestRegressor(
            n_estimators=100, 
            max_depth=15, 
            n_jobs=-1,
            random_state=42
        ), n_jobs=-1)
        
        model.fit(X_train_flat, y_train)
        
        # --- E. Evaluate ALL SENSORS ---
        if X_val is not None:
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            y_pred_scaled = model.predict(X_val_flat)
            y_pred = scaler.inverse_transform(y_pred_scaled)
            y_true = scaler.inverse_transform(y_val)
            
            # Create a Directory for Plots
            eval_dir = os.path.join(MODELS_DIR, f"{key}_eval_plots")
            os.makedirs(eval_dir, exist_ok=True)
            logger.info(f"   Generating plots in: {eval_dir}")
            
            global_mape = []
            
            # LOOP OVER EVERY SENSOR
            for i, col_name in enumerate(all_cols):
                # Calculate R2 for this specific sensor
                sensor_r2 = r2_score(y_true[:, i], y_pred[:, i])
                
                # Calculate Safe MAPE
                mask = np.abs(y_true[:, i]) > 0.01
                if np.sum(mask) > 0:
                    sensor_mape = np.mean(np.abs((y_true[mask, i] - y_pred[mask, i]) / y_true[mask, i])) * 100
                    global_mape.append(sensor_mape)
                else:
                    sensor_mape = 0.0

                # PLOT
                plt.figure(figsize=(10, 4))
                plt.plot(y_true[:, i], label='Actual', color='blue', alpha=0.7)
                plt.plot(y_pred[:, i], label='Predicted', color='orange', linestyle='--', alpha=0.8)
                plt.title(f"{col_name}\nR2: {sensor_r2:.2f} | MAPE: {sensor_mape:.1f}%")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Sanitize filename
                safe_col = _norm_str(col_name)
                plt.savefig(os.path.join(eval_dir, f"{safe_col}.png"))
                # plt.savefig(os.path.join(eval_dir, f"{key}_{safe_col}_RANDOM_FOREST.png"))
                plt.close()
                
            logger.info(f"   [RESULT] Global Avg MAPE: {np.mean(global_mape):.2f}%")

        # --- F. Save ---
        model_path, scaler_path, meta_path = _model_paths(f"{key}_OUTPUT", "RANDOM_FOREST")
        _save_model_any(model, model_path, "RANDOM_FOREST", p3_1_log=logger)
        joblib.dump(scaler, scaler_path)
        
        _save_meta(meta_path, {
            "cols": all_cols,
            "timesteps": CONFIG["LOOKBACK"],
            "config": CONFIG,
            "last_trained": datetime.now(timezone.utc).isoformat()
        })
        logger.info(f"   [SAVED] {key}")

if __name__ == "__main__":
    train_product_model(target_drug="Antares")