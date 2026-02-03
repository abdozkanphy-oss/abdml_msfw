# ==============================================================================
# [NEW] RANDOM FOREST PREDICTION HANDLER
# ==============================================================================
def _predict_product_rf(message, drug_name, p3_1_log):
    """
    Dedicated handler for the Golden Random Forest Models.
    Robustness Fix: Handles 'null' values in JSON message correctly.
    """
    status = {"ok": False, "wrote": False, "reason": "init_rf"}
    try:
        # --- 1. Robust Identifier Parsing ---
        # Handle case where prodList is None (explicit null in JSON)
        prod_list = message.get("prodList") or []
        st_no = "UNKNOWN"
        if prod_list and len(prod_list) > 0:
            st_no = prod_list[0].get("stNo", "UNKNOWN")

        batch_id = message.get("joRef", "UNKNOWN_BATCH")
        op_tc = message.get("opTc", "UNKNOWN")
        ws_id = message.get("wsId", "UNKNOWN")
        
        # Buffer Key
        buffer_key = f"OPTC_{op_tc}_WS_{ws_id}_ST_{st_no}".replace(" ", "_").replace("-", "_")
        
        # --- 2. Update Buffer ---
        ts_str = message.get("crDt")
        try:
            if not ts_str: ts = datetime.now(timezone.utc)
            elif ts_str.endswith("Z"): ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            else: ts = datetime.fromisoformat(ts_str)
        except: ts = datetime.now(timezone.utc)

        # FIX: Use 'or []' to handle explicit None/null in JSON
        in_vars = message.get("inVars") or []
        out_vals = message.get("outVals") or []

        row_vals = {}
        for inv in in_vars:
            try: row_vals[inv["eqNm"]] = float(inv.get("genReadVal", 0))
            except: pass
        for outv in out_vals:
            try: row_vals[outv["eqNm"]] = float(outv.get("cntRead", 0))
            except: pass
            
        if not row_vals:
            status["reason"] = "no_values"
            return status

        _rf_buffers[buffer_key].append_row(ts, row_vals)
        
        # --- 3. Load Model ---
        safe_drug = _norm_str_rf(drug_name)
        model_key = f"DRUG_{safe_drug}_OUTPUT"
        
        base_path = os.path.join(MODELS_DIR, f"{model_key}__RANDOM_FOREST")
        model_path = f"{base_path}.pkl"
        scaler_path = f"{base_path}_scaler.pkl"
        meta_path = f"{base_path}_meta.json"
        
        if not os.path.exists(model_path):
            status["reason"] = f"model_missing_{model_key}"
            return status 

        rf_model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        with open(meta_path, "r") as f: meta = json.load(f)
        
        cols = meta.get("cols", [])
        lookback = meta.get("timesteps", 20)
        
        # --- 4. Prepare Data ---
        df_buf = _rf_buffers[buffer_key].df
        
        # Resample
        df_res = df_buf.resample("1min").mean().ffill().fillna(0.0)
        
        # Time Elapsed Feature
        if meta.get("config", {}).get("USE_TIME_ELAPSED", True):
            _calculate_time_elapsed(batch_id, df_res.index[-1])
            start_ts = _BATCH_START_MAP.get(batch_id, df_res.index[0])
            
            idx = df_res.index
            if idx.tz is not None and start_ts.tzinfo is None:
                 start_ts = start_ts.replace(tzinfo=idx.tz)
            elif idx.tz is None and start_ts.tzinfo is not None:
                 idx = idx.tz_localize(start_ts.tzinfo)
            
            elapsed_minutes = (idx - start_ts).total_seconds() / 60.0
            df_res['meta_time_elapsed'] = np.clip(elapsed_minutes, 0, None)

        # Align Columns
        df_ready = df_res.reindex(columns=cols, fill_value=0.0)
        
        if len(df_ready) < lookback:
            status["reason"] = f"gathering_data_{len(df_ready)}"
            return status
            
        # Inference
        X_window = df_ready.iloc[-lookback:].values
        X_scaled = scaler.transform(X_window)
        X_scaled = np.clip(X_scaled, -10.0, 10.0)
        
        # Predict
        X_flat = X_scaled.reshape(1, -1)
        y_pred_scaled = rf_model.predict(X_flat)
        y_pred = scaler.inverse_transform(y_pred_scaled)[0]
        
        current_actuals = df_ready.iloc[-1]
        payload = []
        
        # Anomalies
        for i, col in enumerate(cols):
            if col == "meta_time_elapsed": continue
            
            pred = float(y_pred[i])
            act = float(current_actuals[col])
            diff = abs(pred - act)
            
            is_anom = False
            if abs(act) > 10.0:
                if (diff / (abs(act) + 1e-6)) > 0.15: is_anom = True
            else:
                if diff > 3.0: is_anom = True
                
            if col in row_vals or is_anom:
                payload.append({
                    "equipment_name": col,
                    "prediction": pred,
                    "actual": act,
                    "anomaly": is_anom,
                    "confidence": 0.90
                })
                
        # Save
        ref_key = f"{buffer_key}__ALG_RANDOM_FOREST"
        ScadaRealTimePredictions.saveData(
            key=ref_key,
            now_ts=datetime.now(timezone.utc),
            algorithm="RANDOM_FOREST",
            input_payload=[],
            output_payload=payload,
            meta={"drug": drug_name, "batch": batch_id},
            p3_1_log=p3_1_log
        )
        
        status["ok"] = True
        status["wrote"] = True
        status["reason"] = "success_rf"
        return status
        
    except Exception as e:
        status["reason"] = f"rf_error: {e}"
        if p3_1_log: p3_1_log.error(f"[ProductRF] Error: {e}", exc_info=True)
        return status