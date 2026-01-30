
import os
import numpy as np
import pandas as pd

from cassandra_utils.models.dw_tbl_multiple_anomalies3 import DwTblMultipleAnomalies
from thread.phase_2_multivariate_lstm_pipeline._2_2_pre_processing_layer import generate_sequences_only, scale_live_data_from_dir
from thread.phase_2_multivariate_lstm_pipeline._2_3_processing_layer import predict_with_model
from thread.phase_2_multivariate_lstm_pipeline._2_4_post_processing_layer import compute_reconstruction_error,live_anomaly_detection, compute_anomaly_importance, sensorwise_error_score

from utils.logger_2 import setup_logger

p2_5_driver_inf_log = setup_logger(
    "p2_5_driver_inf_log", "logs/p2_5_driver_inf.log"
)


import numpy as np
import pandas as pd

def clean_history(X_history, X_in):
    ts = X_in.index[0] if hasattr(X_in.index, '__getitem__') else None
    new_row = X_in.drop(columns=["timestamp"], errors="ignore").to_numpy().flatten()

    # --- Determine expected length dynamically ---
    lengths = [len(np.array(r).flatten()) for r in X_history] + [len(new_row)]
    expected_len = max(set(lengths), key=lengths.count)

    cleaned_history = []
    timestamps = []

    for idx, row in enumerate(X_history):
        arr = np.array(row).flatten()
        orig_len = len(arr)

        ts_candidate = None
        if arr.dtype == object:
            ts_candidate = next((x for x in arr if isinstance(x, (pd.Timestamp, np.datetime64))), None)
            arr = np.array([x for x in arr if not isinstance(x, (pd.Timestamp, np.datetime64))])

        if len(arr) != expected_len:
            if len(arr) > expected_len:
                arr = arr[-expected_len:]
            elif len(arr) < expected_len:
                arr = np.pad(arr, (expected_len-len(arr), 0), constant_values=np.nan)

        cleaned_history.append(arr.astype(float))
        timestamps.append(ts_candidate)

    if len(new_row) != expected_len:
        if len(new_row) > expected_len:
            new_row = new_row[-expected_len:]
        else:
            new_row = np.pad(new_row, (expected_len-len(new_row), 0), constant_values=np.nan)

    cleaned_history.append(new_row.astype(float))
    timestamps.append(ts)

    # --- Final stack ---
    X = pd.DataFrame(np.vstack(cleaned_history))
    return cleaned_history, timestamps, X


def exe_live(X_in,X_history,main_data,key,message):
#Step 1
    print("2.6 STEP 1")
    p2_5_driver_inf_log.debug(f"Initialzing prediction for {main_data} & key {key}")
    p2_5_driver_inf_log.debug(f"Inputs received : \n{X_in} ")

    try:

        X_history, ts_history, X = clean_history(X_history, X_in)
        print("df generated")
    except Exception as e:
        print("df missed")
        p2_5_driver_inf_log.debug(f"Error in merging live data with historical N sequences: {e}")

#Step 2: Scaling
    print("2.6 STEP 2 - Scaling")
    X_scaled= scale_live_data_from_dir(X, key )

#Step 3: Sequence Generation/Validation
    print("2.6 STEP 3 - Seq Gen & Validation")
    X_sequences= generate_sequences_only(X_scaled, timesteps=20)
    p2_5_driver_inf_log.debug("Inputs scaled and sequences generated from X_history")

#Step 4: Initializing Model from directory
    print("2.6 STEP 4 - Initializing Model from directory")
    p2_5_driver_inf_log.debug("Initiazling Model from directory ...")
    model_dir = 'models/phase2models'
    latest_model = sorted([f for f in os.listdir(model_dir) if f.startswith(key) and f.endswith('.keras')])[-1]
    model_path = os.path.join(model_dir, latest_model)

#Step 5: Initializing Live Prediction
    print("2.6 STEP 5 - Initializing Live Prediction")
    p2_5_driver_inf_log.debug("Initialzing Live Prediction")
    prediction_live = predict_with_model(model_path, X_sequences)
#    p2_5_driver_inf_log.debug(f"live pred result -->  {prediction_live}")

#Step 6: Calculating Reconstruction Error, Status, Anomaly Importance
    print("2.6 STEP 6 - Calculations")
    p2_5_driver_inf_log.debug("Computing Reconstruction Error")
#    p2_5_driver_inf_log.debug(f"input to recon error -> {X_sequences , prediction_live}")
    recon_error_live = compute_reconstruction_error(X_sequences, prediction_live)
    p2_5_driver_inf_log.debug(f"Reconstruction Error: {recon_error_live} ")
    anomaly_live = live_anomaly_detection(recon_error_live, key)
    p2_5_driver_inf_log.debug(f"Live Anomaly: {anomaly_live} ")
    anomaly_imp,threshold = compute_anomaly_importance(recon_error_live, key)
    p2_5_driver_inf_log.debug(f"Anomaly Importance: {anomaly_imp} ")

#Step 7: Calculating for Heatmap    
    print("2.6 STEP 7 - Heatmap")
    heatmap_values = sensorwise_error_score(X_sequences, prediction_live)

    sensor_values = {}
    for item in main_data['outputvaluelist']:
        eq_no = item.get('eqNo')
        sensor_values[str(eq_no)] = {k: str(v) for k, v in item.items()}

    heatmap_sensordata = {
            eq_no: {**sensor_data, 'cntRead': str(heatmap_values[i])}
            for i, (eq_no, sensor_data) in enumerate(sensor_values.items())
        }
    
#Step 9: Sending Downstream to Cassandra
    p2_5_driver_inf_log.debug("Initialzing Cassandra ORM for sending predictions downstream")

    DwTblMultipleAnomalies.saveData(
        topic_data=message,
        main_data=main_data,
        anomaly_score_=recon_error_live,
        anomaly_detected_=anomaly_live,
        anomaly_importance_=anomaly_imp,
        sensor_values_=sensor_values,
        heatmap_threshold_=threshold,
        heatmap_=heatmap_sensordata
    )
    print("Prediction - Step 10F")
    p2_5_driver_inf_log.debug("Finished Inference")
