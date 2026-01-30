import os
import numpy as np
import pandas as pd

from cassandra_utils.models.dw_raw_data_data import dw_raw_data_data
from thread.phase_2_multivariate_lstm_pipeline._2_2_pre_processing_layer import generate_training_dataframe, generate_sequences_with_timestamps, has_nan_cntread, scale_and_save_training_data
from thread.phase_2_multivariate_lstm_pipeline._2_3_processing_layer import initial_train_and_save, check_model_exists, predict_with_model
from thread.phase_2_multivariate_lstm_pipeline._2_4_post_processing_layer import compute_reconstruction_error, compute_and_save_threshold, detect_anomalies
from utils.logger_2 import setup_logger

p2_5_driver_tr_log = setup_logger(
    "p2_5_driver_tr_log", "logs/p2_5_driver_tr.log"
)

# FETCHING HISTORICAL DATA FOR MODEL (TRAINING)
def preload_training_data(values_to_fetch=200):
    """
    Preload initial training data from Cassandra.
    Called once at startup or periodically if needed.
    """
    try:
#        returnList, inputList, outputList, batchList = dw_single_data_data.fetchData(values_to_fetch)
        returnList, inputList, outputList, batchList = dw_raw_data_data.fetchData(values_to_fetch)
        if not returnList:
            p2_5_driver_tr_log.warning("2_1_1 - No data found in fetch call")
            p2_5_driver_tr_log.error("2_1_1 - No data received for training")
            raise Exception("No training data fetched")
        else:
            p2_5_driver_tr_log.debug("2_1_1 - Sucessfully preloading data")
            return returnList, inputList, outputList, batchList
    except Exception as e:
        p2_5_driver_tr_log.error(f"2_1_1 - Error preloading training data: {e}")

def execute_training(key):
    print("Starting Training")

#Step 1 - Fetch historical data for training
    print("2.5 STEP 1 - Historical Data fetch")
    p2_5_driver_tr_log.debug("Phase 2_5_ Training, Fetching Data...")
    returnList, inputList, outputList, batchList = preload_training_data(20000)
    p2_5_driver_tr_log.debug("Phase 2_5_ Loaded Training Data")

#Step 2: Filtering data according to key
    print("2.5 STEP 2 - filtering by key")
    filtered_returnList = []
    matching_indices = []

    for idx, item in enumerate(returnList):
        wsid_ = item.workstationid
        stid_ = item.producelist[0]['stId'] if item.producelist else None
        key_ = f"{wsid_}_{stid_}"

        if key_ == key:
            matching_indices.append(idx)

    filtered_batchList = [batchList[i] for i in matching_indices]
    filtered_returnList = [returnList[i] for i in matching_indices]
    filtered_outputList = [outputList[i] for i in matching_indices]

    print(f"found {len(filtered_outputList)} records")
#Step 3: Check for Nan Values
    try:    
        print("2.5 STEP 3 - Checking for NAN")
        for inst in filtered_returnList:
            if has_nan_cntread(inst) == True:
                p2_5_driver_tr_log.warning(f" Found NaN values in Training Data \n :")
                p2_5_driver_tr_log.warning(f"({[getattr(inst, col) for col in inst._columns.keys()]})")
            else:
                continue
    except Exception as e:
        p2_5_driver_tr_log.debug(f"Error filtering for Nan Values {e}")

#Step 4 Validate if Enough samples are available for model training
    print("2.5 STEP 4 - Validate Enough Training Data")
    size_of_tr_data = len(filtered_outputList)
    p2_5_driver_tr_log.debug(f"filtered outputlist shape:{size_of_tr_data} for key: {key}")

    if size_of_tr_data < 100:
        p2_5_driver_tr_log.debug(f"NOT Building model, training size is low: {size_of_tr_data}")
        print(f"found {size_of_tr_data} samples, skipping training")
        raise
    else:
        print(f"found {size_of_tr_data} samples initializing training")
        p2_5_driver_tr_log.debug("Sufficient Data Avaiable, Training Model")

#Step 5 : Build Dataframe, Scale and Generate Sequences
        print("2.5 STEP 5 - Build Dataframe/extract features")
        p2_5_driver_tr_log.debug("Phase 2_5_ Building Dataframe")
        try:
            X = generate_training_dataframe(filtered_batchList,filtered_returnList,filtered_outputList)
            p2_5_driver_tr_log.debug("Feature Extraction Complete")
        except Exception as e:
            print(e)

#Step 6: Convert to appropiate forms : sensor value columns to float
        print("2.5 STEP 6A - Conversion to appropiate forms for sensor values")
        sensor_cols = [col for col in X.columns if col.startswith('sensor_value_')]
        for col in sensor_cols:
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                X[col] = X[col].astype(float)  # force type
            except Exception as e:
                print(f"[ERROR] Failed to convert {col} to float: {e}")

#Step 6b: Convert all other columns (except timestamp and sensor values) to int
        print("2.5 STEP 6B - Converting other variables")
        cols_to_int = [col for col in X.columns if col not in sensor_cols and col != 'timestamp']
        for col in cols_to_int:
            try:
                X[col] = X[col].astype(float)
            except Exception as e:
                p2_5_driver_tr_log.debug(f"[ERROR] Failed to convert {col} to int: {e}")

#Step 7: Final logs
        print("2.5 STEP 7 - Final logging")
        p2_5_driver_tr_log.debug(f"[CLEANED] X dtypes:\n{X.dtypes}")
        p2_5_driver_tr_log.debug(f"[CLEANED] X shape: {X.shape}")
        p2_5_driver_tr_log.debug(f"[CLEANED] Any NaNs: {X.isna().values.any()}")
        p2_5_driver_tr_log.debug(X.head())

#Step 8: Handling Nans
        print("2.5 STEP 8 - Dropping NA")
        p2_5_driver_tr_log.debug(f"Before dropna shape: {X.shape}")
        p2_5_driver_tr_log.debug(f" Total columns with NaN: {X.isna().sum()}")
        try:
            X.dropna(inplace=True)
        except Exception as e:
            p2_5_driver_tr_log.debug(f"Error in Handling Nans {e}")

        p2_5_driver_tr_log.debug(f"After fillna: X shape: {X.shape}, contains NaNs: {np.any(np.isnan(X))}, contains Infs: {np.any(np.isinf(X))}")

#Step 9: Scaling Data
        print("2.5 STEP 9 - Scaling Data")
        p2_5_driver_tr_log.debug("Phase 2_5_ Scaling the Dataframe")
        X_scaled, scaler_path, min_max_path = scale_and_save_training_data(X,key)
        p2_5_driver_tr_log.debug(f"X_scaled shape: {X_scaled.shape}, contains NaNs: {np.any(np.isnan(X_scaled))}, contains Infs: {np.any(np.isinf(X_scaled))}")
 
#Step 10: Generating Sequences
        print("2.5 STEP 10 - Seq Gen")
        p2_5_driver_tr_log.debug("Phase 2_5_ Generating Sequences")
        X_sequences, X_seq_timestamps = generate_sequences_with_timestamps(X_scaled, timesteps=20)

#Step 11: Training and Testing Splits
        print("2.5 STEP 11 - Training/Testing Splits")
        p2_5_driver_tr_log.debug("Phase 2_5_ Spliting sequences into training and testing sets")

        train_size = int(len(X_sequences) * 0.8)
        X_sequences_train = X_sequences[:train_size]
        X_sequences_test = X_sequences[train_size:]
        timestamps_test = X_seq_timestamps[train_size:].reset_index(drop=True)

        p2_5_driver_tr_log.debug(f" Training samples: {len(X_sequences_train)},  Testing samples: {len(X_sequences_test)}")

#Step 12: Model Training
        print("2.5 STEP 12 - Training")
        p2_5_driver_tr_log.debug("Starting Model Training Phase...")

        model_dir = 'models/phase2models'
        model_files = sorted([f for f in os.listdir(model_dir) if f.startswith(key) and f.endswith('.keras')])

        if not model_files:
            latest_model = ""
        else:
            latest_model = model_files[-1]

        model_path = os.path.join(model_dir, latest_model)

        if not check_model_exists(key):
            p2_5_driver_tr_log.debug("No previous Model found, Training a fresh LSTM model")
            model_path = initial_train_and_save(X_sequences_train, X_sequences_train.shape[1], X_sequences_train.shape[2],key)
        else:
            p2_5_driver_tr_log.debug("Model already exists.")    

# Step 13: Post Model Training Prediction
        print("2.5 STEP 13 - Post Training Predictions")
        p2_5_driver_tr_log.debug("Model Trained, Initializing Predictions")
        predictions_test = predict_with_model(model_path, X_sequences_test)
        p2_5_driver_tr_log.debug(f"Completed predictions, Predictions shape: {predictions_test.shape}")
        p2_5_driver_tr_log.debug(f"Predictions contain NaNs: {np.any(np.isnan(predictions_test))}")


# Step 14: Calculating Threshold, No of Anomalies in test set, Sending last N sequences to buffer
        print("2.5 STEP 14 - Threshold, Anomaly etc calculations")
        p2_5_driver_tr_log.debug("Computing and Saving Threshold from test set for later inference processes")

        p2_5_driver_tr_log.debug("Computing Recon Error")
        recon_error_test = compute_reconstruction_error(X_sequences_test, predictions_test)
        p2_5_driver_tr_log.debug("Computing threshold")
        threshold, threshold_path = compute_and_save_threshold(recon_error_test, key)
        p2_5_driver_tr_log.debug("Computing no of anomalies detected")
        anomaly = detect_anomalies(recon_error_test, threshold)

        p2_5_driver_tr_log.debug(f"Model Sucessfully trained, scalers and thresholds saved, anomalies detected on set {np.count_nonzero(anomaly)}")

        p2_5_driver_tr_log.debug(f"getting last n seqs from shape :{X.shape} and seq are {X.tail(20).iloc[:, 1:].to_numpy()}")
        last_20_seqs = X.tail(20).iloc[:, 1:].to_numpy()
        p2_5_driver_tr_log.debug(f"X {key} DataFrame columns: {list(X.columns)}")
        p2_5_driver_tr_log.debug(f"last n seqs shape : {last_20_seqs.shape}")
        print("Pipeline complete")
        return last_20_seqs    