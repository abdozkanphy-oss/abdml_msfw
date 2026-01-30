import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from utils.logger_2 import setup_logger

#Loggers
p2_4_inf_log = setup_logger(
    "p2_4_postprocessing_layer_inference_logger", "logs/p2_4_postprocessing_layer_inference.log"
)

p2_4_train_log = setup_logger(
    "p2_4_postprocessing_layer_training_logger", "logs/p2_4_postprocessing_layer_training.log"
)

#Training + Inference
def compute_reconstruction_error(y_true, y_pred):
    """
    Compute mean absolute reconstruction error across timesteps and features.
    
    Returns:
    - error: 1D array of error per sample
    """
    error = np.mean(np.abs(y_true - y_pred), axis=(1, 2))
    return error

#Training
def compute_and_save_threshold(recon_error, key, percentile=95, save_dir='models/phase2models/threshold'):
    """
    Computes anomaly threshold from reconstruction error, deletes old threshold files,
    and saves the new one with a timestamp.
    """
    p2_4_train_log.debug("=== Starting compute_and_save_threshold ===")

    # Step 1: Input info
    p2_4_train_log.debug(f"Input type: {type(recon_error)}")
    p2_4_train_log.debug(f"Attempting to convert to numpy array if not already.")
    recon_error = np.asarray(recon_error)

    # Step 2: Validate shape
    p2_4_train_log.debug(f"Array shape: {recon_error.shape}")
    if recon_error.ndim != 1:
        p2_4_train_log.error(f"Expected 1D array, got shape {recon_error.shape}")
        raise ValueError(f"Expected 1D array for recon_error, got shape: {recon_error.shape}")

    # Step 3: Empty check
    if recon_error.size == 0:
        p2_4_train_log.error("recon_error array is empty.")
        raise ValueError("recon_error is empty.")

    # Step 4: NaN/Inf checks
    if np.isnan(recon_error).any():
        p2_4_train_log.error("recon_error contains NaN values.")
        raise ValueError("recon_error contains NaN values.")
    if np.isinf(recon_error).any():
        p2_4_train_log.error("recon_error contains infinite values.")
        raise ValueError("recon_error contains infinite values.")

    p2_4_train_log.debug(f"Input reconstruction error array size: {len(recon_error)}")
    p2_4_train_log.debug(f"Using percentile: {percentile}, save_dir: {save_dir}")

    # Step 5: Ensure save directory exists
    try:
        p2_4_train_log.debug(f"Ensuring save directory exists: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        p2_4_train_log.debug(f"Directory {save_dir} ensured to exist.")
    except Exception as e:
        p2_4_train_log.error(f"Failed to create directory {save_dir}: {e}")
        raise

    # Step 6: Delete old threshold files
    try:
        p2_4_train_log.debug("Clearing old threshold files in the directory.")
        for f in os.listdir(save_dir):
            if f.startswith(key) and f.endswith('.npz'):
                old_file_path = os.path.join(save_dir, f)
                p2_4_train_log.debug(f"Deleting old file: {old_file_path}")
                os.remove(old_file_path)
        p2_4_train_log.debug("Old threshold files cleared successfully.")
    except Exception as e:
        p2_4_train_log.error(f"Error clearing old threshold files: {e}")
        raise

    # Step 7: Compute threshold
    try:
        p2_4_train_log.debug("Computing threshold using np.percentile.")
        threshold = np.percentile(recon_error, percentile)
        p2_4_train_log.debug(f"Threshold computed successfully: {threshold:.6f}")
    except Exception as e:
        p2_4_train_log.error(f"Failed to compute threshold: {e}")
        raise

    # Step 8: Save threshold
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        threshold_path = os.path.join(save_dir, f'{key}_{timestamp}.npz')
        p2_4_train_log.debug(f"Saving threshold and recon_error to: {threshold_path}")
        np.savez(threshold_path, threshold=threshold, recon_error=recon_error)
        p2_4_train_log.debug("Threshold file saved successfully.")
    except Exception as e:
        p2_4_train_log.error(f"Error saving threshold file: {e}")
        raise

    # Final logs
    p2_4_train_log.debug(f"Percentile used: {percentile}")
    p2_4_train_log.debug(f"Final threshold value: {threshold:.6f}")
    p2_4_train_log.debug(f"Threshold file path: {threshold_path}")
    p2_4_train_log.debug("=== compute_and_save_threshold complete ===")

    return threshold, threshold_path

#Training
def detect_anomalies(recon_error, threshold):
    """
    Detects anomalies based on reconstruction error and threshold.

    Parameters:
    - recon_error (np.ndarray): Array of reconstruction errors.
    - threshold (float): Threshold for anomaly detection.

    Returns:
    - anomalies (np.ndarray): Boolean array where True = anomaly.
    """
    anomalies = recon_error > threshold
    p2_4_train_log.debug(f"Detected {np.sum(anomalies)} anomalies out of {len(recon_error)} samples.")
    return anomalies

#For Inference Modules
def live_anomaly_detection(recon_error, key, threshold_dir='models/phase2models/threshold'):
    """
    Loads latest threshold and detects anomalies from reconstruction errors.

    Parameters:
    - recon_error (np.ndarray or float): Reconstruction error(s), can be scalar or array.
    - threshold_dir (str): Directory containing saved threshold .npz file.

    Returns:
    - anomalies (np.ndarray or bool): True where anomaly detected.
    """
    # Load latest threshold file
    threshold_files = sorted(
        [f for f in os.listdir(threshold_dir) if f.startswith(key) and f.endswith('.npz')],
        reverse=True
    )
    if not threshold_files:
        p2_4_inf_log.debug(f" No threshold file found in {threshold_dir}")
        raise FileNotFoundError(f" No threshold file found in {threshold_dir}")

    threshold_path = os.path.join(threshold_dir, threshold_files[0])
    data = np.load(threshold_path)
    threshold = float(data['threshold'])

#    print("threshold value : ",threshold)
    p2_4_inf_log.debug(f" Live detection using threshold {threshold} and recon error {recon_error}")

    # Detect anomalies
    recon_error = np.asarray(recon_error)
    anomalies = recon_error > threshold

    return anomalies

def compute_anomaly_importance(recon_error, key ,threshold_dir='models/phase2models/threshold'):
    """
    Computes anomaly importance by dividing reconstruction error by threshold.

    Parameters:
    - recon_error (float or np.ndarray): Reconstruction error(s).
    - threshold_dir (str): Path to directory containing threshold .npz file.

    Returns:
    - importance (float or np.ndarray): Importance score(s).
    """
    # Load latest threshold
    threshold_files = sorted(
        [f for f in os.listdir(threshold_dir) if f.startswith(key) and f.endswith('.npz')],
        reverse=True
    )
    if not threshold_files:
        p2_4_inf_log.error(f"No threshold file found in {threshold_dir}")
        raise FileNotFoundError(f"No threshold file found in {threshold_dir}")

    threshold_path = os.path.join(threshold_dir, threshold_files[0])
    threshold = float(np.load(threshold_path)['threshold'])

    p2_4_inf_log.debug(f"Loaded threshold: {threshold} ")
#    print("Loaded threshold:", threshold)

    # Convert input to numpy array for consistency
    recon_error = np.asarray(recon_error)

    # Importance is ratio of error to threshold
    importance = recon_error / threshold
    p2_4_inf_log.debug(f"Recon error : {recon_error}")
    p2_4_inf_log.debug(f"Importance  : {importance}")

    return importance,threshold

def sensorwise_error_score(X_actual, X_pred):
    """
    X_actual_df: DataFrame with 1 row (actual sensor values)
    X_pred: 3D numpy array (n_seq, timesteps, n_sensors)

    Returns:
    - 1D array of absolute errors per sensor
    """
    actual = X_actual[-1, -1, :]

    predicted = X_pred[-1, -1, :]

    error = np.abs(actual - predicted)
    return error

