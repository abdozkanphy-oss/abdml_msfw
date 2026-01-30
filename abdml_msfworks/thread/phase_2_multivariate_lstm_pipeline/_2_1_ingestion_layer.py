import json
import numpy as np
import os
import pytz
import pickle
import threading
from collections import deque
from datetime import datetime
##
from cassandra_utils.models.dw_raw_data_data import dw_raw_data_data
from thread.phase_2_multivariate_lstm_pipeline._2_2_pre_processing_layer import generate_testing_dataframe,type_casting_df
from thread.phase_2_multivariate_lstm_pipeline._2_3_processing_layer import check_model_exists
from thread.phase_2_multivariate_lstm_pipeline._2_5_driver_training import execute_training
from thread.phase_2_multivariate_lstm_pipeline._2_6_driver_inference import exe_live
##
from modules.kafka_modules import kafka_consumer3
from utils.logger_2 import setup_logger

#Logger for debugging
p2_1_inf_log = setup_logger(
    "p2_1_ingestion_layer_inference_logger", "logs/p2_1_ingestion_layer_inference.log"
)

# Global buffer for past 20 dataframes (N sequences for LSTM)
buffer_per_key = {}  # key -> deque
BUFFER_DIR = "models/phase2models/historical-buffer/"

# Lock mechanism to ensure one message is completely processed at a time
processing_lock = threading.Lock()


#PROCESS_CURRENT_DATE_ONLY = True
consumer3  = kafka_consumer3()


def update_buffer(key, new_data):
    if key not in buffer_per_key:
        buffer_per_key[key] = deque(maxlen=19)
    buffer_per_key[key].append(new_data)

def save_buffer_to_file(key):
    """Save the buffer for a specific key to the disk."""
    
    p2_1_inf_log.debug(f"Attempting to save buffer for key: {key}")
    
    if key in buffer_per_key:
        p2_1_inf_log.debug(f"Buffer for key {key} exists, proceeding to save...")
        buffer_path = os.path.join(BUFFER_DIR, f"{key}_buffer.pkl")
        
        # Ensure the directory exists
        os.makedirs(BUFFER_DIR, exist_ok=True)  # Create the directory if it does not exist
        
        p2_1_inf_log.debug(f"Saving buffer to file path: {buffer_path}")
        
        try:
            with open(buffer_path, 'wb') as f:
                p2_1_inf_log.debug(f"Serializing buffer for key {key}...")
                pickle.dump(buffer_per_key[key], f)            
            p2_1_inf_log.debug(f"Buffer for key {key} saved successfully to {buffer_path}")
        
        except Exception as e:
            p2_1_inf_log.debug(f"Error while saving buffer for key {key}: {e}")
    
    else:
        p2_1_inf_log.debug(f"Error: Buffer for key {key} not found in buffer_per_key.")


def load_buffer_from_file(key):
    """Load the buffer for a specific key from the disk."""
    
    p2_1_inf_log.debug(f"Attempting to load buffer for key: {key}")
    
    if not os.path.exists(BUFFER_DIR):
        p2_1_inf_log.warning(f"Warning: The directory {BUFFER_DIR} does not exist. No buffers can be loaded.")
        return

    buffer_path = os.path.join(BUFFER_DIR, f"{key}_buffer.pkl")
    p2_1_inf_log.debug(f"Looking for buffer file at: {buffer_path}")

    if os.path.exists(buffer_path):
        try:
            with open(buffer_path, 'rb') as f:
                p2_1_inf_log.debug(f"Deserializing buffer for key {key}...")
                buffer_per_key[key] = pickle.load(f)            
            p2_1_inf_log.debug(f"Buffer for key {key} loaded successfully from {buffer_path}")        
        except Exception as e:
            p2_1_inf_log.debug(f"Error while loading buffer for key {key}: {e}")
    else:
        p2_1_inf_log.debug(f"No saved buffer found for key {key}.")
        
def _is_valid_message_date(message, today):
    """
    Checks if all 'measDt' values in 'outVals' exist and match today's date.
    Returns True if valid, False otherwise.
    """
    for out_val in message.get('outVals', []):
        meas_dt_ms = out_val.get('measDt')
        if meas_dt_ms is not None:
            meas_date = datetime.fromtimestamp(meas_dt_ms / 1000, tz=pytz.UTC).date()
            if meas_date != today:
                p2_1_inf_log.warning("Date does not match today")
                return False  # Date does not match today, invalid message
        else:
            return False  # measDt missing, invalid message
    return True

# Main Logic of the system
def consumer_preprocess_2():
    print(" in consuming ")
    try:
        while True:
    
        # Step 1 - Fetching Message from Topic 
            print("2.1 STEP 1- Fetching Message from Topic")   
            msg = consumer3.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                p2_1_inf_log.error(f"phase2_consumer - Consumer error: {msg.error()}")
                continue

            try:
        # Step 2 Processing Message - Unloading in json format
                print("2.1 STEP 2 - Unloading in json format")           
                raw_value = msg.value().decode('utf-8')
                message = json.loads(raw_value)


                # Defensive: ensure inputVariableList key is present and is a list
                if 'inputVariableList' not in message or message['inputVariableList'] is None:
                    message['inputVariableList'] = []
                

        # Step 3 Filter messages by date -                 #START WHEN TOPIC IS INITIALIZED
                print("2.1 STEP 3 - Date filter XX measurement date wise")   
                print(datetime.fromtimestamp(int(message['crDt']) / 1000, tz=pytz.UTC).date())
#                today = datetime.now(pytz.UTC).date()
#                if not _is_valid_message_date(message,today):
#                    print
#                    p2_1_inf_log.info("Old message encountered, stopping consumer loop")
#                    break  # Stop processing further messages
                # print("date validity checked")

                if isinstance(message, list):
                    p2_1_inf_log.error("phase2_consumer - Message is a list, skipping")
                    continue

#                print(message)
        # Step 4    Saving message to Cassandra
                print("2.1 STEP 4 - Saving message to Cassandra")   
                data = dw_raw_data_data.saveData(message)
                p2_1_inf_log.debug(f"phase2_consumer - Data saved in Cassandra: {data}")

                print(message)            
        # Step 5 : Extract key from Message based on combination of WorkstationID and StockID
                print("2.1 STEP 5 - Extract key")   
                wsid = message['wsId']
                stid = message['prodList'][0]['stId'] if message['prodList'] else None
                key = f"{wsid}_{stid}"

                p2_1_inf_log.debug(f"The key for current execution :{key}")

        # Step 6: Check if Pretrained model for key exists otherwise train it
                print(f"2.1 STEP 6 - Check if Pretrained model for key : {key}")   
                if not check_model_exists(key):
                    print("2.1 STEP 6A - Not exists a Pretrained model")   
                    p2_1_inf_log.debug(f"No previous Model found for key, Training a fresh LSTM model for key : {key}")
                    last_n_seqs = execute_training(key)
                    p2_1_inf_log.debug(f"Completed Training Phase for key'{key}'")
                    print("2.1 STEP 6Aa - Training Completed")
                    for seq in last_n_seqs:
                        update_buffer(key,seq)

                    save_buffer_to_file(key)
                    p2_1_inf_log.debug("Buffer for last n sequences saved to directory")

                else:
                    p2_1_inf_log.debug(f"Model already exists for key {key}")
        # Step 6b : If pretrained model exists continue with inference
                    print("2.1 STEP 6B - Model Exists")   
                    current_df = generate_testing_dataframe(data)
                    current_df = type_casting_df(current_df)
#                    p2_1_inf_log.debug(f"inputs -> {current_df}")                
                    current_row = current_df.values[0]

                    if not np.any(np.isnan(current_df)) and not np.any(np.isinf(current_df)):
                        p2_1_inf_log.debug("Input data is clean: no NaNs or Infs.")

        #Step 7 : Fetch Last N Sequences and send it with current to Model
                        print("2.1 STEP 7 - Fetching N seqs")
                        if key in buffer_per_key:
                            print("2.1 STEP 7A- found in buffer")
                            p2_1_inf_log.debug(f"{key} key found in buffer for last n sequences executing prediction")
                            exe_live(current_df,buffer_per_key[key],data,key,message)
                            update_buffer(key,current_df.drop(columns=["timestamp"], errors="ignore").values[0])
                        else:
                            print("2.1 STEP 7B - loading for dir")   
                            p2_1_inf_log.error(f"Key not found in buffer for key:'{key}'")
                            load_buffer_from_file(key)
                            if key in buffer_per_key:
                                print("key found in buffer (loading from dir) executing predeiction")
                                exe_live(current_df,buffer_per_key[key],data,key,message)
                                update_buffer(key,current_df.drop(columns=["timestamp"], errors="ignore").values[0])
                            else:
                                p2_1_inf_log.error("Key not found in buffer(even after loading from directory)")
                    else:
                        p2_1_inf_log.warning(f"Input data is not clean Skipping, data = {current_df}")        

                consumer3.commit()
                p2_1_inf_log.debug(f"phase2_consumer - Offset committed for message offset: {msg.offset()}")

            except Exception as e:
                p2_1_inf_log.error(f"phase2_consumer - Error processing message: {e}")

                continue

    except Exception as e:
        p2_1_inf_log.error(f"phase2_consumer - Fatal error in consumer loop: {e}")
        print(e)

    finally:
        consumer3.close()
        p2_1_inf_log.info("phase2_consumer - Consumer closed and resources cleaned up")

