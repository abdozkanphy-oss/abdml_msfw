from concurrent.futures import ThreadPoolExecutor
from utils.config_reader import ConfigReader
from thread.phase_3_correlation._3_2_correlations import execute_phase_three
from thread.phase_3_correlation._3_2_handle_retrain import execute_retrain
#from thread.phase_3_correlation._3_phase3 import execute_phase_three
#from thread.phase_2_multivariate_lstm_pipeline._2_1_ingestion_layer import consumer_preprocess_2


cfg = ConfigReader()
threads = []

# There are two consumer threads, one for preprocessing and one for aggregation.
with ThreadPoolExecutor(max_workers=cfg['thread_num']) as executor:
    #for i in range(cfg['consumer_preprocess_thread_2']): # This thread is for real time prediction.
        #threads.append(executor.submit(consumer_preprocess_2))
    for i in range(cfg['execute_phase_three_thread']): # This thread is for extracting correlation/feature inportance/real time prediction data.
        threads.append(executor.submit(execute_phase_three))
    for i in range(cfg['retrain_thread']): # This thread is for extracting correlation/feature inportance/real time prediction data.
        threads.append(executor.submit(execute_retrain))