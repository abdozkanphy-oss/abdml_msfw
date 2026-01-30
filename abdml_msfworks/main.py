from concurrent.futures import ThreadPoolExecutor
from utils.config_reader import ConfigReader
from thread.phase_3_correlation._3_2_correlations import execute_phase_three
from thread.phase_3_correlation._3_2_handle_retrain import execute_retrain

cfg = ConfigReader()
threads = []

# There are two consumer threads, one for preprocessing and one for aggregation.
with ThreadPoolExecutor(max_workers=cfg['thread_num']) as executor:
    for i in range(cfg['execute_phase_three_thread']): # This thread is for extracting correlation/feature inportance/real time prediction data.
        threads.append(executor.submit(execute_phase_three))
    for i in range(cfg['retrain_thread']): # This thread is for extracting correlation/feature inportance/real time prediction data.
        threads.append(executor.submit(execute_retrain))