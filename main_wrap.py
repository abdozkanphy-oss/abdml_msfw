from concurrent.futures import ThreadPoolExecutor
from utils.config_reader import ConfigReader
from utils.logger import logger
from thread.phase_3_correlation._3_2_correlations import execute_phase_three
from thread.phase_3_correlation._3_2_handle_retrain import execute_retrain

cfg = ConfigReader()

def _wrap(name, fn):
    try:
        logger.info(f"[main] starting: {name}")
        fn()
    except Exception as e:
        logger.error(f"[main] thread crashed: {name} | {e}", exc_info=True)
        raise

with ThreadPoolExecutor(max_workers=cfg["thread_num"]) as executor:
    futures = []
    for _ in range(cfg["execute_phase_three_thread"]):
        futures.append(executor.submit(_wrap, "execute_phase_three", execute_phase_three))
    for _ in range(cfg["retrain_thread"]):
        futures.append(executor.submit(_wrap, "execute_retrain", execute_retrain))

    # Block so exceptions are not swallowed
    for f in futures:
        f.result()
