
# # import pandas as pd
# # import numpy as np
# # from utils.logger import ph3_1_logger
# # import inspect
# # import psutil
# # import gc
# # from typing import Dict, List
# # from cassandra_utils.models.correlation_summary import ProductionCorrelation, ProductionCorrelationSummary
# # from datetime import datetime
# # import pytz

# # # Key generation functions
# # def input_key_generation(input_var: Dict, prefix: bool = True) -> str:
# #     var_name = input_var.get('varNm', '').replace(' ', '_')
# #     return f"in_{var_name}" if prefix and var_name else var_name

# # def output_key_generation(output: Dict, prefix: bool = True) -> str:
# #     eq_name = output.get('eqNm', '').replace(' ', '_')
# #     return f"out_{eq_name}" if prefix and eq_name else eq_name

# # def log(msg):
# #     func = inspect.stack()[1].function
# #     ph3_1_logger.debug(f"[{func}] - {msg}")

# # class Phase3:
# #     def __init__(self, lookback_length: int = 5000, window: int = 50, memory_check_interval: int = 100, max_memory_percent: float = 70):
# #         self._window = window
# #         self._lookback_length = lookback_length
# #         self._all_df = pd.DataFrame()
# #         self._past_key = []
# #         self._processing_counter = 0
# #         self._memory_check_interval = memory_check_interval
# #         self._max_memory_percent = max_memory_percent
# #         self._last_memory_check = 0

# #     def add_data(self, data: Dict) -> None:
# #         # 1. Map input fields to expected format
# #         data = self._map_input_fields(data)
        
# #         # 2. Check required fields
# #         required_fields = ['joOpId', 'unique_code', 'wsId', 'outputValueList']
# #         if not all(field in data for field in required_fields):
# #             missing = [field for field in required_fields if field not in data]
# #             ph3_1_logger.error(f"Missing required fields: {missing}. Skipping processing.")
# #             return
            
# #         # 3. Generate process_id safely
# #         process_id = f"{data['joOpId']}_{data['unique_code']}"
# #         ph3_1_logger.debug(f"Processing row with process_id: {process_id}")

# #         if self._is_memory_exceeded():
# #             ph3_1_logger.warning("Memory threshold exceeded - skipping data processing")
# #             gc.collect()
# #             return

# #         try:
# #             idx = len(self._all_df)
# #             current_key = []
# #             outputDict = {}
# #             inputDict = {}
# #             row = {}

# #             # Process output values
# #             output_list = data.get('outputValueList', [])
# #             ph3_1_logger.debug(f"OutputValueList: {output_list}")
# #             for output in output_list:
# #                 if not isinstance(output, dict) or 'eqNm' not in output or 'cntRead' not in output:
# #                     ph3_1_logger.warning(f"Invalid output entry: {output}")
# #                     continue
# #                 key = output_key_generation(output, prefix=True)
# #                 if not key.startswith('out_'):
# #                     ph3_1_logger.warning(f"Invalid output key generated: {key}")
# #                     continue
# #                 try:
# #                     # Handle different numeric types and nulls
# #                     value = float(output["cntRead"]) if output["cntRead"] is not None else np.nan
# #                 except (ValueError, TypeError):
# #                     ph3_1_logger.warning(f"Invalid cntRead in output: {output}")
# #                     value = np.nan
# #                 row[key] = value
# #                 current_key.append(key)
# #                 if not np.isnan(value):
# #                     outputDict[output['eqNm']] = value

# #             # Process input variables
# #             for input_var in data.get('inputVariableList', []):
# #                 if not isinstance(input_var, dict) or 'varNm' not in input_var:
# #                     ph3_1_logger.warning(f"Invalid input entry: {input_var}")
# #                     continue
# #                 key = input_key_generation(input_var, prefix=True)
# #                 if not key.startswith('in_'):
# #                     ph3_1_logger.warning(f"Invalid input key generated: {key}")
# #                     continue
# #                 current_key.append(key)
# #                 try:
# #                     # Handle different numeric types and nulls
# #                     value = float(input_var.get('actVal')) if input_var.get('actVal') is not None else np.nan
# #                 except (ValueError, TypeError):
# #                     ph3_1_logger.warning(f"Invalid actVal in input: {input_var}")
# #                     value = np.nan
# #                 row[key] = value
# #                 if not np.isnan(value):
# #                     inputDict[input_var['varNm']] = value

# #             ph3_1_logger.debug(f"Row data: {row}")
# #             ph3_1_logger.debug(f"Current keys: {current_key}")

# #             if not row:
# #                 ph3_1_logger.error("No valid data processed for row")
# #                 return

# #             # Add to dataframe
# #             self._all_df = pd.concat(
# #                 [self._all_df, pd.DataFrame([row], index=[idx])],
# #                 axis=0
# #             )

# #             # Maintain dataframe size
# #             if len(self._all_df) > self._lookback_length:
# #                 self._all_df = self._all_df.tail(self._lookback_length)

# #             # Prepare data for Cassandra
# #             cassandra_data = {
# #                 'joOpId': process_id,
# #                 'workStationId': data['wsId'],
# #                 'produceList': data.get('prodList', []),
# #                 'createDate': data.get('crDt', datetime.now(pytz.UTC))
# #             }

# #             # Process window with prepared data
# #             self._process_window(current_key, cassandra_data, outputDict, inputDict)

# #             self._conditional_memory_check()

# #         except Exception as e:
# #             ph3_1_logger.error(f"Data processing failed: {str(e)}", exc_info=True)
# #             self._emergency_cleanup()
# #         finally:
# #             gc.collect()

# #     def _map_input_fields(self, data: Dict) -> Dict:
# #         """Map input fields to expected format based on your data structure"""
# #         # 1. Rename fields to match expected names
# #         if 'inVars' in data and 'inputVariableList' not in data:
# #             data['inputVariableList'] = data.pop('inVars')
# #         if 'outVals' in data and 'outputValueList' not in data:
# #             data['outputValueList'] = data.pop('outVals')
# #         if 'prodList' in data and 'produceList' not in data:
# #             data['produceList'] = data['prodList']
            
# #         # 2. Extract unique_code from output values if not present
# #         if 'unique_code' not in data:
# #             if data.get('outputValueList') and isinstance(data['outputValueList'], list):
# #                 # Find first output with unqCd
# #                 for output in data['outputValueList']:
# #                     if 'unqCd' in output:
# #                         data['unique_code'] = output['unqCd']
# #                         break
# #             if 'unique_code' not in data:
# #                 ph3_1_logger.warning("No unique_code found in output values")
# #                 data['unique_code'] = "default_" + str(datetime.now().timestamp())
                
# #         # 3. Ensure required fields exist
# #         if 'wsId' in data and 'workStationId' not in data:
# #             data['workStationId'] = data['wsId']
            
# #         return data

# #     def _process_window(self, current_key: List[str], data: Dict, outputDict: Dict, inputDict: Dict) -> None:
# #         try:
# #             process_data = ProductionCorrelation.saveData(
# #                 message=data
# #             )
# #             log(f"Data saved to ProductionCorrelation: {process_data}")

# #             if process_data:
# #                 summary_data = ProductionCorrelationSummary.update_summary(process_data)
# #                 log(f"Data saved to ProductionCorrelationSummary: {summary_data}")
# #                 if summary_data is None:
# #                     ph3_1_logger.error("Failed to save ProductionCorrelationSummary: No data saved")

# #         except Exception as e:
# #             ph3_1_logger.error(f"Processing error: {str(e)}", exc_info=True)
# #         finally:
# #             gc.collect()

# #     def _is_memory_exceeded(self) -> bool:
# #         process = psutil.Process()
# #         return process.memory_percent() > self._max_memory_percent

# #     def _conditional_memory_check(self) -> None:
# #         self._processing_counter += 1
# #         if self._processing_counter - self._last_memory_check >= self._memory_check_interval:
# #             self._check_memory_usage()
# #             self._last_memory_check = self._processing_counter

# #     def _check_memory_usage(self) -> None:
# #         process = psutil.Process()
# #         mem_info = process.memory_info()
# #         total_mem = psutil.virtual_memory().total
# #         mem_usage_mb = mem_info.rss / (1024 ** 2)
# #         usage_percent = (mem_info.rss / total_mem) * 100
# #         metrics = {'rss_mb': mem_usage_mb, 'usage_percent': usage_percent}
# #         ph3_1_logger.info(f"Memory metrics: {metrics}")
# #         if self._is_memory_exceeded():
# #             self._emergency_cleanup()

# #     def _emergency_cleanup(self) -> None:
# #         ph3_1_logger.warning("Initiating emergency cleanup")
# #         if not self._all_df.empty:
# #             self._all_df = self._all_df.tail(self._lookback_length)
# #         self._past_key = []
# #         gc.collect()
# #         ph3_1_logger.info("Emergency cleanup completed")

# #     def cleanup(self) -> None:
# #         ph3_1_logger.info("Cleaning up Phase3 resources")
# #         self._all_df = pd.DataFrame()
# #         self._past_key = []
# #         gc.collect()




# import pandas as pd
# import numpy as np
# from utils.logger import ph3_1_logger
# import inspect
# import psutil
# import gc
# from typing import Dict, List
# from cassandra_utils.models.correlation_summary import ProductionCorrelation, ProductionCorrelationSummary
# from datetime import datetime
# import pytz

# # Key generation functions
# def input_key_generation(input_var: Dict, prefix: bool = True) -> str:
#     var_name = input_var.get('varNm', '').replace(' ', '_')
#     return f"in_{var_name}" if prefix and var_name else var_name

# def output_key_generation(output: Dict, prefix: bool = True) -> str:
#     eq_name = output.get('eqNm', '').replace(' ', '_')
#     return f"out_{eq_name}" if prefix and eq_name else eq_name

# def log(msg):
#     func = inspect.stack()[1].function
#     ph3_1_logger.debug(f"[{func}] - {msg}")

# class Phase3:
#     def __init__(self, lookback_length: int = 5000, window: int = 50, memory_check_interval: int = 100, max_memory_percent: float = 70):
#         self._window = window
#         self._lookback_length = lookback_length
#         self._all_df = pd.DataFrame()
#         self._past_key = []
#         self._processing_counter = 0
#         self._memory_check_interval = memory_check_interval
#         self._max_memory_percent = max_memory_percent
#         self._last_memory_check = 0

#     def add_data(self, data: Dict) -> None:
#         # 1. Map input fields to expected format
#         data = self._map_input_fields(data)
        
#         # 2. Check required fields
#         required_fields = ['joOpId', 'unique_code', 'wsId', 'outputValueList']
#         if not all(field in data for field in required_fields):
#             missing = [field for field in required_fields if field not in data]
#             ph3_1_logger.error(f"Missing required fields: {missing}. Skipping processing.")
#             return
            
#         # 3. Generate process_id safely
#         process_id = f"{data['joOpId']}_{data['unique_code']}"
#         ph3_1_logger.debug(f"Processing row with process_id: {process_id}")

#         if self._is_memory_exceeded():
#             ph3_1_logger.warning("Memory threshold exceeded - skipping data processing")
#             gc.collect()
#             return

#         try:
#             idx = len(self._all_df)
#             current_key = []
#             outputDict = {}
#             inputDict = {}
#             row = {}

#             # Process output values
#             output_list = data.get('outputValueList', [])
#             ph3_1_logger.debug(f"OutputValueList: {output_list}")
#             for output in output_list:
#                 if not isinstance(output, dict) or 'eqNm' not in output or 'cntRead' not in output:
#                     ph3_1_logger.warning(f"Invalid output entry: {output}")
#                     continue
#                 key = output_key_generation(output, prefix=True)
#                 if not key.startswith('out_'):
#                     ph3_1_logger.warning(f"Invalid output key generated: {key}")
#                     continue
#                 try:
#                     # Handle different numeric types and nulls
#                     value = float(output["cntRead"]) if output["cntRead"] is not None else np.nan
#                 except (ValueError, TypeError):
#                     ph3_1_logger.warning(f"Invalid cntRead in output: {output}")
#                     value = np.nan
#                 row[key] = value
#                 current_key.append(key)
#                 if not np.isnan(value):
#                     outputDict[output['eqNm']] = value

#             # Process input variables
#             input_var_list = data.get('inputVariableList', [])
#             if not isinstance(input_var_list, list):
#                 ph3_1_logger.error(f"inputVariableList is not a list: {input_var_list}")
#                 return
#             for input_var in input_var_list:
#                 if not isinstance(input_var, dict) or 'varNm' not in input_var:
#                     ph3_1_logger.warning(f"Invalid input entry: {input_var}")
#                     continue
#                 key = input_key_generation(input_var, prefix=True)
#                 if not key.startswith('in_'):
#                     ph3_1_logger.warning(f"Invalid input key generated: {key}")
#                     continue
#                 try:
#                     # Handle different numeric types and nulls
#                     value = float(input_var.get('actVal')) if input_var.get('actVal') is not None else np.nan
#                 except (ValueError, TypeError):
#                     ph3_1_logger.warning(f"Invalid actVal in input: {input_var}")
#                     value = np.nan
#                 row[key] = value
#                 current_key.append(key)
#                 if not np.isnan(value):
#                     inputDict[input_var['varNm']] = value

#             ph3_1_logger.debug(f"Row data: {row}")
#             ph3_1_logger.debug(f"Current keys: {current_key}")

#             if not row:
#                 ph3_1_logger.error("No valid data processed for row")
#                 return

#             # Add to dataframe
#             self._all_df = pd.concat(
#                 [self._all_df, pd.DataFrame([row], index=[idx])],
#                 axis=0
#             )

#             # Maintain dataframe size
#             if len(self._all_df) > self._lookback_length:
#                 self._all_df = self._all_df.tail(self._lookback_length)

#             # Prepare data for Cassandra
#             cassandra_data = {
#                 'joOpId': process_id,
#                 'workStationId': data['wsId'],
#                 'produceList': data.get('produceList', []),
#                 'createDate': data.get('crDt', datetime.now(pytz.UTC))
#             }

#             # Process window with prepared data
#             self._process_window(current_key, cassandra_data, outputDict, inputDict)

#             self._conditional_memory_check()

#         except Exception as e:
#             ph3_1_logger.error(f"Data processing failed: {str(e)}", exc_info=True)
#             self._emergency_cleanup()
#         finally:
#             gc.collect()

#     def _map_input_fields(self, data: Dict) -> Dict:
#         """Map input fields to expected format based on your data structure"""
#         # Log raw input data for debugging
#         ph3_1_logger.debug(f"Raw input data: {data}")

#         # 1. Rename fields to match expected names
#         if 'inVars' in data and 'inputVariableList' not in data:
#             data['inputVariableList'] = data.pop('inVars')
#         if 'outVals' in data and 'outputValueList' not in data:
#             data['outputValueList'] = data.pop('outVals')
#         if 'prodList' in data and 'produceList' not in data:
#             data['produceList'] = data['prodList']
        
#         # 2. Ensure inputVariableList is a list
#         if 'inputVariableList' not in data or data['inputVariableList'] is None:
#             data['inputVariableList'] = []
#             ph3_1_logger.warning("inputVariableList was missing or None, set to empty list")
        
#         # 3. Extract unique_code from output values if not present
#         if 'unique_code' not in data:
#             if data.get('outputValueList') and isinstance(data['outputValueList'], list):
#                 # Find first output with unqCd
#                 for output in data['outputValueList']:
#                     if 'unqCd' in output:
#                         data['unique_code'] = output['unqCd']
#                         break
#             if 'unique_code' not in data:
#                 ph3_1_logger.warning("No unique_code found in output values")
#                 data['unique_code'] = "default_" + str(datetime.now().timestamp())
                
#         # 4. Ensure required fields exist
#         if 'wsId' in data and 'workStationId' not in data:
#             data['workStationId'] = data['wsId']
            
#         return data

#     def _process_window(self, current_key: List[str], data: Dict, outputDict: Dict, inputDict: Dict) -> None:
#         try:
#             process_data = ProductionCorrelation.saveData(
#                 message=data
#             )
#             log(f"Data saved to ProductionCorrelation: {process_data}")

#             if process_data:
#                 summary_data = ProductionCorrelationSummary.update_summary(process_data)
#                 log(f"Data saved to ProductionCorrelationSummary: {summary_data}")
#                 if summary_data is None:
#                     ph3_1_logger.error("Failed to save ProductionCorrelationSummary: No data saved")

#         except Exception as e:
#             ph3_1_logger.error(f"Processing error: {str(e)}", exc_info=True)
#         finally:
#             gc.collect()

#     def _is_memory_exceeded(self) -> bool:
#         process = psutil.Process()
#         return process.memory_percent() > self._max_memory_percent

#     def _conditional_memory_check(self) -> None:
#         self._processing_counter += 1
#         if self._processing_counter - self._last_memory_check >= self._memory_check_interval:
#             self._check_memory_usage()
#             self._last_memory_check = self._processing_counter

#     def _check_memory_usage(self) -> None:
#         process = psutil.Process()
#         mem_info = process.memory_info()
#         total_mem = psutil.virtual_memory().total
#         mem_usage_mb = mem_info.rss / (1024 ** 2)
#         usage_percent = (mem_info.rss / total_mem) * 100
#         metrics = {'rss_mb': mem_usage_mb, 'usage_percent': usage_percent}
#         ph3_1_logger.info(f"Memory metrics: {metrics}")
#         if self._is_memory_exceeded():
#             self._emergency_cleanup()

#     def _emergency_cleanup(self) -> None:
#         ph3_1_logger.warning("Initiating emergency cleanup")
#         if not self._all_df.empty:
#             self._all_df = self._all_df.tail(self._lookback_length)
#         self._past_key = []
#         gc.collect()
#         ph3_1_logger.info("Emergency cleanup completed")

#     def cleanup(self) -> None:
#         ph3_1_logger.info("Cleaning up Phase3 resources")
#         self._all_df = pd.DataFrame()
#         self._past_key = []
#         gc.collect()



import pandas as pd
import numpy as np
from utils.logger import ph3_1_logger
import inspect
import psutil
import gc
from typing import Dict, List
from cassandra_utils.models.correlation_summary import ProductionCorrelation, ProductionCorrelationSummary
from datetime import datetime
import pytz

# Key generation functions
def input_key_generation(input_var: Dict, prefix: bool = True) -> str:
    var_name = input_var.get('varNm', '').replace(' ', '_')
    return f"in_{var_name}" if prefix and var_name else var_name

def output_key_generation(output: Dict, prefix: bool = True) -> str:
    eq_name = output.get('eqNm', '').replace(' ', '_')
    return f"out_{eq_name}" if prefix and eq_name else eq_name

def log(msg):
    func = inspect.stack()[1].function
    ph3_1_logger.debug(f"[{func}] - {msg}")

class Phase3:
    def __init__(self, lookback_length: int = 5000, window: int = 50, memory_check_interval: int = 100, max_memory_percent: float = 70):
        self._window = window
        self._lookback_length = lookback_length
        self._all_df = pd.DataFrame()
        self._past_key = []
        self._processing_counter = 0
        self._memory_check_interval = memory_check_interval
        self._max_memory_percent = max_memory_percent
        self._last_memory_check = 0

    def add_data(self, data: Dict) -> None:
        # 1. Map input fields to expected format
        data = self._map_input_fields(data)
        
        # 2. Check required fields
        required_fields = ['joOpId', 'unique_code', 'wsId', 'outputValueList']
        if not all(field in data for field in required_fields):
            missing = [field for field in required_fields if field not in data]
            ph3_1_logger.error(f"Missing required fields: {missing}. Skipping processing.")
            return
            
        # 3. Generate process_id safely
        process_id = f"{data['joOpId']}_{data['unique_code']}"
        ph3_1_logger.debug(f"Processing row with process_id: {process_id}")

        if self._is_memory_exceeded():
            ph3_1_logger.warning("Memory threshold exceeded - skipping data processing")
            gc.collect()
            return

        try:
            idx = len(self._all_df)
            current_key = []
            outputDict = {}
            inputDict = {}
            row = {}

            # Process output values
            output_list = data.get('outputValueList', [])
            ph3_1_logger.debug(f"OutputValueList: {output_list}")
            for output in output_list:
                if not isinstance(output, dict) or 'eqNm' not in output or 'cntRead' not in output:
                    ph3_1_logger.warning(f"Invalid output entry: {output}")
                    continue
                key = output_key_generation(output, prefix=True)
                if not key.startswith('out_'):
                    ph3_1_logger.warning(f"Invalid output key generated: {key}")
                    continue
                try:
                    # Handle different numeric types and nulls
                    value = float(output["cntRead"]) if output["cntRead"] is not None else np.nan
                except (ValueError, TypeError):
                    ph3_1_logger.warning(f"Invalid cntRead in output: {output}")
                    value = np.nan
                row[key] = value
                current_key.append(key)
                if not np.isnan(value):
                    outputDict[output['eqNm']] = value

            # Process input variables
            input_var_list = data.get('inputVariableList', [])
            if not isinstance(input_var_list, list):
                ph3_1_logger.error(f"inputVariableList is not a list: {input_var_list}")
                return
            for input_var in input_var_list:
                if not isinstance(input_var, dict) or 'varNm' not in input_var:
                    ph3_1_logger.warning(f"Invalid input entry: {input_var}")
                    continue
                key = input_key_generation(input_var, prefix=True)
                if not key.startswith('in_'):
                    ph3_1_logger.warning(f"Invalid input key generated: {key}")
                    continue
                try:
                    # Handle different numeric types and nulls
                    value = float(input_var.get('actVal')) if input_var.get('actVal') is not None else np.nan
                except (ValueError, TypeError):
                    ph3_1_logger.warning(f"Invalid actVal in input: {input_var}")
                    value = np.nan
                row[key] = value
                current_key.append(key)
                if not np.isnan(value):
                    inputDict[input_var['varNm']] = value

            ph3_1_logger.debug(f"Row data: {row}")
            ph3_1_logger.debug(f"Current keys: {current_key}")

            if not row:
                ph3_1_logger.error("No valid data processed for row")
                return

            # Add to dataframe
            self._all_df = pd.concat(
                [self._all_df, pd.DataFrame([row], index=[idx])],
                axis=0
            )

            # Maintain dataframe size
            if len(self._all_df) > self._lookback_length:
                self._all_df = self._all_df.tail(self._lookback_length)

            # Prepare data for Cassandra, ensuring UTC timestamp
            create_date = data.get('crDt', datetime.now(pytz.timezone('Asia/Karachi')))
            if isinstance(create_date, (int, float)):
                create_date = datetime.fromtimestamp(create_date / 1000, tz=pytz.timezone('Asia/Karachi')).astimezone(pytz.UTC)
            elif isinstance(create_date, str):
                create_date = datetime.fromisoformat(create_date.replace('Z', '+00:00')).astimezone(pytz.UTC)
            elif isinstance(create_date, datetime) and create_date.tzinfo is None:
                create_date = create_date.replace(tzinfo=pytz.timezone('Asia/Karachi')).astimezone(pytz.UTC)
            
            cassandra_data = {
                'joOpId': process_id,
                'workStationId': data['wsId'],
                'produceList': data.get('produceList', []),
                'createDate': create_date
            }

            # Process window with prepared data
            self._process_window(current_key, cassandra_data, outputDict, inputDict)

            self._conditional_memory_check()

        except Exception as e:
            ph3_1_logger.error(f"Data processing failed: {str(e)}", exc_info=True)
            self._emergency_cleanup()
        finally:
            gc.collect()

    def _map_input_fields(self, data: Dict) -> Dict:
        """Map input fields to expected format based on your data structure"""
        # Log raw input data for debugging
        ph3_1_logger.debug(f"Raw input data: {data}")

        # 1. Rename fields to match expected names
        if 'inVars' in data and 'inputVariableList' not in data:
            data['inputVariableList'] = data.pop('inVars')
        if 'outVals' in data and 'outputValueList' not in data:
            data['outputValueList'] = data.pop('outVals')
        if 'prodList' in data and 'produceList' not in data:
            data['produceList'] = data['prodList']
        
        # 2. Ensure inputVariableList is a list
        if 'inputVariableList' not in data or data['inputVariableList'] is None:
            data['inputVariableList'] = []
            ph3_1_logger.warning("inputVariableList was missing or None, set to empty list")
        
        # 3. Extract unique_code from output values if not present
        if 'unique_code' not in data:
            if data.get('outputValueList') and isinstance(data['outputValueList'], list):
                # Find first output with unqCd
                for output in data['outputValueList']:
                    if 'unqCd' in output:
                        data['unique_code'] = output['unqCd']
                        break
            if 'unique_code' not in data:
                ph3_1_logger.warning("No unique_code found in output values")
                data['unique_code'] = "default_" + str(datetime.now().timestamp())
                
        # 4. Ensure required fields exist
        if 'wsId' in data and 'workStationId' not in data:
            data['workStationId'] = data['wsId']
            
        return data

    def _process_window(self, current_key: List[str], data: Dict, outputDict: Dict, inputDict: Dict) -> None:
        try:
            process_data = ProductionCorrelation.saveData(
                message=data
            )
            log(f"Data saved to ProductionCorrelation: {process_data}")

            if process_data:
                summary_data = ProductionCorrelationSummary.update_summary(process_data)
                log(f"Data saved to ProductionCorrelationSummary: {summary_data}")
                if summary_data is None:
                    ph3_1_logger.error("Failed to save ProductionCorrelationSummary: No data saved")

        except Exception as e:
            ph3_1_logger.error(f"Processing error: {str(e)}", exc_info=True)
        finally:
            gc.collect()

    def _is_memory_exceeded(self) -> bool:
        process = psutil.Process()
        return process.memory_percent() > self._max_memory_percent

    def _conditional_memory_check(self) -> None:
        self._processing_counter += 1
        if self._processing_counter - self._last_memory_check >= self._memory_check_interval:
            self._check_memory_usage()
            self._last_memory_check = self._processing_counter

    def _check_memory_usage(self) -> None:
        process = psutil.Process()
        mem_info = process.memory_info()
        total_mem = psutil.virtual_memory().total
        mem_usage_mb = mem_info.rss / (1024 ** 2)
        usage_percent = (mem_info.rss / total_mem) * 100
        metrics = {'rss_mb': mem_usage_mb, 'usage_percent': usage_percent}
        ph3_1_logger.info(f"Memory metrics: {metrics}")
        if self._is_memory_exceeded():
            self._emergency_cleanup()

    def _emergency_cleanup(self) -> None:
        ph3_1_logger.warning("Initiating emergency cleanup")
        if not self._all_df.empty:
            self._all_df = self._all_df.tail(self._lookback_length)
        self._past_key = []
        gc.collect()
        ph3_1_logger.info("Emergency cleanup completed")

    def cleanup(self) -> None:
        ph3_1_logger.info("Cleaning up Phase3 resources")
        self._all_df = pd.DataFrame()
        self._past_key = []
        gc.collect()