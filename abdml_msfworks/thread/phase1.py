from tensorflow import keras
from utils.logger import ph3_logger, feature_logger
import os
from statistics import mode
from cassandra_utils.models.dw_tbl_anomalies_2 import DwTblAnomalies2
from cassandra_utils.models.dw_tbl_multiple_anomalies import DwTblMultipleAnomalies
import pandas as pd
import numpy as np
from scipy import stats
import tensorflow as tf
from sklearn.model_selection import train_test_split
from modules.key_generator import output_key_generation
import inspect
import random
from cassandra_utils.models.phase1_report_anomaly import save_anomaly_data as save_phase1_anomaly_data
from cassandra_utils.models.phase2_anomaly_model import save_anomaly_data2
from datetime import datetime, timedelta
import pytz
import json

def log(msg):
    func = inspect.stack()[1].function
    feature_logger.debug(f"[{func}] - {msg}")

def safe_int(value, default=0):
    if value is None or value == '':
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def float_mode(data, bin_size=0.1):
    if not data:
        return 0.0
    bins = np.arange(min(data), max(data) + bin_size, bin_size)
    binned = np.digitize(data, bins)
    mode_bin = np.bincount(binned).argmax()
    return round(bins[mode_bin], 2)

def compare_measDt_granularity(data):
    meas_dt = data.get("measDt", "")
    granularities = []
    
    if not meas_dt or not meas_dt.isdigit():
        ph3_logger.warning(f"compare_measDt_granularity - Invalid or missing measDt: {meas_dt}. No granularities selected.")
        return granularities
    
    try:
        timestamp = int(meas_dt) / 1000
        weekly_time = datetime.fromtimestamp(timestamp, tz=pytz.UTC)
        ph3_logger.debug(f"compare_measDt_granularity - Converted measDt {meas_dt} to {weekly_time}")
    except (ValueError, TypeError) as e:
        ph3_logger.error(f"compare_measDt_granularity - Invalid measDt format: {meas_dt}. Error: {e}. No granularities selected.")
        return granularities
    
    rules = [
        {
            "name": "one_minute",
            "start": weekly_time.replace(second=0, microsecond=0),
            "duration": timedelta(minutes=1)
        },
        {
            "name": "ten_minute",
            "start": weekly_time.replace(minute=(weekly_time.minute // 10) * 10, second=0, microsecond=0),
            "duration": timedelta(minutes=10)
        },
        {
            "name": "shiftly",
            "start": weekly_time.replace(hour=(weekly_time.hour // 8) * 8, minute=0, second=0, microsecond=0),
            "duration": timedelta(hours=8)
        }
    ]
    
    for rule in rules:
        if weekly_time >= rule["start"] and weekly_time < rule["start"] + rule["duration"]:
            granularities.append(rule["name"])
            ph3_logger.debug(f"compare_measDt_granularity - Selected granularity: {rule['name']} for measDt: {meas_dt}")
    
    if not granularities:
        ph3_logger.warning(f"compare_measDt_granularity - No granularity matched for measDt: {meas_dt}")
    
    return granularities

class Phase1:
    def __init__(self, lookback_length=5000, window=50):
        self._lookback = lookback_length
        self._window = window
        self.__correlationThreshold = 1.5
        self._output_training_dict = {}
        self._output_predict_dict = {}
        self._all_data_list = []
        self._model_cache = {}
        self._correlation_phase2 = []
        self._past_keys_phase2 = []
        self.__current_mode = {}

    def add_allData_predict(self, data):
        if data is not None:
            self._all_data_list.append(data)

    def add_output_training(self, dataList):
        for data in dataList:
            key = output_key_generation(data)
            if key not in self._output_training_dict:
                self._output_training_dict[key] = []
            self._output_training_dict[key].append(float(data['cntRead']))

    def add_output_predict(self, main_data, outputDataList):
        ph3_logger.debug(f"add_output_predict - The input outputDatalist Len is {len(outputDataList)}")
        tmp = [data for data in outputDataList if bool(data['anomDetAct'])]
        outputDataList = tmp
        ph3_logger.debug(f"add_output_predict - The input outputDatalist Len is after anomDetAct {len(outputDataList)}")

        for data in outputDataList:
            key = output_key_generation(data)
            if key not in self._output_predict_dict:
                self._output_predict_dict[key] = []
            self._output_predict_dict[key].append(float(data['cntRead']))
            self._output_predict_dict[key] = self._output_predict_dict[key][-self._lookback:]
            ph3_logger.debug(f"add_output_predict - message is saved to output_predict_dict key - {key} len {self._output_predict_dict[key]}")
            self.phase1_prediction(data, main_data, key)

        self.phase2_prediction(outputDataList=outputDataList, main_data=main_data)

    def phase1_prediction(self, data, main_data, key):
        log(f"getting y for key {key}")
        ph3_logger.debug(f"phase1_prediction - getting y value for {key}")
        ph3_logger.debug(f"phase1_prediction - measDt: {data.get('measDt', 'missing')}")
        actual_value = float(data['cntRead'])
        y = self.predict_mode_y(key)
        history = self._output_predict_dict[key][-self._lookback:]
        std_dev = np.std(history) if len(history) > 10 else 0.1
        upper = y + 2 * std_dev
        lower = y - 2 * std_dev
        anomaly = actual_value > upper or actual_value < lower
        ph3_logger.debug(f"phase1_prediction - key={key}, actual={actual_value}, predicted={y}, upper={upper}, lower={lower}, anomaly={anomaly}")
        
        try:
            data_saved = DwTblAnomalies2.saveData(message=data, main_message=main_data, y=y, key=key)
            ph3_logger.debug(f"phase1_prediction - data saved to the database single sensor {data_saved} ")
            
        except Exception as e:
            ph3_logger.error(f"add_output_predict - Failed to save to dw_tbl_anomalies_2: {e}")
            raise
        
        granularities = compare_measDt_granularity(data)
        table_map = {
            "one_day": "tbl_one_day_anomaly",
            "one_hour": "tbl_one_hour_anomaly",
            "one_minute": "tbl_one_minute_anomaly",
            "one_week": "tbl_one_week_anomaly",
            "shiftly": "tbl_shiftly_anomaly",
            "ten_minute": "tbl_ten_minute_anomaly"
        }
        
        for granularity in granularities:
            expected_table = table_map.get(granularity, "unknown")
            ph3_logger.debug(f"phase1_prediction - Attempting to save with granularity={granularity}, expected table={expected_table}")
            if granularity == "shiftly" and expected_table != "tbl_shiftly_anomaly":
                ph3_logger.error(f"phase1_prediction - Incorrect table mapping for shiftly granularity: {expected_table}")
                continue
            try:
                save_phase1_anomaly_data(message=data, main_message=main_data, y=y, key=key, granularity=granularity)
                ph3_logger.debug(f"phase1_prediction - Successfully saved to {expected_table} with granularity {granularity}")
            except Exception as e:
                ph3_logger.error(f"add_output_predict - Failed to save to anomaly table with granularity {granularity}: {e}")
                raise
        
        return anomaly

    def phase2_prediction(self, outputDataList, main_data):
        current_values = {}
        current_keys = []
        anomaly = 0
        upper_limit = None
        lower_limit = None
        sensor_dict = {}

        # Validate integer fields in outputDataList
        integer_fields = [
            "measDocId", "eqId", "eqMeasPtId", "eqCodeGrpHdrId", "maxErrCntHr",
            "maxErrCntMin", "maxErrCntShft", "stockId"
        ]
        for data in outputDataList:
            for field in integer_fields:
                data[field] = safe_int(data.get(field), 0)

        for data in outputDataList:
            key = output_key_generation(data)
            current_values[key] = float(data['cntRead'])
            current_keys.append(key)

            if key not in self._output_predict_dict:
                ph3_logger.error(f"phase2_prediction - No data exists for this key")
                raise

        if not current_keys:
            ph3_logger.error(f"phase2_prediction - No key is passed")
            raise

        ph3_logger.debug(f"phase2_prediction - keys and real value {current_values}")

        for key in current_values:
            temp = {}
            y = self.predict_mode_y(key)
            current = current_values[key]
            history = self._output_predict_dict[key][-self._lookback:]
            std_dev = np.std(history) if len(history) > 10 else 0.1
            if std_dev == 0:
                std_dev = 0.1
                ph3_logger.debug(f"phase2_prediction - Zero std_dev for {key}, using minimum std_dev=0.1")

            upper = y + 3 * std_dev
            lower = y - 3 * std_dev

            if upper < 0.1:
                upper = 0.1
                lower = -0.1

            sensor_anomaly = 1 if (current > upper or current < lower) else 0
            if sensor_anomaly:
                anomaly = 1
                ph3_logger.debug(f"phase2_prediction - Anomaly detected for {key}: actual={current}, predicted={y}, upper={upper}, lower={lower}, std_dev={std_dev}")

            temp['actual_value'] = str(current)
            temp['predicted_value'] = str(y)
            temp['predicted_upper'] = str(upper)
            temp['predicted_lower'] = str(lower)
            temp['anomaly'] = str(sensor_anomaly)
            sensor_dict[key] = temp

            if upper_limit is None or upper > upper_limit:
                upper_limit = upper
            if lower_limit is None or lower < lower_limit:
                lower_limit = lower

        if len(current_keys) >= 2:
            temp = {}
            for key in current_keys:
                li = self._output_predict_dict[key][-self._window:]
                if len(li) < self._window:
                    padding = [li[-1] if li else 0.0] * (self._window - len(li))
                    temp[key] = li + padding
                else:
                    temp[key] = li

            ph3_logger.debug(f"phase2_prediction - temp data for correlation - {temp}")

            EnoughLen = all(len(temp[key]) >= self._window for key in temp)
            if EnoughLen:
                padding_check = {}
                for key in temp:
                    data = temp[key]
                    last_value = data[-1]
                    count = 0
                    for value in reversed(data):
                        if value == last_value:
                            count += 1
                        else:
                            break
                    padding_check[key] = count

                heavily_padded = any(padding_check[key] > self._window // 2 for key in padding_check)
                ph3_logger.debug(f"phase2_prediction - Padding check: {padding_check}, heavily_padded={heavily_padded}")

                if heavily_padded:
                    ph3_logger.warning(f"phase2_prediction - Skipping correlation due to heavy padding: {padding_check}")
                else:
                    all_low_variability = True
                    for key in temp:
                        data = temp[key]
                        std_dev = np.std(data) if len(data) > 10 else 0.1
                        if std_dev >= 0.1:
                            all_low_variability = False
                            break

                    if all_low_variability:
                        ph3_logger.debug(f"phase2_prediction - Skipping correlation: all sensors have low variability (std_dev < 0.1)")
                    else:
                        cor = pd.DataFrame(temp).corr().fillna(0)
                        self._correlation_phase2.append(cor)

                        if set(self._past_keys_phase2) != set(current_keys):
                            ph3_logger.warning("phase2_prediction - keys changed, Resetting correlation")
                            self._correlation_phase2 = []
                        elif len(self._correlation_phase2) >= 2:
                            diff = np.linalg.norm(self._correlation_phase2[-1] - self._correlation_phase2[-2])
                            ph3_logger.debug(f"phase2_prediction - Correlation difference: {diff}")
                            if diff > self.__correlationThreshold:
                                anomaly = 1
                                ph3_logger.debug(f"phase2_prediction - Anomaly detected due to correlation change: {diff}")
                            elif diff >= 0.6 and anomaly != 1:
                                anomaly = diff * 1.2
                                ph3_logger.debug(f"phase2_prediction - Scaled anomaly due to correlation change: diff={diff}, anomaly={anomaly}")

        self._past_keys_phase2 = current_keys

        key = "_".join(current_keys)
        data2 = DwTblMultipleAnomalies.saveData(message=outputDataList[0], main_message=main_data,
                                               anomaly=anomaly, key=key,
                                               upper=upper_limit, lower=lower_limit,
                                               sensor_dict=sensor_dict)
        ph3_logger.debug(f"phase2_prediction - data saved to the database multiple sensor {data2} with anomaly={anomaly}")
        granularities = compare_measDt_granularity(outputDataList[0])
        table_map = {
            "one_day": "tbl_p2_one_day_anomaly",
            "one_hour": "tbl_p2_one_hour_anomaly",
            "one_minute": "tbl_p2_one_minute_anomaly",
            "one_week": "tbl_p2_one_week_anomaly",
            "shiftly": "tbl_p2_shiftly_anomaly",
            "ten_minute": "tbl_p2_ten_minute_anomaly"
        }

        for granularity in granularities:
            expected_table = table_map.get(granularity, "unknown")
            ph3_logger.debug(f"phase2_prediction - Attempting to save with granularity={granularity}, expected table={expected_table}")
            try:
                data = save_anomaly_data2(
                    message=outputDataList[0],
                    main_message=main_data,
                    anomaly=anomaly,
                    key=key,
                    upper=upper_limit,
                    lower=lower_limit,
                    sensor_dict=sensor_dict,
                    granularity=granularity
                )
                ph3_logger.debug(f"main phase2_prediction - Successfully saved to {expected_table} with granularity {granularity}")
            except Exception as e:
                ph3_logger.error(f"phase2_prediction - Failed to save to {expected_table} with granularity {granularity}: {e}")
                raise

        return anomaly

    def load_model(self, key='temp'):
        model_filename = key + ".keras"
        if model_filename in self._model_cache:
            return self._model_cache[model_filename]

        model_path = os.path.join('models', model_filename)
        if not os.path.exists(model_path):
            if len(self._output_training_dict[key]) < self._window * 10:
                ph3_logger.warning(f"load_model - not enough data for key {key} len - {len(self._output_training_dict[key])}")
                return None
            else:
                data = self._output_training_dict[key]
                ph3_logger.debug(f"load_model - Training data for key {key}: len={len(data)}, min={min(data)}, max={max(data)}")
                X_list = []
                y_list = []

                for i in range(len(data) - self._window):
                    x = data[i:i+self._window]
                    y = data[i+self._window]
                    X_list.append(x)
                    y_list.append(y)

                    if random.random() < 0.1:
                        temp_num = y + y * random.uniform(-1, 1)
                        x = data[i:i+self._window]
                        X_list.append(x)
                        y_list.append(temp_num)

                x = np.array(X_list)
                y = np.array(y_list)
                X_train, X_test, y_train, y_test = train_test_split(x, y, shuffle=False)

                model = tf.keras.models.Sequential([
                    tf.keras.layers.Input(shape=(self._window, 1)),
                    tf.keras.layers.LSTM(64, return_sequences=True),
                    tf.keras.layers.LSTM(32),
                    tf.keras.layers.Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse', metrics=['mse'])
                model.fit(X_train, y_train, epochs=40, validation_data=(X_test, y_test), verbose=0)
                model.save(model_path)

        model = tf.keras.models.load_model(model_path)
        self._model_cache[model_filename] = model
        return model

    def predict_mode_y(self, key):
        li = [float(x) for x in self._output_predict_dict[key][-self._window:]]
        log(f"1- li for key {key} - li: {li}")
        ph3_logger.debug(f"predict_mode_y - Historical data for {key}: len={len(li)}, min={min(li) if li else 0}, max={max(li) if li else 0}, mean={np.mean(li) if li else 0}")

        if len(set(li)) == 1:
            constant_value = li[0]
            log(f"predict_mode_y - Data for {key} is constant: {constant_value}")
            return constant_value

        weights = np.linspace(0.1, 1.0, len(li))
        weighted_avg = np.average(li, weights=weights) if li else 0.0
        log(f"2- Weighted moving average for {key}: {weighted_avg:.2f}")

        if key not in self.__current_mode:
            self.__current_mode[key] = stats.trim_mean(li, 0.2) if li else 0.0
            log(f"3- Using trim mean for {key} - {self.__current_mode[key]:.2f}")
            ph3_logger.debug(f"predict_mode_y - the return value is inside not key {self.__current_mode[key]}")
        else:
            mod = float_mode(li, bin_size=0.1)
            history = self._output_predict_dict[key][-self._window:]
            std_dev = np.std(history) if len(history) > 10 else 0.1
            upper = self.__current_mode[key] + 2 * std_dev
            lower = self.__current_mode[key] - 2 * std_dev

            if mod > upper or mod < lower:
                self.__current_mode[key] = mod
                log(f"4- Anomaly new mod {key} - {self.__current_mode[key]:.2f}")

        model_output = self.load_model(key=key)
        if model_output:
            if len(li) < self._window:
                log(f"5- Insufficient data for {key}: len={len(li)}, padding to {self._window}")
                padding = [li[-1] if li else 0.0] * (self._window - len(li))
                li_extended = li + padding
            else:
                li_extended = li

            inputLi = np.array(li_extended).reshape(1, self._window, 1)
            returnOutput = round(model_output.predict(inputLi, verbose=0)[0][0], 2)
            log(f"6- Model output for {key} - {returnOutput}")
            log(f"6- DEBUG - input: {inputLi.flatten()} - model predicted: {model_output.predict(inputLi, verbose=0)[0][0]} rounded: {returnOutput}")

            recent_range = (min(li), max(li)) if li else (0, 0)
            if returnOutput < recent_range[0] - 0.1 or returnOutput > recent_range[1] + 0.1:
                log(f"7- Model prediction {returnOutput} is outside recent range {recent_range}, falling back to weighted average")
                returnOutput = round(weighted_avg, 2)
        else:
            returnOutput = self.__current_mode[key]

        log(f"8- Final value for {key} - {returnOutput}")
        return returnOutput
