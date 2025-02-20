#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MySQL-based VEST Database Access and Plotting

This module provides convenient functions to connect to VEST's MySQL database
via a connection pool, load data by shot/field, correct time arrays for DAQ
triggers, retrieve date or shot lists, and plot results.
"""

import os
import re
import time
import numpy as np
import mysql.connector
import matplotlib.pyplot as plt
import json

from mysql.connector.pooling import MySQLConnectionPool

import os
import yaml
from cryptography.fernet import Fernet

# Define the paths for the encryption key and configuration file
KEY_FILE = os.path.expanduser("~/.vest/encryption_key.key")
CONFIG_FILE = os.path.expanduser("~/.vest/database_raw_info.yaml")


# Function to load or generate an encryption key
def load_or_generate_key():
    # Ensure the directory for the key file exists
    key_dir = os.path.dirname(KEY_FILE)
    os.makedirs(key_dir, exist_ok=True)

    if os.path.exists(KEY_FILE):
        # Load the existing key
        with open(KEY_FILE, "rb") as key_file:
            return key_file.read()
    else:
        # Generate a new key and save it to the key file
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as key_file:
            key_file.write(key)
        return key


# Class to manage encrypted configuration settings
class SecureConfigManager:
    def __init__(self):
        # Initialize encryption key and cipher
        self.key = load_or_generate_key()
        self.cipher = Fernet(self.key)

    def encrypt(self, plain_text):
        # Encrypt plain text and return the encrypted string
        return self.cipher.encrypt(plain_text.encode()).decode()

    def decrypt(self, encrypted_text):
        # Decrypt encrypted text and return the plain text string
        return self.cipher.decrypt(encrypted_text.encode()).decode()

    def get_info(self):
        """
        Prompt the user to input configuration details and save them to a YAML file.
        """
        hostname = input("Enter the database hostname: ")
        username = input("Enter the database username: ")
        password = input("Enter the database password: ")
        database = "VEST"

        # Encrypt the password before saving
        encrypted_password = self.encrypt(password)

        # Create a dictionary to hold the configuration
        config_data = {
            "hostname": hostname,
            "username": username,
            "password": encrypted_password,
            "database": database,
        }

        # Save the configuration to the YAML file
        with open(CONFIG_FILE, "w") as file:
            yaml.dump(config_data, file)
        print(f"Configuration saved to {CONFIG_FILE}")

    def load_config(self):
        """
        Load configuration from the YAML file. If the file does not exist,
        prompt the user to input details.
        """
        if os.path.exists(CONFIG_FILE):
            # Load configuration from the YAML file
            with open(CONFIG_FILE, "r") as file:
                config_data = yaml.safe_load(file)

            hostname = config_data["hostname"]
            username = config_data["username"]
            password = self.decrypt(config_data["password"])
            database = config_data["database"]

            return hostname, username, password, database
        else:
            # If configuration does not exist, initialize it
            print(f"No configuration file found at {CONFIG_FILE}. Initializing setup...")
            self.get_info()
            return self.load_config()


# Main configuration function
def configuration():
    """
    Main function to handle configuration management for vaft.db.raw.
    This function loads configuration details and provides them for use
    in the library.
    """
    manager = SecureConfigManager()
    hostname, username, password, database = manager.load_config()

    # Print the loaded configuration for debugging (can be removed in production)
    print("Configuration loaded:")
    print(f"Hostname: {hostname}")
    print(f"Username: {username}")
    print(f"Password: {password}")
    print(f"Database: {database}")

    # Return the configuration details
    return hostname, username, password, database





# ------------------------------------------------------------------------
# Global Database Pool
# ------------------------------------------------------------------------
DB_POOL = None
HOSTNAME, USERNAME, PASSWORD, DATABASE = configuration()


def init_pool() -> None:
    """
    Initializes the global MySQLConnectionPool. Must be called before loading data.

    Uses the vaft.database.raw.configuration() function to retrieve database
    host, user, password, and database name, which are then used to create
    a connection pool with up to four connections.
    """
    global DB_POOL
    DB_POOL = MySQLConnectionPool(
        pool_name="mypool",
        pool_size=4,  # maximum number of concurrent connections
        host=HOSTNAME,
        database=DATABASE,
        user=USERNAME,
        password=PASSWORD
    )



def _load_from_shot_waveform_2(
    db_conn: mysql.connector.connection_cext.CMySQLConnection,
    shot: int,
    field: int
) -> tuple:
    """
    Loads shot data (time, value) from table 'shotDataWaveform_2'
    for the specified shot and field code.

    :param db_conn: Active MySQL database connection.
    :param shot: Shot number.
    :param field: Field code.
    :return: (time_array, data_array) as np.ndarrays. If no data found, returns ([0.],[0.]).
    """
    cursor = db_conn.cursor()
    com = (
        "SELECT shotDataWaveformTime, shotDataWaveformValue "
        "FROM shotDataWaveform_2 "
        f"WHERE shotCode = {shot} AND shotDataFieldCode = {field} "
        "ORDER BY shotDataWaveformTime ASC"
    )
    cursor.execute(com)
    result = np.array(cursor.fetchall())
    cursor.close()

    if result.size > 0:
        return result.T[0], result.T[1]
    return np.array([0.0]), np.array([0.0])


def _load_from_shot_waveform_3(
    db_conn: mysql.connector.connection_cext.CMySQLConnection,
    shot: int,
    field: int
) -> tuple:
    """
    Loads shot data (time, value) from table 'shotDataWaveform_3'
    for the specified shot and field code.

    :param db_conn: Active MySQL database connection.
    :param shot: Shot number.
    :param field: Field code.
    :return: (time_array, data_array) as np.ndarrays. If multiple or zero rows
             are found for the same shot/field, returns ([0.],[0.]) or logs a warning.
    """
    cursor = db_conn.cursor()
    com = (
        "SELECT shotDataWaveformTime, shotDataWaveformValue "
        "FROM shotDataWaveform_3 "
        f"WHERE shotCode = {shot} AND shotDataFieldCode = {field}"
    )
    cursor.execute(com)
    myresult = cursor.fetchall()
    cursor.close()

    if len(myresult) != 1:
        print(
            f"Warning: shot={shot}, field={field} has multiple/no rows. "
            "Returning ([0.],[0.])."
        )
        return np.array([0.0]), np.array([0.0])

    shot_time_str = myresult[0][0]
    shot_val_str = myresult[0][1]

    # Remove bracket characters
    shot_time_str = re.sub(r"[\[\]]", "", shot_time_str)
    shot_val_str = re.sub(r"[\[\]]", "", shot_val_str)

    # Split by comma and convert to float
    time_vals = np.array([float(x) for x in shot_time_str.split(",")])
    data_vals = np.array([float(x) for x in shot_val_str.split(",")])

    return time_vals, data_vals


def _load_from_sample_file(
    shot: int,
    fields: list,
    sample_opt
) -> tuple:
    import os
    import json

    if isinstance(sample_opt, str):
        json_path = sample_opt
    else:
        json_path = f"SHOT_{shot}.json"

    if not os.path.isfile(json_path):
        print(f"[_load_from_sample_file] Sample JSON file not found: {json_path}")
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        shot_json = json.load(f)

    file_shot = shot_json.get("shot")
    if file_shot is not None and file_shot != shot:
        print(f"[_load_from_sample_file] Warning: JSON shot={file_shot}, requested shot={shot}.")

    data_dict = shot_json.get("data", {})
    if not data_dict:
        print(f"[_load_from_sample_file] No 'data' found in JSON for shot={shot}")
        return None

    if not fields:
        fields = list(map(int, data_dict.keys()))

    import numpy as np

    time_arrays = []
    data_arrays = []

    for fld in fields:
        fld_str = str(fld)
        if fld_str not in data_dict:
            print(f"[_load_from_sample_file] Field {fld} not found in JSON. Skipping...")
            continue

        time_list = data_dict[fld_str].get("time", [])
        data_list = data_dict[fld_str].get("data", [])
        tvals = np.array(time_list, dtype=float)
        dvals = np.array(data_list, dtype=float)

        time_arrays.append(tvals)
        data_arrays.append(dvals)

    if not time_arrays:
        print(f"[_load_from_sample_file] No valid fields loaded from JSON for shot={shot}.")
        return None

    time_ref = time_arrays[0]
    stack_list = []
    for arr in data_arrays:
        min_len = min(len(time_ref), len(arr))
        stack_list.append(arr[:min_len])
        time_ref = time_ref[:min_len]

    data_stack = np.column_stack(stack_list)

    if len(fields) == 1:
        return time_ref, data_stack.ravel()
    else:
        return time_ref, data_stack


def _daq_trigger_time_correction(shot: int, time_arr: np.ndarray) -> np.ndarray:
    """
    Corrects the time array for DAQ trigger delay for fast-DAQ channels.

    If max(time) <= 0.101, we shift the times to account for the second
    A/I trigger. The shift is determined by shot-specific breakpoints.
    """
    if time_arr.max() <= 0.101:
        if shot < 41446:
            trigger_sec = 0.24
        elif 41446 <= shot <= 41451:
            trigger_sec = 0.26
        elif 41452 <= shot <= 41659:
            trigger_sec = 0.24
        else:  # shot >= 41660
            trigger_sec = 0.26
    else:
        trigger_sec = 0.0

    return time_arr + trigger_sec


def load(
    shot: int,
    fields=None,
    max_retries: int = 3,
    daq_type: int = None,
    sample_opt=False
) -> tuple:
    """
    High-level data loader for the VEST database. Must run init_pool() first.

    1) sample_opt:
       - False (default): DB에서 로드
       - True or str(파일 경로): 샘플 JSON 파일에서 로드
    2) DB 로직:
       - shot 범위에 따라 _load_from_shot_waveform_2 / _load_from_shot_waveform_3 사용
       - 여러 필드를 2D로 스택
    3) fields:
       - None이면 빈 리스트로 간주(샘플 파일의 경우 => 모든 필드, DB로직의 경우 => 직접 지정 필요)
       - int -> 단일 필드
       - list -> 다중 필드
    4) 반환:
       - 단일 필드 => (time, data_1D)
       - 다중 필드 => (time, data_2D)
    """
    # fields 정규화
    if fields is None:
        fields = []
    elif isinstance(fields, int):
        fields = [fields]

    # ------------------------------
    # A) sample_opt인 경우: JSON 파일 로딩
    # ------------------------------
    if sample_opt:
        from_sample = _load_from_sample_file(shot, fields, sample_opt)
        if from_sample is None:
            return None
        else:
            # time_arr, data_arr(2D or 1D)
            return from_sample

    # ------------------------------
    # B) DB에서 로딩
    # ------------------------------
    global DB_POOL
    if DB_POOL is None:
        print("Error: DB_POOL is not initialized. Run init_pool() first.")
        return None

    attempts = 0
    while attempts < max_retries:
        conn = None
        try:
            conn = DB_POOL.get_connection()
            time_arrays, data_arrays = [], []

            # 최소 하나 이상의 필드는 지정되어 있어야 DB에서 로드 가능
            if not fields:
                print("[load] No fields specified for DB loading.")
                return None

            for fld in fields:
                if 29349 < shot <= 42190:
                    tvals, dvals = _load_from_shot_waveform_2(conn, shot, fld)
                elif shot > 42190:
                    tvals, dvals = _load_from_shot_waveform_3(conn, shot, fld)
                else:
                    print("Shot number out of range for these tables.")
                    return None

                time_arrays.append(tvals)
                data_arrays.append(dvals)

            if not time_arrays:
                print(f"[load] No data found for shot={shot} from DB.")
                return None

            # 여러 필드를 스택
            time_ref = time_arrays[0]
            stack_list = []
            for arr in data_arrays:
                min_len = min(len(time_ref), len(arr))
                stack_list.append(arr[:min_len])
                time_ref = time_ref[:min_len]

            # DAQ trigger delay correction 
            time_ref = _daq_trigger_time_correction(shot, time_ref)


            data_stack = np.column_stack(stack_list)

            if len(fields) == 1:
                return time_ref, data_stack.ravel()
            else:
                return time_ref, data_stack

        except mysql.connector.Error as err:
            print(f"Error connecting to MySQL (try {attempts+1}): {err}")
            attempts += 1
            time.sleep(1)
        finally:
            if conn and conn.is_connected():
                conn.close()

    print("[load] Error: Could not retrieve data after max_retries.")
    return None


def name(field: int) -> tuple:
    """
    Retrieves the shotDataFieldName and shotDataFieldRemark from the
    'shotDataField' table for a given field code.

    :param field: field code integer.
    :return: (field_name, field_remark) as strings, or (None, None).
    """
    global DB_POOL
    if DB_POOL is None:
        print("Error: DB_POOL is not initialized. Run init_pool() first.")
        return None, None

    conn = DB_POOL.get_connection()
    cursor = conn.cursor()
    # It's unclear if a SELECT statement is missing 'SELECT shotDataFieldName...' 
    # so we'll fix that:
    com = (
        "SELECT shotDataFieldName, shotDataFieldRemark "
        f"FROM shotDataField WHERE shotDataFieldCode = {field}"
    )
    cursor.execute(com)
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    if result is not None:
        return result[0], result[1]
    return None, None


def plot(
    shots,
    fields,
    semilogy_opt: bool = False,
    norm_opt: bool = False,
    xlims=None,
) -> None:
    """
    Plots data for the 3 standard scenarios:

    1. Single shot, single field -> single line plot.
    2. Multiple shots, single field -> multiple lines, one per shot.
    3. Single shot, multiple fields -> multiple lines, one per field.

    For multiple data sets, a legend is shown. Units/names fetched from DB.
    :param shots: int or list of int
    :param fields: int or list of int
    :param semilogy_opt: If True, uses semilogy. Defaults to False.
    :param norm_opt: If True, normalizes data to (-1, 1). Defaults to False.
    """
    if isinstance(shots, int):
        shots = [shots]
    if isinstance(fields, int):
        fields = [fields]

    def normalize(data):
        """Normalizes the data to the range (-1, 1)."""
        return 2 * (data - data.min()) / (data.max() - data.min()) - 1

    # 1) Single shot, single field
    if len(shots) == 1 and len(fields) == 1:
        shot = shots[0]
        field = fields[0]
        # load
        loaded = load(shot, field)
        if loaded is None:
            print("No data loaded.")
            return
        time_vals, data_vals = loaded

        if norm_opt:
            data_vals = normalize(data_vals)

        fname, funit = name(field)
        label_str = f"shot {shot}, field {field}"

        if semilogy_opt:
            plt.semilogy(time_vals, data_vals, label=label_str)
        else:
            plt.plot(time_vals, data_vals, label=label_str)

        norm_status = "Normalized" if norm_opt else "Raw"
        plt.title(f"shot={shot}, field={field}, name={fname} ({norm_status})")
        plt.xlabel("time (s)")
        if xlims is not None and len(xlims) == 2:
            plt.xlim(xlims)
        plt.ylabel(f"{funit}")
        plt.legend()
        plt.grid(True)
        plt.show()

    # 2) multiple shots, single field
    elif len(shots) > 1 and len(fields) == 1:
        field = fields[0]
        fname, funit = name(field)

        for sh in shots:
            loaded = load(sh, field)
            if loaded is None:
                continue
            tvals, dvals = loaded

            if norm_opt:
                dvals = normalize(dvals)

            if semilogy_opt:
                plt.semilogy(tvals, dvals, label=f"shot {sh}")
            else:
                plt.plot(tvals, dvals, label=f"shot {sh}")

        norm_status = "Normalized" if norm_opt else "Raw"
        plt.title(f"field={field}, name={fname} ({norm_status})")
        if xlims is not None and len(xlims) == 2:
            plt.xlim(xlims)
        plt.xlabel("time (s)")
        plt.ylabel(f"{funit}")
        plt.legend()
        plt.grid(True)
        plt.show()

    # 3) single shot, multiple fields
    elif len(shots) == 1 and len(fields) > 1:
        shot = shots[0]
        loaded = load(shot, fields)
        if loaded is None:
            print("No data loaded.")
            return
        time_vals, data_vals = loaded  # data_vals => shape (N, #fields)

        for idx, fld in enumerate(fields):
            col = data_vals[:, idx]
            if norm_opt:
                col = normalize(col)

            fname, funit = name(fld)
            lbl = f"{fname}[{funit}]"
            if semilogy_opt:
                plt.semilogy(time_vals, col, label=lbl)
            else:
                plt.plot(time_vals, col, label=lbl)

        norm_status = "Normalized" if norm_opt else "Raw"
        plt.title(f"shot={shot} ({norm_status})")
        if xlims is not None and len(xlims) == 2:
            plt.xlim(xlims)
        plt.xlabel("time (s)")
        plt.legend()
        plt.grid(True)
        plt.show()

    else:
        print("Error: Unsupported shot-field configuration for plotting.")


def date_from_shot(shot: int) -> tuple:
    """
    Returns (date_str, datetime_obj) for the given shot number from the shot table.

    :param shot: Shot number.
    :return: (date_str in 'YYYY-MM-DD', datetime_obj).
    """
    global DB_POOL
    if DB_POOL is None:
        print("Error: DB_POOL not initialized. run init_pool() first.")
        return None, None

    conn = DB_POOL.get_connection()
    cursor = conn.cursor()
    com = f"SELECT recordDateTime FROM shot WHERE shotNumber = {shot}"
    cursor.execute(com)
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    if result is None:
        return None, None

    datetime_obj = result[0]
    date_str = datetime_obj.strftime("%Y-%m-%d")
    return date_str, datetime_obj


def shots_from_date(date_str: str) -> list:
    """
    Returns a list of shotNumbers for the given date (YYYY-MM-DD).

    :param date_str: e.g. '2023-06-01'
    :return: list of shot numbers
    """
    global DB_POOL
    if DB_POOL is None:
        print("Error: DB_POOL not initialized. run init_pool() first.")
        return []

    conn = DB_POOL.get_connection()
    cursor = conn.cursor()
    com = (
        "SELECT DISTINCT shotNumber FROM shot "
        f"WHERE DATE(recordDateTime) = '{date_str}'"
    )
    cursor.execute(com)
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    if results:
        return [int(x[0]) for x in results]
    return []


def last_shot() -> int:
    """
    Returns the maximum shotCode from the shot table.

    :return: The last shot code as integer, or None if not found.
    """
    global DB_POOL
    if DB_POOL is None:
        print("Error: DB_POOL not initialized. run init_pool() first.")
        return None

    conn = DB_POOL.get_connection()
    cursor = conn.cursor()
    com = "SELECT MAX(shotCode) FROM shot"
    cursor.execute(com)
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    if result and result[0] is not None:
        return int(result[0])
    return None


# Additional placeholder:
# def trigger(shot, field):
#   pass

"""
Generate Sample 
"""
def get_all_field_codes_for_shot(shot: int, max_retries: int = 3):
    """
    Returns all field codes used in the given shot.
    Shot range:
      - 29349 < shot <= 42190 -> shotDataWaveform_2
      - shot > 42190         -> shotDataWaveform_3
      - Other ranges         -> None
    """
    global DB_POOL
    if DB_POOL is None:
        print("Error: DB_POOL is not initialized. Run init_pool() first.")
        return None

    attempts = 0
    while attempts < max_retries:
        conn = None
        try:
            conn = DB_POOL.get_connection()
            cursor = conn.cursor()

            if 29349 < shot <= 42190:
                # Retrieve field codes from shotDataWaveform_2
                query = (
                    "SELECT DISTINCT shotDataFieldCode "
                    "FROM shotDataWaveform_2 "
                    f"WHERE shotCode = {shot}"
                )
            elif shot > 42190:
                # Retrieve field codes from shotDataWaveform_3
                query = (
                    "SELECT DISTINCT shotDataFieldCode "
                    "FROM shotDataWaveform_3 "
                    f"WHERE shotCode = {shot}"
                )
            else:
                print("Shot number out of range for these tables.")
                return None

            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()
            conn.close()

            # rows -> [(field1,), (field2,), ...]
            field_codes = [r[0] for r in rows]
            return field_codes

        except mysql.connector.Error as e:
            print(f"Error connecting to MySQL (try {attempts+1}): {e}")
            attempts += 1
            time.sleep(1)
        finally:
            if conn and conn.is_connected():
                conn.close()

    print("Error: Could not retrieve field codes after max_retries.")
    return None


def store_shot_as_json(
    shot: int,
    output_path: str,
    max_retries: int = 3,
    daq_type: int = 0
):
    """
    - First, retrieve all field codes for the specified shot (get_all_field_codes_for_shot).
    - Load the entire shot data in the form of (time, data2D) using the load function.
    - Split by field and save as a JSON file.

    :param shot: Shot number
    :param output_path: Path to save the JSON file
    :param max_retries: Number of attempts for load function and field code retrieval
    :param daq_type: daq_type option for the load function
    :return: True (success), False (failure)
    """
    # 1) Retrieve all field codes
    field_codes = get_all_field_codes_for_shot(shot, max_retries=max_retries)
    if not field_codes:
        print(f"[store_shot_as_json] No valid field codes found for shot {shot}.")
        return False

    # 2) Retrieve the entire shot data using load
    #    If field_codes are provided to fields, (time, data2D) is returned
    load_result = load(shot, fields=field_codes, max_retries=max_retries, daq_type=daq_type)
    if load_result is None:
        print(f"[store_shot_as_json] Failed to load data for shot {shot}.")
        return False

    time_array, data_2d = load_result
    if data_2d.ndim < 2:
        print("[store_shot_as_json] Data dimension unexpected.")
        return False

    # 3) Construct a dictionary to save the entire shot data
    #    Example structure:
    #    {
    #      "shot": 12345,
    #      "data": {
    #         1: {"time": [...], "data": [...]},
    #         2: {"time": [...], "data": [...]},
    #         ...
    #      }
    #    }
    shot_data_dict = {
        "shot": shot,
        "data": {}
    }

    # Separate time and data for all fields
    # time_array.shape = (N,)
    # data_2d.shape = (N, len(field_codes))
    for i, fcode in enumerate(field_codes):
        shot_data_dict["data"][str(fcode)] = {
            "time": time_array.tolist(),
            "data": data_2d[:, i].tolist()
        }

    # 4) Save as a JSON file
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(shot_data_dict, f, ensure_ascii=False, indent=2)
        print(f"[store_shot_as_json] Shot {shot} saved to {output_path}")
        return True
    except Exception as e:
        print(f"[store_shot_as_json] Failed to save JSON: {e}")
        return False


# # main
# if __name__ == "__main__":
#     shot = 39915
#     # Path to save the JSON file
#     output_file = f"shot_{shot}.json"
#     # Save
#     store_shot_as_json(shot, output_file)
