#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MySQL-based VEST Database Access and Plotting

This module provides convenient functions to connect to VEST's MySQL Raw Daq Signal database
via a connection pool, load data by shot/field, correct time arrays for DAQ
triggers, retrieve date or shot lists, and plot results.
"""

import os
import re
import time
from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np
import mysql.connector
import matplotlib.pyplot as plt
import json
import gzip
from mysql.connector.pooling import MySQLConnectionPool
import logging
from datetime import datetime

import os
import yaml
from cryptography.fernet import Fernet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
KEY_FILE = os.path.expanduser("~/.vest/encryption_key.key")
CONFIG_FILE = os.path.expanduser("~/.vest/database_raw_info.yaml")

# DAQ related constants
FAST_DT = 4e-6   # Fast DAQ sampling interval (seconds)
SLOW_DT = 4e-5   # Slow DAQ sampling interval (seconds)
SLOW_DT_THRESHOLD = 5e-6  # Threshold for slow/fast DAQ classification

# Database connection related constants
MAX_RETRIES = 3
POOL_SIZE = 4

# Field codes to exclude
EXCLUDED_FIELD_CODES = {110, 111, 112, 113}  # Processed Triple Probe Signals

def load_or_generate_key() -> bytes:
    """
    Load or generate an encryption key.

    Returns:
        bytes: The encryption key
    """
    key_dir = os.path.dirname(KEY_FILE)
    os.makedirs(key_dir, exist_ok=True)

    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "rb") as key_file:
            return key_file.read()
    else:
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as key_file:
            key_file.write(key)
        return key

class SecureConfigManager:
    def __init__(self):
        self.key = load_or_generate_key()
        self.cipher = Fernet(self.key)

    def encrypt(self, plain_text: str) -> str:
        return self.cipher.encrypt(plain_text.encode()).decode()

    def decrypt(self, encrypted_text: str) -> str:
        return self.cipher.decrypt(encrypted_text.encode()).decode()

    def get_info(self) -> None:
        """Prompt user for database configuration and save to YAML."""
        try:
            hostname = input("Enter the database hostname: ")
            username = input("Enter the database username: ")
            password = input("Enter the database password: ")
            database = "VEST"

            encrypted_password = self.encrypt(password)
            config_data = {
                "hostname": hostname,
                "username": username,
                "password": encrypted_password,
                "database": database,
            }

            with open(CONFIG_FILE, "w") as file:
                yaml.dump(config_data, file)
                logger.info(f"Configuration saved to {CONFIG_FILE}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise

    def load_config(self) -> Tuple[str, str, str, str]:
        """
        Load database configuration from YAML file.

        Returns:
            Tuple[str, str, str, str]: (hostname, username, password, database)
        """
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, "r") as file:
                    config_data = yaml.safe_load(file)

                    return (
                        config_data["hostname"],
                        config_data["username"],
                        self.decrypt(config_data["password"]),
                        config_data["database"]
                    )
            else:
                logger.info(f"No configuration file found at {CONFIG_FILE}. Initializing setup...")
                self.get_info()
                return self.load_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

# Global Database Pool
DB_POOL: Optional[MySQLConnectionPool] = None

def setup_raw_db() -> None:
    """Initialize database configuration."""
    return SecureConfigManager().get_info()

def configuration() -> Tuple[str, str, str, str]:
    """Load database configuration."""
    scm = SecureConfigManager()
    return scm.load_config()

def init_pool() -> None:
    """Initialize the global MySQLConnectionPool."""
    global DB_POOL
    try:
        HOSTNAME, USERNAME, PASSWORD, DATABASE = configuration()
        DB_POOL = MySQLConnectionPool(
            pool_name="mypool",
            pool_size=POOL_SIZE,
            host=HOSTNAME,
            database=DATABASE,
            user=USERNAME,
            password=PASSWORD
        )
        logger.info("Database connection pool initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
        raise

def _load_from_shot_waveform_2(
    db_conn: mysql.connector.MySQLConnection,
    shot: int,
    field: int
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load shot data from shotDataWaveform_2 table.

    Args:
        db_conn: Active MySQL database connection
        shot: Shot number
        field: Field code

    Returns:
        Tuple of (time_array, data_array) as np.ndarrays
    """
    try:
        cursor = db_conn.cursor()
        query = (
            "SELECT shotDataWaveformTime, shotDataWaveformValue "
            "FROM shotDataWaveform_2 "
            f"WHERE shotCode = {shot} AND shotDataFieldCode = {field} "
            "ORDER BY shotDataWaveformTime ASC"
        )
        cursor.execute(query)
        result = np.array(cursor.fetchall())
        cursor.close()

        if result.size > 0:
            return result.T[0], result.T[1]
        return np.array([0.0]), np.array([0.0])
    except Exception as e:
        logger.error(f"Error loading from shotDataWaveform_2: {e}")
        raise

def _load_from_shot_waveform_3(
    db_conn: mysql.connector.MySQLConnection,
    shot: int,
    field: int
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load shot data from shotDataWaveform_3 table.

    Args:
        db_conn: Active MySQL database connection
        shot: Shot number
        field: Field code

    Returns:
        Tuple of (time_array, data_array) as np.ndarrays
    """
    try:
        cursor = db_conn.cursor()
        query = (
            "SELECT shotDataWaveformTime, shotDataWaveformValue "
            "FROM shotDataWaveform_3 "
            f"WHERE shotCode = {shot} AND shotDataFieldCode = {field}"
        )
        cursor.execute(query)
        myresult = cursor.fetchall()
        cursor.close()

        if len(myresult) != 1:
            logger.warning(
                f"shot={shot}, field={field} has multiple/no rows. "
                "Returning ([0.],[0.])."
            )
            return np.array([0.0]), np.array([0.0])

        shot_time_str = re.sub(r"[\[\]]", "", myresult[0][0])
        shot_val_str = re.sub(r"[\[\]]", "", myresult[0][1])

        time_vals = np.array([float(x) for x in shot_time_str.split(",")])
        data_vals = np.array([float(x) for x in shot_val_str.split(",")])

        return time_vals, data_vals
    except Exception as e:
        logger.error(f"Error loading from shotDataWaveform_3: {e}")
        raise

def _load_from_sample_file(
    shot: int,
    fields: List[int],
    sample_opt: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load data from sample JSON file.

    Args:
        shot: Shot number
        fields: List of field codes to load
        sample_opt: JSON file path (.gz extension included)

    Returns:
        Tuple of (time_array, data_array) as np.ndarrays or None if loading fails
    """
    try:
        json_path = sample_opt if isinstance(sample_opt, str) else f"SHOT_{shot}.json"

        if not os.path.isfile(json_path):
            logger.error(f"Sample JSON file not found: {json_path}")
            return None

        with gzip.open(json_path, "rt", encoding="utf-8") as f:
            shot_json = json.load(f)

        file_shot = shot_json.get("shot")
        if file_shot is not None and file_shot != shot:
            logger.warning(f"JSON shot={file_shot}, requested shot={shot}.")

        data_dict = shot_json.get("fields", {})
        if not data_dict:
            logger.error(f"No 'fields' found in JSON for shot={shot}")
            return None

        if not fields:
            fields = list(map(int, data_dict.keys()))

        time_arrays = []
        data_arrays = []

        for fld in fields:
            fld_str = str(fld)
            entry = data_dict.get(fld_str)
            if entry is None:
                logger.warning(f"Field {fld} not found in JSON. Skipping...")
                continue

            raw_data = entry.get("data", [])
            if not raw_data:
                logger.warning(f"No data array for field {fld}. Skipping...")
                continue

            dt = SLOW_DT if entry.get("type") == "slow" else FAST_DT
            n = len(raw_data)
            tvals = np.arange(n, dtype=float) * dt
            dvals = np.array(raw_data, dtype=float)

            if dt == FAST_DT:
                tvals = tvals + _daq_trigger_time_correction(shot)

            time_arrays.append(tvals)
            data_arrays.append(dvals)

        if not time_arrays:
            logger.error(f"No valid fields loaded from JSON for shot={shot}.")
            return None
        
        time_ref = time_arrays[0]
        data_stack = np.column_stack([
            arr[:len(time_ref)] for arr in data_arrays
        ])
        time_ref = time_ref[:min(len(arr) for arr in data_arrays)]

        return (time_ref, data_stack.ravel()) if len(fields) == 1 else (time_ref, data_stack)

    except Exception as e:
        logger.error(f"Error loading from sample file: {e}")
        return None

def _daq_trigger_time_correction(shot: int) -> float:
    """
    Correct time array for DAQ trigger delay.

    Args:
        shot: Shot number

    Returns:
        float: Time shift value
    """
    if shot < 41446:
        return 0.24
    elif 41446 <= shot <= 41451:
        return 0.26
    elif 41452 <= shot <= 41659:
        return 0.24
    else:  # shot >= 41660
        return 0.26

def load_raw(
    shot: int,
    fields: Optional[Union[int, List[int]]] = None,
    max_retries: int = MAX_RETRIES,
    daq_type: Optional[int] = None,
    sample_opt: Union[bool, str] = False
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    High-level data loader for the VEST database.

    Args:
        shot: Shot number
        fields: Field code(s) to load
        max_retries: Maximum number of connection retries
        daq_type: DAQ type
        sample_opt: Sample file path or False for DB loading

    Returns:
        Tuple of (time_array, data_array) as np.ndarrays or None if loading fails
    """
    try:
        # Normalize fields parameter
        if fields is None:
            fields = []
        elif isinstance(fields, int):
            fields = [fields]

        # Load from sample file if specified
        if isinstance(sample_opt, str):
            result = _load_from_sample_file(shot, fields, sample_opt)
            return result if result is not None else None

        # Initialize DB pool if needed
        global DB_POOL
        if DB_POOL is None:
            logger.info("DB_POOL not initialized. Initializing automatically...")
            init_pool()

        # Load from database
        if not fields:
            logger.error("No fields specified for DB loading.")
            return None

        attempts = 0
        while attempts < max_retries:
            conn = None
            try:
                conn = DB_POOL.get_connection()
                time_arrays, data_arrays = [], []

                for fld in fields:
                    if 29349 < shot <= 42190:
                        tvals, dvals = _load_from_shot_waveform_2(conn, shot, fld)
                    elif shot > 42190:
                        tvals, dvals = _load_from_shot_waveform_3(conn, shot, fld)
                    else:
                        logger.error("Shot number out of range for these tables.")
                        return None

                    time_arrays.append(tvals)
                    data_arrays.append(dvals)

                if not time_arrays:
                    logger.error(f"No data found for shot={shot} from DB.")
                    return None

                # Stack multiple fields
                time_ref = time_arrays[0]
                min_len = min(len(arr) for arr in data_arrays)
                data_stack = np.column_stack([
                    arr[:min_len] for arr in data_arrays
                ])
                time_ref = time_ref[:min_len]

                # Apply DAQ trigger correction
                if time_ref[-1] < 0.101:
                    time_ref = time_ref + _daq_trigger_time_correction(shot)

                return (time_ref, data_stack.ravel()) if len(fields) == 1 else (time_ref, data_stack)

            except mysql.connector.Error as err:
                logger.error(f"Error connecting to MySQL (try {attempts+1}): {err}")
                attempts += 1
                time.sleep(1)
            finally:
                if conn and conn.is_connected():
                    conn.close()

            logger.error("Could not retrieve data after max_retries.")
        return None

    except Exception as e:
        logger.error(f"Error in load_raw: {e}")
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
        loaded = load_raw(shot, field)
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
            loaded = load_raw(sh, field)
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
        loaded = load_raw(shot, fields)
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

            # rows -> [(field1,), (field2,), ...]
            field_codes = [r[0] for r in rows]
            # remove 110, 111, 112, 113 (Processed Triple Probe Signal which has different time array)
            field_codes = [f for f in field_codes if f not in EXCLUDED_FIELD_CODES]

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

def dump_all_raw_signals_for_shot(
    shot: int,
    output_path: str = None,
    max_retries: int = 3,
    daq_type: int = 0,
    slow_dt_threshold: float = 5e-6,  # Time interval threshold for slow DAQ [4e-5 sec/sample] vs Fast DAQ [4e-6 sec/sample] classification
    plot_opt: bool = False
    ) -> bool:
    """
    Store shot data as JSON GZIP file (.json.gz) with the following steps:
    1. Retrieve list of field codes
    2. Load (time, data1D) using load_raw
    3. Classify as fast/slow based on sampling interval (time[1]-time[0])
    4. Create shot_data = { "shot": shot, "fields": {fcode: {type, data}} }
    5. Save as gzip compressed JSON (.json.gz)
    6. If plot_opt is True, display and save signals as subplots
    """
    # Set default output path
    if output_path is None:
        output_path = os.path.join(os.getcwd(), f"vest_raw_{shot}.json.gz")
    elif not output_path.endswith(".gz"):
        output_path += ".gz"

    # 1) Retrieve field codes
    field_codes = get_all_field_codes_for_shot(shot, max_retries=max_retries)
    if not field_codes:
        print(f"[store_shot_as_json] No valid field codes for shot {shot}")
        return False

    # 2) Load data and determine DAQ type
    shot_data = {
        "shot": shot,
        "fields": {}
    }

    # Prepare subplot if plot_opt is True
    if plot_opt == 1:
        n_fields = len(field_codes)
        n_cols = int(np.ceil(np.sqrt(n_fields)))
        n_rows = int(np.ceil(n_fields / n_cols))
        plt.figure(figsize=(20, 20))
        plot_idx = 1

    for fcode in field_codes:
        try:
            time, data = load_raw(
                shot, fcode,
                max_retries=max_retries,
                daq_type=daq_type
            )
        except Exception as e:
            print(f"[store_shot_as_json] load_raw failed for field {fcode}: {e}")
            continue

        if len(time) < 2:
            print(f"[store_shot_as_json] insufficient time points for field {fcode}")
            continue

        # Classify as fast/slow based on sampling interval
        is_slow = (time[1] - time[0]) >= slow_dt_threshold
        daq_label = "slow" if is_slow else "fast"

        shot_data["fields"][str(fcode)] = {
            "type": daq_label,
            "data": data.tolist()
        }

        # Display signal as subplot if plot_opt is True
        if plot_opt == 1:
            plt.subplot(n_rows, n_cols, plot_idx)
            plt.plot(time, data)
            field_name, field_remark = name(fcode)
            title = f"Field {fcode}"
            if field_name:
                title += f"\n{field_name}"
            if field_remark:
                title += f"\n{field_remark}"
            plt.title(title, fontsize=8)
            plt.grid(True)
            plot_idx += 1

    if not shot_data["fields"]:
        print(f"[store_shot_as_json] No data loaded for shot {shot}")
        return False

    # Save all subplots if plot_opt is True
    if plot_opt == 1:
        plt.tight_layout()
        plot_path = output_path.replace('.json.gz', '_signals.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[store_shot_as_json] Signal plots saved to {plot_path}")

    # 3) Save as gzip compressed JSON
    try:
        with gzip.open(output_path, "wt", encoding="utf-8") as gz:
            json.dump(shot_data, gz, ensure_ascii=False, indent=2)
        print(f"[store_shot_as_json] Shot {shot} saved to {output_path}")
        return True
    except Exception as e:
        print(f"[store_shot_as_json] Failed to write JSON: {e}")
        return False

# -----------------------------------------------------------------------------
# TEST FUNCTIONS (for development and debugging)
# ----------------
def compare_db_and_dumped_raw_signals_for_shot(
    shot: int,
    output_path: str = None,
    max_retries: int = 3,
    daq_type: int = 0,
    slow_dt_threshold: float = 5e-6
    ) -> bool:
    """
    Compare and plot original signals from database with signals loaded from JSON file
    
    Parameters:
    -----------
    shot : int
        Shot number to compare
    output_path : str, optional
        JSON file path. If None, automatically generated
    max_retries : int, default=3
        Number of DB connection retries
    daq_type : int, default=0
        DAQ type
    slow_dt_threshold : float, default=5e-6
        Time interval threshold for slow/fast DAQ classification
        
    Returns:
    --------
    bool
        Success status
    """
    # 1) Retrieve field codes from DB
    field_codes = get_all_field_codes_for_shot(shot, max_retries=max_retries)
    if not field_codes:
        print(f"[compare_signals] No valid field codes for shot {shot}")
        return False

    # 2) Set JSON file path
    if output_path is None:
        output_path = os.path.join(os.getcwd(), f"vest_raw_{shot}.json.gz")
    elif not output_path.endswith(".gz"):
        output_path += ".gz"

    # 3) Set up subplot layout
    n_fields = len(field_codes)
    n_cols = int(np.ceil(np.sqrt(n_fields)))
    n_rows = int(np.ceil(n_fields / n_cols))
    plt.figure(figsize=(20, 20))
    plot_idx = 1

    # 4) Compare DB and JSON data for each field
    for fcode in field_codes:
        try:
            # Load data from DB
            db_time, db_data = load_raw(shot, fcode, max_retries=max_retries, daq_type=daq_type)
            
            # Load data from JSON
            json_time, json_data = load_raw(shot, fcode, sample_opt=output_path)
            
            if db_time is None or json_time is None:
                print(f"[compare_signals] Failed to load data for field {fcode}")
                continue

            # Create subplot
            plt.subplot(n_rows, n_cols, plot_idx)
            
            # Plot DB data
            plt.plot(db_time, db_data, 'b-', label='DB Data', alpha=0.7)
            
            # Plot JSON data
            plt.plot(json_time, json_data, 'r--', label='JSON Data', alpha=0.7)
            
            # Set title
            field_name, field_remark = name(fcode)
            title = f"Field {fcode}"
            if field_name:
                title += f"\n{field_name}"
            if field_remark:
                title += f"\n{field_remark}"
            
            # Calculate data difference
            if len(db_data) == len(json_data):
                max_diff = np.max(np.abs(db_data - json_data))
                title += f"\nMax Diff: {max_diff:.2e}"
            else:
                title += f"\nLength Mismatch: DB={len(db_data)}, JSON={len(json_data)}"
            
            plt.title(title, fontsize=8)
            plt.grid(True)
            plt.legend(fontsize=6)
            plot_idx += 1

        except Exception as e:
            print(f"[compare_signals] Error processing field {fcode}: {e}")
            continue

    # 5) Save all plots
    plt.tight_layout()
    plot_path = output_path.replace('.json.gz', '_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[compare_signals] Comparison plots saved to {plot_path}")
    
    return True


# MAIN FUNCTION - SIMPLE TEST ROUTINE
if __name__ == "__main__":
    print("=" * 60)
    print("VEST DATABASE RAW.PY - FUNCTION TEST")
    print("=" * 60)
    
    # Test 1: Configuration functions
    print("\n1. Testing configuration functions...")
    try:
        key = load_or_generate_key()
        print("✓ load_or_generate_key: OK")
    except Exception as e:
        print(f"✗ load_or_generate_key: FAILED - {e}")
    
    try:
        scm = SecureConfigManager()
        print("✓ SecureConfigManager: OK")
    except Exception as e:
        print(f"✗ SecureConfigManager: FAILED - {e}")
    
    # Test 2: Database connection
    print("\n2. Testing database connection...")
    try:
        init_pool()
        print("✓ init_pool: OK")
        db_ok = True
    except Exception as e:
        print(f"✗ init_pool: FAILED - {e}")
        db_ok = False
    
    if db_ok:
        # Test 3: Basic database functions
        print("\n3. Testing basic database functions...")
        
        try:
            last_shot_num = last_shot()
            print(f"✓ last_shot: OK (last shot: {last_shot_num})")
        except Exception as e:
            print(f"✗ last_shot: FAILED - {e}")
            last_shot_num = None
        
        if last_shot_num:
            try:
                date_str, date_obj = date_from_shot(last_shot_num)
                print(f"✓ date_from_shot: OK (date: {date_str})")
            except Exception as e:
                print(f"✗ date_from_shot: FAILED - {e}")
            
            try:
                field_codes = get_all_field_codes_for_shot(last_shot_num)
                print(f"✓ get_all_field_codes_for_shot: OK (found {len(field_codes)} fields)")
            except Exception as e:
                print(f"✗ get_all_field_codes_for_shot: FAILED - {e}")
                field_codes = []
            
            if field_codes:
                try:
                    field_name, field_remark = name(field_codes[0])
                    print(f"✓ name: OK (field {field_codes[0]}: {field_name})")
                except Exception as e:
                    print(f"✗ name: FAILED - {e}")
                
                try:
                    time_vals, data_vals = load_raw(last_shot_num, field_codes[0])
                    print(f"✓ load_raw (single field): OK (loaded {len(data_vals)} points)")
                except Exception as e:
                    print(f"✗ load_raw (single field): FAILED - {e}")
                
                if len(field_codes) >= 2:
                    try:
                        time_vals, data_vals = load_raw(last_shot_num, field_codes[:2])
                        print(f"✓ load_raw (multiple fields): OK (loaded {data_vals.shape} data)")
                    except Exception as e:
                        print(f"✗ load_raw (multiple fields): FAILED - {e}")
        
        # Test 4: Date functions
        print("\n4. Testing date functions...")
        try:
            shots = shots_from_date("2023-06-01")
            print(f"✓ shots_from_date: OK (found {len(shots)} shots)")
        except Exception as e:
            print(f"✗ shots_from_date: FAILED - {e}")
        
        # Test 5: DAQ time correction
        print("\n5. Testing DAQ time correction...")
        test_shots = [40000, 41500, 42000, 45000]
        for shot in test_shots:
            try:
                correction = _daq_trigger_time_correction(shot)
                print(f"✓ _daq_trigger_time_correction (shot {shot}): OK ({correction}s)")
            except Exception as e:
                print(f"✗ _daq_trigger_time_correction (shot {shot}): FAILED - {e}")
        
        # Test 6: Data dumping functions
        print("\n6. Testing data dumping functions...")
        if last_shot_num:
            try:
                result = dump_all_raw_signals_for_shot(last_shot_num, plot_opt=False)
                print(f"✓ dump_all_raw_signals_for_shot: OK")
            except Exception as e:
                print(f"✗ dump_all_raw_signals_for_shot: FAILED - {e}")
            
            try:
                result = compare_db_and_dumped_raw_signals_for_shot(last_shot_num)
                print(f"✓ compare_db_and_dumped_raw_signals_for_shot: OK")
            except Exception as e:
                print(f"✗ compare_db_and_dumped_raw_signals_for_shot: FAILED - {e}")
        
        # Test 7: Plotting functions
        print("\n7. Testing plotting functions...")
        if last_shot_num and field_codes:
            try:
                plot(last_shot_num, field_codes[0])
                print("✓ plot (single shot, single field): OK")
            except Exception as e:
                print(f"✗ plot (single shot, single field): FAILED - {e}")
            
            if len(field_codes) >= 2:
                try:
                    plot(last_shot_num, field_codes[:2])
                    print("✓ plot (single shot, multiple fields): OK")
                except Exception as e:
                    print(f"✗ plot (single shot, multiple fields): FAILED - {e}")
    
    # Test 8: Error handling
    print("\n8. Testing error handling...")
    try:
        load_raw(-1, 1)
        print("✗ load_raw (invalid shot): Should have failed")
    except:
        print("✓ load_raw (invalid shot): Correctly handled error")
    
    try:
        date_from_shot(-1)
        print("✗ date_from_shot (invalid shot): Should have failed")
    except:
        print("✓ date_from_shot (invalid shot): Correctly handled error")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)
