"""
MySQL-based VEST Database Access and Plotting

This module provides convenient functions to connect to VEST's MySQL Raw Daq Signal database
via a connection pool, load data by shot/field, correct time arrays for DAQ
triggers, retrieve date or shot lists, and plot results.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import yaml
try:  # Rendering lives in vaft.plot; this is only an availability probe.
    import matplotlib as _matplotlib
except ImportError:
    _matplotlib = None
try:
    from cryptography.fernet import Fernet
except ImportError:
    Fernet = None

try:
    import mysql.connector as mysql_connector
    from mysql.connector.pooling import MySQLConnectionPool
except ImportError:
    mysql_connector = None
    MySQLConnectionPool = Any

    class MysqlError(Exception):
        """Fallback MySQL error type when mysql.connector is unavailable."""

else:
    MysqlError = mysql_connector.Error

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
    _require_fernet()
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
        _require_fernet()
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

SQL_TABLE_PATH = Path(__file__).resolve().parents[1] / "data" / "legacy" / "sql_table.txt"
RawSource = str | os.PathLike[str]


class RawSignalUnavailableError(LookupError):
    """Raised when a required VEST raw waveform is absent or unusable."""

    def __init__(
        self,
        shot: int,
        field: int | str,
        reason: str,
        *,
        signal_name: str | None = None,
    ) -> None:
        self.shot = int(shot)
        self.field = field
        self.reason = str(reason)
        self.signal_name = signal_name
        label = f" ({signal_name})" if signal_name else ""
        super().__init__(
            f"Required VEST raw signal is unavailable for shot {self.shot}, "
            f"field {self.field}{label}: {self.reason}. Verify that the raw "
            "archive contains this field or that the VEST SQL service returned data."
        )


def require_signal(
    loaded: Optional[Tuple[np.ndarray, np.ndarray]],
    *,
    shot: int,
    field: int | str,
    signal_name: str | None = None,
    min_samples: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and return a required single-channel raw waveform.

    Raw database readers retain their legacy ``None`` return for compatibility.
    Scientific mappers call this helper at the point where a missing waveform
    would otherwise be indistinguishable from a real zero-valued measurement.
    """
    if loaded is None:
        raise RawSignalUnavailableError(
            shot,
            field,
            "the data source returned no waveform",
            signal_name=signal_name,
        )

    time_values, data_values = loaded
    time_array = np.asarray(time_values, dtype=float).reshape(-1)
    data_array = np.asarray(data_values, dtype=float)
    if data_array.ndim > 1 and sum(size > 1 for size in data_array.shape) > 1:
        raise RawSignalUnavailableError(
            shot,
            field,
            f"expected one data channel, received shape {data_array.shape}",
            signal_name=signal_name,
        )
    data_array = data_array.reshape(-1)
    if time_array.size != data_array.size:
        raise RawSignalUnavailableError(
            shot,
            field,
            f"time/data lengths differ ({time_array.size} != {data_array.size})",
            signal_name=signal_name,
        )
    if time_array.size < int(min_samples):
        raise RawSignalUnavailableError(
            shot,
            field,
            f"received {time_array.size} sample(s); at least {int(min_samples)} are required",
            signal_name=signal_name,
        )
    return time_array, data_array


def _format_sample_path(template: str, shot: int) -> str:
    try:
        return template.format(shot=int(shot))
    except Exception:
        return template


def _resolve_sample_path(
    shot: int,
    sample_opt: bool | RawSource = False,
) -> Optional[str]:
    if isinstance(sample_opt, (str, os.PathLike)) and os.fspath(sample_opt):
        return _format_sample_path(os.fspath(sample_opt), shot)
    return None


def _archive_payload(shot: int, sample_path: str) -> dict:
    """Read and validate one archived raw dump.

    An archive that cannot be trusted is an export failure, not a missing
    signal, so every problem here raises instead of degrading to SQL.
    """
    if not os.path.isfile(sample_path):
        raise FileNotFoundError(
            f"Archived raw source not found for shot={shot}: {sample_path}"
        )
    opener = gzip.open if sample_path.endswith(".gz") else open
    with opener(sample_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Raw archive is not a JSON object: {sample_path}")
    archive_shot = payload.get("shot")
    try:
        archive_shot_number = int(archive_shot)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Raw archive has no valid shot number: {sample_path}"
        ) from error
    if archive_shot_number != int(shot):
        raise ValueError(
            f"Raw archive shot mismatch: requested {shot}, but {sample_path} "
            f"contains shot {archive_shot}"
        )
    fields = payload.get("fields")
    if not isinstance(fields, dict) or not fields:
        raise ValueError(f"Raw archive has no fields mapping: {sample_path}")
    return payload


def _archive_field_codes(shot: int, sample_path: str) -> list[int]:
    """Return the field codes an archived raw dump actually carries."""
    payload = _archive_payload(shot, sample_path)
    return sorted(int(code) for code in payload["fields"])


def _require_mysql() -> None:
    if mysql_connector is None:
        raise ImportError("mysql-connector-python is required for SQL loading")


def _require_fernet() -> None:
    if Fernet is None:
        raise ImportError("cryptography is required for encrypted DB configuration")


def _require_matplotlib() -> None:
    if _matplotlib is None:
        raise ImportError("matplotlib is required for raw plotting helpers")


def sql_loading_available() -> bool:
    """Return whether SQL-backed waveform loading is available in this env."""
    return mysql_connector is not None

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
        _require_mysql()
        HOSTNAME, USERNAME, PASSWORD, DATABASE = configuration()
        DB_POOL = MySQLConnectionPool(
            pool_name="mypool",
            pool_size=POOL_SIZE,
            host=HOSTNAME,
            database=DATABASE,
            user=USERNAME,
            password=PASSWORD,
            connection_timeout=10,
        )
        logger.info("Database connection pool initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
        raise

def _load_from_shot_waveform_2(
    db_conn: Any,
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
    db_conn: Any,
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
            logger.warning(f"JSON shot={file_shot}, requested shot={shot}. Skipping archived sample.")
            return None

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
        
        min_len = min(len(arr) for arr in data_arrays)
        time_ref = time_arrays[0][:min_len]
        data_stack = np.column_stack([
            arr[:min_len] for arr in data_arrays
        ])

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
    sample_opt: bool | RawSource = False
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    High-level data loader for the VEST database.

    Args:
        shot: Shot number
        fields: Field code(s) to load
        max_retries: Maximum number of connection retries
        daq_type: DAQ type
        sample_opt: Authoritative sample file path or False for live DB loading

    Returns:
        Tuple of (time_array, data_array) as np.ndarrays or None if loading fails
    """
    try:
        # Normalize fields parameter
        if fields is None:
            fields = []
        elif isinstance(fields, int):
            fields = [fields]

        # An explicit archived source is authoritative and never falls back to SQL.
        sample_path = _resolve_sample_path(shot, sample_opt)
        if sample_path:
            if not os.path.isfile(sample_path):
                raise FileNotFoundError(
                    f"Archived raw source not found for shot={shot}: {sample_path}"
                )
            result = _load_from_sample_file(shot, fields, sample_path)
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

            except MysqlError as err:
                logger.error(f"Error connecting to MySQL (try {attempts+1}): {err}")
                attempts += 1
                time.sleep(1)
            finally:
                if conn and conn.is_connected():
                    conn.close()

            logger.error("Could not retrieve data after max_retries.")
        return None

    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error in load_raw: {e}")
        return None


def vest_connection_pool(pool_size: int = 4) -> None:
    """Compatibility wrapper for donor-style SQL pool initialization."""
    global POOL_SIZE
    POOL_SIZE = int(pool_size)
    init_pool()


def vest_check_table(mydb: Any, shot: int) -> int:
    """Return the waveform table generation used by a shot."""
    del mydb
    if 29349 < shot <= 42190:
        return 2
    if shot > 42190:
        return 3
    return 1


def vest_load_shot_waveform_2(mydb: Any, shot: int, field: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper for donor-style waveform table 2 access."""
    return _load_from_shot_waveform_2(mydb, shot, field)


def vest_load_shot_waveform_3(mydb: Any, shot: int, field: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper for donor-style waveform table 3 access."""
    return _load_from_shot_waveform_3(mydb, shot, field)


def vest_load(
    shot: int,
    field: int,
    max_retries: int = MAX_RETRIES,
    sample_opt: bool | RawSource = False,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Compatibility wrapper for donor-style single-signal loading."""
    sample_path = _resolve_sample_path(shot, sample_opt)
    if sample_path:
        return load_raw(shot, field, max_retries=max_retries, sample_opt=sample_path)
    if not sql_loading_available():
        return None
    return load_raw(shot, field, max_retries=max_retries)


def _sql_table_mapping() -> dict[str, int]:
    with open(SQL_TABLE_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def vest_load_by_name(
    shot: int,
    name: str,
    sample_opt: bool | RawSource = False,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load a waveform by signal name using the shipped lookup table."""
    try:
        table = _sql_table_mapping()
    except (FileNotFoundError, json.JSONDecodeError) as error:
        logger.error(f"Cannot load SQL mapping file: {error}")
        return None

    field = table.get(name)
    if field is None:
        logger.error(f"Unknown signal name in sql_table.txt: {name}")
        return None

    return vest_load(shot, int(field), sample_opt=sample_opt)


vest_load_shotWaveform_2 = vest_load_shot_waveform_2
vest_load_shotWaveform_3 = vest_load_shot_waveform_3
vest_loadn = vest_load_by_name


def vest_date(shot: int) -> Optional[str]:
    """Return shot date as YYYY-MM-DD."""
    date_str, _ = date_from_shot(shot)
    return date_str


def vest_shots(date: str) -> list[int]:
    """Return shot list for a date string."""
    return shots_from_date(date)


def vestdb_is_data_exist(shot_code: int, shot_data_field_code: int) -> bool:
    """Return whether a signal exists for a shot in raw storage."""
    loaded = vest_load(shot_code, shot_data_field_code)
    if loaded is None:
        return False
    _, data = loaded
    return bool(np.asarray(data).size > 0)


def load(shot: int, field: int, max_retries: int = MAX_RETRIES):
    """Legacy alias kept for existing machine_mapping wrappers."""
    return vest_load(shot, field, max_retries=max_retries)

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
    *,
    ax=None,
    show: bool = True,
):
    """
    Plots raw waveforms for the 3 standard scenarios:

    1. Single shot, single field -> single line plot.
    2. Multiple shots, single field -> multiple lines, one per shot.
    3. Single shot, multiple fields -> multiple lines, one per field.

    This function loads and labels the data; rendering is delegated to
    :func:`vaft.plot.render_line_series`, so no Matplotlib code lives in the
    database namespace (issue #63).

    :param shots: int or list of int
    :param fields: int or list of int
    :param semilogy_opt: If True, uses a logarithmic y axis. Defaults to False.
    :param norm_opt: If True, normalizes data to (-1, 1). Defaults to False.
    :param xlims: Optional ``(low, high)`` time limits.
    :param ax: Optional axes to draw into.
    :param show: Display the figure. Defaults to True for backward compatibility.
    :return: ``(Figure, Axes)``, or ``None`` when no data could be loaded.
    """
    _require_matplotlib()
    from vaft.plot import LineSeries, Series, render_line_series

    if isinstance(shots, int):
        shots = [shots]
    if isinstance(fields, int):
        fields = [fields]

    def normalize(data):
        """Normalizes the data to the range (-1, 1)."""
        return 2 * (data - data.min()) / (data.max() - data.min()) - 1

    traces: list = []
    norm_status = "Normalized" if norm_opt else "Raw"
    y_label = ""

    # 1) Single shot, single field
    if len(shots) == 1 and len(fields) == 1:
        shot, field = shots[0], fields[0]
        loaded = load_raw(shot, field)
        if loaded is None:
            print("No data loaded.")
            return None
        time_vals, data_vals = loaded
        if norm_opt:
            data_vals = normalize(data_vals)
        fname, y_label = name(field)
        title = f"shot={shot}, field={field}, name={fname} ({norm_status})"
        traces.append(
            Series(x=time_vals, y=data_vals, label=f"shot {shot}, field {field}")
        )

    # 2) multiple shots, single field
    elif len(shots) > 1 and len(fields) == 1:
        field = fields[0]
        fname, y_label = name(field)
        title = f"field={field}, name={fname} ({norm_status})"
        for shot in shots:
            loaded = load_raw(shot, field)
            if loaded is None:
                continue
            time_vals, data_vals = loaded
            if norm_opt:
                data_vals = normalize(data_vals)
            traces.append(Series(x=time_vals, y=data_vals, label=f"shot {shot}"))

    # 3) single shot, multiple fields
    elif len(shots) == 1 and len(fields) > 1:
        shot = shots[0]
        loaded = load_raw(shot, fields)
        if loaded is None:
            print("No data loaded.")
            return None
        time_vals, data_vals = loaded  # data_vals => shape (N, #fields)
        title = f"shot={shot} ({norm_status})"
        for index, field in enumerate(fields):
            column = data_vals[:, index]
            if norm_opt:
                column = normalize(column)
            fname, funit = name(field)
            traces.append(
                Series(x=time_vals, y=column, label=f"{fname}[{funit}]")
            )

    else:
        raise ValueError(
            "plot() supports one shot with one or many fields, or many shots "
            f"with one field; got {len(shots)} shots and {len(fields)} fields"
        )

    if not traces:
        print("No data loaded.")
        return None

    model = LineSeries(
        series=tuple(traces),
        x_label="time",
        x_unit="s",
        y_label=y_label,
        title=title,
        x_limits=tuple(xlims) if xlims is not None and len(xlims) == 2 else None,
        log_y=bool(semilogy_opt),
    )
    return render_line_series(model, ax=ax, show=show, figsize=(8.0, 4.0))


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
        logger.info("DB_POOL not initialized. Initializing automatically...")
        init_pool()

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

        except MysqlError as e:
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
    plot_opt: bool = False,
    sample_opt: bool | RawSource = False
    ) -> bool:
    """
    Store shot data as JSON GZIP file (.json.gz) with the following steps:
    1. Retrieve list of field codes
    2. Load (time, data1D) using load_raw
    3. Classify as fast/slow based on sampling interval (time[1]-time[0])
    4. Create shot_data = { "shot": shot, "fields": {fcode: {type, data}} }
    5. Save as gzip compressed JSON (.json.gz)
    6. If plot_opt is True, display and save signals as subplots

    ``sample_opt`` is an authoritative archived raw source, using the same
    convention as :func:`load_raw`: the field codes come from the archive and no
    step falls back to SQL. The archive is re-derived rather than copied, so the
    output is a canonical dump regardless of how the source was written.
    """
    if plot_opt == 1:
        _require_matplotlib()

    # Set default output path
    if output_path is None:
        output_path = os.path.join(os.getcwd(), f"vest_raw_{shot}.json.gz")
    elif not output_path.endswith(".gz"):
        output_path += ".gz"

    # 1) Retrieve field codes, from the archive when one is given
    sample_path = _resolve_sample_path(shot, sample_opt)
    if sample_path is not None:
        field_codes = _archive_field_codes(shot, sample_path)
    else:
        field_codes = get_all_field_codes_for_shot(shot, max_retries=max_retries)
    if not field_codes:
        print(f"[store_shot_as_json] No valid field codes for shot {shot}")
        return False

    # 2) Load data and determine DAQ type
    shot_data = {
        "shot": shot,
        "fields": {}
    }

    # Collect one panel model per field when plotting is requested; rendering
    # happens once at the end through vaft.plot (issue #63).
    panel_models: list = []

    for fcode in field_codes:
        try:
            time, data = load_raw(
                shot, fcode,
                max_retries=max_retries,
                daq_type=daq_type,
                sample_opt=sample_opt
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

        # Collect the signal as a panel model if plot_opt is True
        if plot_opt == 1:
            panel_models.append(_field_panel(fcode, time, data))

    if not shot_data["fields"]:
        print(f"[store_shot_as_json] No data loaded for shot {shot}")
        return False

    # Render and save all panels if plot_opt is True
    if plot_opt == 1 and panel_models:
        plot_path = output_path.replace('.json.gz', '_signals.png')
        _save_field_panels(panel_models, plot_path)
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
    _require_matplotlib()

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

    # 3) Collect one comparison panel per field
    panel_models: list = []

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

            # Calculate data difference for the panel title
            if len(db_data) == len(json_data):
                max_diff = np.max(np.abs(db_data - json_data))
                suffix = f"Max Diff: {max_diff:.2e}"
            else:
                suffix = f"Length Mismatch: DB={len(db_data)}, JSON={len(json_data)}"

            panel_models.append(
                _field_panel(
                    fcode,
                    [db_time, json_time],
                    [db_data, json_data],
                    labels=("DB Data", "JSON Data"),
                    styles=({"color": "b", "alpha": 0.7},
                            {"color": "r", "linestyle": "--", "alpha": 0.7}),
                    title_suffix=suffix,
                )
            )

        except Exception as e:
            print(f"[compare_signals] Error processing field {fcode}: {e}")
            continue

    # 5) Render and save all panels
    if not panel_models:
        print(f"[compare_signals] No comparable fields for shot {shot}")
        return False
    plot_path = output_path.replace('.json.gz', '_comparison.png')
    _save_field_panels(panel_models, plot_path)
    print(f"[compare_signals] Comparison plots saved to {plot_path}")

    return True


def _field_panel(
    field_code,
    times,
    values,
    *,
    labels=("",),
    styles=({},),
    title_suffix: str = "",
):
    """Build the ``LineSeries`` panel model for one raw field."""
    from vaft.plot import LineSeries, Series

    if not isinstance(times, (list, tuple)):
        times, values = [times], [values]
    field_name, field_remark = name(field_code)
    title_parts = [f"Field {field_code}"]
    if field_name:
        title_parts.append(str(field_name))
    if field_remark:
        title_parts.append(str(field_remark))
    if title_suffix:
        title_parts.append(title_suffix)
    traces = tuple(
        Series(x=time, y=data, label=label, style=dict(style))
        for time, data, label, style in zip(times, values, labels, styles)
    )
    return LineSeries(series=traces, x_label="time", x_unit="s",
                      title="\n".join(title_parts))


def _save_field_panels(panel_models, plot_path):
    """Render the collected field panels into a square grid and save it."""
    from vaft.plot import Panels, render_panels, save_figure

    columns = int(np.ceil(np.sqrt(len(panel_models))))
    figure, _ = render_panels(
        Panels(models=tuple(panel_models), ncols=columns, share_x=False),
        figsize=(20, 20),
    )
    return save_figure(figure, plot_path)


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
