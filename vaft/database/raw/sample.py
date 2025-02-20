from vaft.database.raw import *
import mysql.connector
import numpy as np
import json
import time
import re

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


# main
if __name__ == "__main__":
    shot = 39915
    # Path to save the JSON file
    output_file = f"shot_{shot}.json"
    # Save
    store_shot_as_json(shot, output_file)
