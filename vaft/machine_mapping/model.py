"""
Model-based and derived physical quantities calculation module for VEST database.

This module provides functions for calculating model-based and derived physical quantities
from raw diagnostic data and storing them in OMAS/IMAS data structure format.
"""

import os
import yaml
import numpy as np
from omas import ODS
import xarray as xr
import re
import struct

def em_coupling(ods: ODS, options: dict = None) -> None:
    """
    Calculate electromagnetic coupling.
    
    Args:
        ods: OMAS data structure
        options: Optional dictionary containing calculation options
            - method: Optional str to specify calculation method
            - params: Optional dict for calculation parameters
    """
    # Implementation will be added
    pass

def equilibrium(ods: ODS, source: str, options: dict = None) -> None:
    """
    Calculate equilibrium quantities.
    
    Args:
        ods: OMAS data structure
        source: Source for equilibrium data (file path or shot number)
        options: Optional dictionary containing calculation options
            - time_slice: Optional float for specific time point
            - method: Optional str to specify calculation method
            - params: Optional dict for calculation parameters
            - source_type: Optional str ('file' or 'shot')
    """
    # Implementation will be added
    pass



# # =============================================================================
# # Equilibrium from External Analysis (Element/Profiles)
# # =============================================================================

# def vfit_element_analysis_from_file(ods: Dict[str, Any], shot: int) -> None:
#     """
#     Populate ODS equilibrium from external element analysis mat file.

#     :param ods: ODS containing 'equilibrium'.
#     :param shot: Shot number.
#     """
#     print("Reading element analysis data from .mat (stub).")
#     # mat = _read_mat_from_ElementAnalysis(shot)
#     # parse mat and store in ods accordingly
#     print("vfit_equilibrium_from_element_analysis done (stub).")


# def vfit_from_profile_fitting_from_file(ods: Dict[str, Any], shot: int) -> None:
#     """
#     Populate ODS equilibrium from external profile fitting mat file.

#     :param ods: ODS containing 'equilibrium'.
#     :param shot: Shot number.
#     """
#     print("Reading profile fitting data from .mat (stub).")
#     # mat = _read_mat_from_ProfileFitting(shot)
#     # parse mat and store in ods
#     print("vfit_equilibrium_from_profile_fitting done (stub).")


def mhd_linear(ods: ODS, source: str, options: dict = None) -> None:
    """
    Calculate linear MHD stability.
    
    Args:
        ods: OMAS data structure
        source: Source for MHD data (file path or shot number)
            "example : /srv/vest.filedb/public/39915" -> read multiple mode for multiple time slice in stability analysis such as dcon, rdcon and stride
        options: Optional dictionary containing calculation options
            - method: Optional str to specify calculation method
            - params: Optional dict for calculation parameters
            - source_type: Optional str ('file' or 'shot')
    """
    def _read_fortran_record_length(f):
        """
        Reads 4 bytes from the file and interprets them as the Fortran unformatted record length.

        Parameters:
            f (file object): Open file object in binary mode.

        Returns:
            int or None: The record length if 4 bytes are read successfully, or None if EOF is reached.
        """
        raw = f.read(4)
        if len(raw) < 4:
            return None
        return struct.unpack("<i", raw)[0]  # little-endian 32-bit integer



    def _read_n_floats(f, n):
        """
        Reads n float32 values (4 bytes each) from file f and returns them as a NumPy array.

        Parameters:
            f (file object): Open file object in binary mode.
            n (int): Number of float32 values to read.
        
        Returns:
            np.ndarray: A NumPy array of float32 values (little-endian).
        
        Raises:
            EOFError: If EOF is reached before reading n*4 bytes.
        """
        raw = f.read(n * 4)
        if len(raw) < n * 4:
            raise EOFError("Unexpected EOF while reading float data.")
        return np.frombuffer(raw, dtype="<f4")  # little-endian float32

    def _read_solutions_bin(filename):
        """
        Reads 'solutions.bin' written by Fortran unformatted I/O.

        Each record contains 7 float32 values in this order:
            [psi, rho, q, Real_xi, Imag_xi, Real_b, Imag_b]

        Multiple ipert blocks are separated by null records (record length 0).

        Parameters:
            filename (str): Path to the solutions.bin file.
        
        Returns:
            np.ndarray: A 3D NumPy array (n_ipert, n_step, 7) containing the data.
        """
        data_blocks = []  # data_blocks[i] is a list of step arrays (each with 7 floats) for ipert block i

        with open(filename, "rb") as f:
            while True:
                length = _read_fortran_record_length(f)
                if length is None:
                    # EOF
                    break
                if length == 0:
                    # Null record: end of current ipert block
                    continue

                num_floats = length // 4  # Usually 7
                arr_step0 = _read_n_floats(f, num_floats)
                trailing_len = _read_fortran_record_length(f)
                if trailing_len != length:
                    print("[Warning] Leading/trailing record length mismatch.")

                steps_for_ipert = [arr_step0]

                while True:
                    length2 = _read_fortran_record_length(f)
                    if length2 is None:
                        break  # EOF
                    if length2 == 0:
                        break  # End of this ipert block
                    nfloat2 = length2 // 4
                    arr2 = _read_n_floats(f, nfloat2)
                    trailing_len2 = _read_fortran_record_length(f)
                    if trailing_len2 != length2:
                        print("[Warning] Mismatch in subsequent record lengths.")
                    steps_for_ipert.append(arr2)

                data_blocks.append(steps_for_ipert)

        n_ipert = len(data_blocks)
        if n_ipert == 0:
            return np.zeros((0, 0, 7), dtype=np.float32)

        max_steps = max(len(steps) for steps in data_blocks)
        arr3d = np.full((n_ipert, max_steps, 7), np.nan, dtype=np.float32)
        for i_ipert, step_list in enumerate(data_blocks):
            for j_step, vec7 in enumerate(step_list):
                arr3d[i_ipert, j_step, :] = vec7

        return arr3d



    for file in os.listdir(source):
        match = re.match(r'dcon_output_n(\d+)\.nc', file)
        if match:
            n = int(match.group(1))
            filepath = os.path.join(source, file)

            try:
                ds = xr.open_dataset(filepath)
                W_t = ds["W_t_eigenvalue"].isel(i=0).sel(mode=1).values.item()

                mode_index = n 
                ods['mhd_linear'][time_slice][mode_index]['n'] = n
                ods['mhd_linear'][time_slice][mode_index]['energy_perturbed'] = W_t

                print(f"[INFO] Stored W_t={W_t} for n={n}")

            except Exception as e:
                print(f"[ERROR] Failed to read {filepath}: {e}")
                continue

    bin_file = os.path.join(source, "solutions.bin")
    if not os.path.exists(bin_file):
        print(f"[ERROR] File not found: {bin_file}")
        return
    
    arr3d = _read_solutions_bin(bin_file) 
    n_ipert, n_step, _ = arr3d.shape

    while len(ods['mhd_linear']) <= time_slice:
        ods['mhd_linear'].append([])

    for n in range(n_ipert):
        mode_entry = {}
        mode_entry['plasma'] = {}
        psi_grid = arr3d[n, :, 0]
        alpha_grid = np.arange(n_step)

        mode_entry['plasma']['grid'] = {
            'dim1': psi_grid.tolist(),
            'dim2': alpha_grid.tolist()
        }

        mode_entry['plasma']['displacement_perpendicular'] = {
            'real': arr3d[n, :, 3].tolist(),
            'imaginary': arr3d[n, :, 4].tolist()
        }

        mode_entry['n'] = n 
        ods['mhd_linear'][time_slice].append(mode_entry)



def pf_passive(ods: ODS, source: str, options: dict = None) -> None:
    """
    Calculate passive PF coil effects.
    
    Args:
        ods: OMAS data structure
        source: Source for passive coil data (file path or shot number)
        options: Optional dictionary containing calculation options
            - time_slice: Optional float for specific time point
            - method: Optional str to specify calculation method
            - params: Optional dict for calculation parameters
            - source_type: Optional str ('file' or 'shot')
    """
    # Implementation will be added
    pass

def pf_plasma(ods: ODS, source: str, options: dict = None) -> None:
    """
    Calculate plasma current distribution effects.
    
    Args:
        ods: OMAS data structure
        source: Source for plasma current data (file path or shot number)
        options: Optional dictionary containing calculation options
            - time_slice: Optional float for specific time point
            - method: Optional str to specify calculation method
            - params: Optional dict for calculation parameters
            - source_type: Optional str ('file' or 'shot')
    """
    # Implementation will be added
    pass 
