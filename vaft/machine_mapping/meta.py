"""
Metadata handling module for VEST database.

This module provides functions for handling metadata and dataset descriptions
in the OMAS/IMAS data structure format.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from uncertainties import unumpy
from omas import *
from omfit_classes.omfit_eqdsk import OMFITgeqdsk
import yaml
import datetime
from omas import ODS

# # load data from file and store in ods

# # naming convention: machine_mapping.<diagnostic_name>_from_file(ods, shotnumber)


def dataset_description(ods: ODS, source: str, options: dict = None) -> None:
    """
    Populate the `dataset_description` IDS in a way that is consistent with the
    IMAS/OMAS schema for `dataset_description` [1]_.

    Parameters
    ----------
    ods:
        OMAS data structure to populate.
    source:
        Shot number or other identifier. Interpreted as a shot number when
        ``options['source_type'] == 'shot'``.
    options:
        Optional dictionary. Recognised keys include

        - ``source_type``: ``'shot'`` (default) or ``'file'``
        - ``description``: free-text description of this ODS
        - ``run``: IMAS run number (default 0)
        - ``pulse_type``: e.g. ``'pulse'`` (default) or ``'simulation'``
        - ``user``: username (defaults to ``os.environ['USER']`` when missing)
        - ``machine``: machine name (default ``'VEST'``)
        - ``provider``: person in charge of producing this data
        - ``creation_date``: ISO-8601 string; defaults to ``now()``
        - ``name``: user-defined name for this IDS occurrence
        - ``homogeneous_time``: 0, 1 or 2 (default 2 for static metadata)
        - ``dd_version``: physics data dictionary version string
        - ``imas_version``: IMAS infrastructure version string

    Notes
    -----
    The following fields are populated to match the schema in
    `dataset description — OMAS <https://gafusion.github.io/omas/schema/schema_dataset%20description.html>`_:

    - ``dataset_description.data_entry.machine``
    - ``dataset_description.data_entry.pulse``
    - ``dataset_description.data_entry.pulse_type``
    - ``dataset_description.data_entry.run``
    - ``dataset_description.data_entry.user``
    - ``dataset_description.ids_properties.comment``
    - ``dataset_description.ids_properties.creation_date``
    - ``dataset_description.ids_properties.name``
    - ``dataset_description.ids_properties.homogeneous_time``
    - ``dataset_description.ids_properties.provider``
    - ``dataset_description.dd_version``
    - ``dataset_description.imas_version``
    """
    from .dataset_description import dataset_description as _dataset_description

    _dataset_description(ods, source, options)

def summary(ods: ODS, source: str, options: dict = None) -> None:
    """
    Add experiment summary information to the ODS structure.
    
    Args:
        ods: OMAS data structure
        source: Source for summary data (file path or shot number)
        options: Optional dictionary containing processing options
            - source_type: Optional str ('file' or 'shot')
            - start_time: Optional float for experiment start time
            - end_time: Optional float for experiment end time
            - plasma_current: Optional float for maximum plasma current
            - toroidal_field: Optional float for toroidal field
            - additional_info: Optional dict for additional summary information
    """
    from .summary import summary as _summary

    _summary(ods, source, options)

def get_metadata(source: str, options: dict = None) -> dict:
    """
    Get metadata information from the source.
    
    Args:
        source: Source for metadata (file path or shot number)
        options: Optional dictionary containing processing options
            - source_type: Optional str ('file' or 'shot')
            - metadata_type: Optional str to specify metadata type
    
    Returns:
        dict: Metadata information
    """
    if options is None:
        options = {}
    
    source_type = options.get('source_type', 'shot')
    metadata_type = options.get('metadata_type', 'all')
    
    if source_type == 'shot':
        # Load metadata from shot database
        metadata = {
            'shot': int(source),
            'timestamp': datetime.datetime.now().isoformat(),
            'source_type': 'shot'
        }
    else:
        # Load metadata from file
        metadata = {
            'file': source,
            'timestamp': datetime.datetime.now().isoformat(),
            'source_type': 'file'
        }
    
    return metadata