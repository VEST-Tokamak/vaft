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
    Add dataset description to the ODS structure.
    
    Args:
        ods: OMAS data structure
        source: Source for dataset description (file path or shot number)
        options: Optional dictionary containing processing options
            - version: Optional int for dataset version (default: 1)
            - source_type: Optional str ('file' or 'shot')
            - creation_date: Optional str for custom creation date
            - description: Optional str for custom description
    """
    if options is None:
        options = {}
    
    # Set basic dataset information
    ods['dataset_description.data_entry.pulse'] = int(source) if options.get('source_type') == 'shot' else None
    ods['dataset_description.version'] = options.get('version', 1)
    ods['dataset_description.creation_date'] = options.get('creation_date', datetime.datetime.now().isoformat())
    ods['dataset_description.source'] = 'VEST database'
    
    # Add custom description if provided
    if 'description' in options:
        ods['dataset_description.description'] = options['description']
    
    # Set data entry information
    ods['dataset_description.data_entry.shot'] = int(source) if options.get('source_type') == 'shot' else None
    ods['dataset_description.data_entry.run'] = options.get('run', 1)
    
    # Set machine information
    ods['dataset_description.machine'] = 'VEST'
    ods['dataset_description.machine.mode'] = 'tokamak'
    
    # Set data quality information
    ods['dataset_description.data_quality.level'] = options.get('quality_level', 'processed')
    ods['dataset_description.data_quality.comment'] = options.get('quality_comment', 'Processed data from VEST database')

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
    if options is None:
        options = {}
    
    # Set basic summary information
    ods['summary.shot'] = int(source) if options.get('source_type') == 'shot' else None
    
    # Set time information
    if 'start_time' in options:
        ods['summary.time.start'] = options['start_time']
    if 'end_time' in options:
        ods['summary.time.end'] = options['end_time']
    
    # Set plasma parameters
    if 'plasma_current' in options:
        ods['summary.plasma_current.maximum'] = options['plasma_current']
    if 'toroidal_field' in options:
        ods['summary.toroidal_field.maximum'] = options['toroidal_field']
    
    # Add additional information if provided
    if 'additional_info' in options:
        for key, value in options['additional_info'].items():
            ods[f'summary.{key}'] = value
    
    # Set experiment status
    ods['summary.status'] = options.get('status', 'completed')
    ods['summary.comment'] = options.get('comment', 'VEST experiment data')

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