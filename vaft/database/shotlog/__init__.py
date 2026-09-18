"""The VEST operation ShotLog: monthly workbooks -> per-shot records (#995).

The ShotLog is what the operators planned and noted for every discharge --
trigger windows, coil and power-supply settings, gas, remarks -- kept as one
Excel workbook per month. This package reads it:

``batch``      choose one workbook per month and convert every sheet
``converter``  one worksheet -> a session document (run groups, shots, cells)
``card``       the fixed per-shot card of the 2023+ template
``records``    sessions -> one record per shot, with effective trigger timing
``archive``    workbooks and records in FileDB under ``legacy/shotlog``

``vaft.machine_mapping.pulse_schedule`` maps a record into the IDS, and
``python -m vaft.cli shotlog`` runs the archive and extraction.

Ported from the standalone ``VEST_ShotLog`` repository, whose web explorer
was not carried over.
"""

from .archive import archive_workbooks, load_record, record_path, write_extraction
from .batch import convert_directory, discover_sources
from .converter import convert_sheet
from .records import DAQ_OFFSET_MS, build_shot_records, trigger_table
from .schema import packaged_registry

__all__ = [
    "DAQ_OFFSET_MS",
    "archive_workbooks",
    "build_shot_records",
    "convert_directory",
    "convert_sheet",
    "discover_sources",
    "load_record",
    "packaged_registry",
    "record_path",
    "trigger_table",
    "write_extraction",
]
