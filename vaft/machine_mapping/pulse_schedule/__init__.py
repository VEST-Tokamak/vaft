"""``pulse_schedule`` for VEST, from the operation ShotLog (#995).

The IDS names the package, as every mapping in ``vaft.machine_mapping`` is
named for the IDS it fills. Its source is the ShotLog -- the operators' monthly
Excel workbooks of what was planned and noted for each discharge -- and the
modules here read it:

``batch``      discover every workbook, choose one record per month
``converter``  one worksheet -> a session document (run groups, shots, cells)
``card``       the fixed per-shot card of the 2023+ template
``records``    sessions -> one record per shot, with effective trigger timing
``archive``    workbooks and records in FileDB under ``legacy/shotlog``
``mapping``    a record -> ``pulse_schedule.event(:)``

``python -m vaft.cli shotlog`` runs the archive and extraction, and the
corrective pipeline's ``ingest_external_diagnostics.py --diagnostic shotlog``
builds the products. Ported from the standalone ``VEST_ShotLog`` repository,
whose web explorer was not carried over.
"""

from .archive import archive_workbooks, load_record, record_path, write_extraction
from .batch import convert_directory, discover_sources
from .converter import convert_sheet
from .mapping import (
    EVENT_TYPES,
    LISTENERS,
    PulseScheduleUnavailableError,
    map_pulse_schedule,
    pulse_schedule,
)
from .records import DAQ_OFFSET_MS, build_shot_records, trigger_table
from .schema import packaged_registry

__all__ = [
    "DAQ_OFFSET_MS",
    "EVENT_TYPES",
    "LISTENERS",
    "PulseScheduleUnavailableError",
    "archive_workbooks",
    "build_shot_records",
    "convert_directory",
    "convert_sheet",
    "discover_sources",
    "load_record",
    "map_pulse_schedule",
    "packaged_registry",
    "pulse_schedule",
    "record_path",
    "trigger_table",
    "write_extraction",
]
