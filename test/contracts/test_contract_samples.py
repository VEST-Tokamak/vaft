import gzip
import json
import unittest
from pathlib import Path

from helpers import format_failures, validate_contract
from spec import CANONICAL_IDS_SPECS, SAMPLE_FILE_IDS


ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "vaft" / "data" / "samples" / "39915" / "omas.json.gz"
CONTRACT_DATA = ROOT / "test" / "data" / "contracts"


def load_json(name: str):
    with open(CONTRACT_DATA / name, "r", encoding="utf-8") as handle:
        return json.load(handle)


class SampleContractTests(unittest.TestCase):
    def assertNoContractFailures(self, failures):
        self.assertEqual(failures, {}, format_failures(failures))

    def test_current_pipeline_sample_smoke_on_39915(self):
        with gzip.open(SAMPLE, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        failures = validate_contract(
            payload,
            CANONICAL_IDS_SPECS,
            ids_names=SAMPLE_FILE_IDS["39915"],
            strict_values=False,
        )
        self.assertNoContractFailures(failures)

    def test_thomson_scattering_sample_smoke(self):
        payload = load_json("thomson_scattering.json")
        failures = validate_contract(
            payload,
            CANONICAL_IDS_SPECS,
            ids_names=SAMPLE_FILE_IDS["thomson_scattering.json"],
            strict_values=False,
        )
        self.assertNoContractFailures(failures)


if __name__ == "__main__":
    unittest.main()
