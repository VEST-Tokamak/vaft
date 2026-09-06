import unittest
from unittest.mock import patch

class PlasmaModelTests(unittest.TestCase):
    @patch("vaft.machine_mapping.magnetics._safe_vest_load", return_value=None)
    def test_vfit_plasma_current_rejects_missing_raw_data(self, _load):
        from vaft.database.raw import RawSignalUnavailableError
        from vaft.machine_mapping import vfit_plasma_current

        with self.assertRaisesRegex(RawSignalUnavailableError, "shot 41672, field 109"):
            vfit_plasma_current(41672)


if __name__ == "__main__":
    unittest.main()
