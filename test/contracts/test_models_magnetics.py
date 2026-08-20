import unittest
from unittest.mock import patch

from helpers import get_path


class MagneticsModelTests(unittest.TestCase):
    def test_vfit_magnetics_static_populates_yaml_geometry(self):
        from vaft.machine_mapping import vfit_magnetics_static

        payload = {}
        vfit_magnetics_static(payload)

        self.assertEqual(get_path(payload, "magnetics.ids_properties.homogeneous_time"), 1)
        self.assertEqual(len(get_path(payload, "magnetics.flux_loop")), 11)
        self.assertEqual(len(get_path(payload, "magnetics.b_field_pol_probe")), 68)
        self.assertEqual(get_path(payload, "magnetics.flux_loop.0.position.0.r"), 0.592)
        self.assertEqual(get_path(payload, "magnetics.flux_loop.0.position.0.z"), 0.685)
        self.assertEqual(get_path(payload, "magnetics.b_field_pol_probe.0.position.r"), 0.089)

    @patch("vaft.machine_mapping.magnetics._safe_vest_load", return_value=None)
    def test_vfit_magnetics_for_shot_rejects_missing_raw_data(self, _load):
        from vaft.database.raw import RawSignalUnavailableError
        from vaft.machine_mapping import vfit_magnetics_for_shot

        payload = {}
        with self.assertRaisesRegex(RawSignalUnavailableError, "shot 41672"):
            vfit_magnetics_for_shot(payload, shot=41672, tstart=0.24, tend=0.34, dt=4e-5)


if __name__ == "__main__":
    unittest.main()
