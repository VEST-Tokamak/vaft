import unittest

import numpy as np


class PlasmaModelTests(unittest.TestCase):
    def test_vfit_plasma_current_returns_offline_zero_waveform_without_sql(self):
        from vaft.machine_mapping import vfit_plasma_current

        time, ip = vfit_plasma_current(41672)
        self.assertEqual(len(time), len(ip))
        self.assertTrue(len(time) > 0)
        self.assertTrue(np.allclose(ip, 0.0))

    def test_vfit_plasma_mgods_startend_detects_signal_window(self):
        from vaft.machine_mapping import vfit_plasma_mgods_startend

        payload = {
            "magnetics": {
                "ip": [
                    {
                        "time": [0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34],
                        "data": [0.0, 0.0, 0.0, 30_000.0, 45_000.0, 44_000.0, 2_000.0, 0.0],
                    }
                ]
            }
        }

        tstart, tend = vfit_plasma_mgods_startend(payload)
        self.assertGreaterEqual(tstart, 0.20)
        self.assertLessEqual(tstart, 0.26)
        self.assertGreaterEqual(tend, 0.30)
        self.assertLessEqual(tend, 0.34)


if __name__ == "__main__":
    unittest.main()
