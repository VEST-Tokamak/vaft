import unittest

import numpy as np

from helpers import get_path


class ConstraintUncertaintyTests(unittest.TestCase):
    def test_default_constraint_uncertainties_follow_donor_defaults(self):
        from vaft.machine_mapping import (
            DEFAULT_CONSTRAINT_UNCERTAINTIES,
            apply_default_constraint_uncertainties,
            vfit_magnetics_for_shot,
            vfit_pf_active_for_shot,
            vfit_tf_dynamic,
            vfit_tf_static,
        )

        payload = {}
        vfit_pf_active_for_shot(payload, shot=41672, tstart=0.24, tend=0.34, dt=4e-5)
        vfit_tf_static(payload)
        vfit_tf_dynamic(payload, shot=41672, tstart=0.24, tend=0.34, dt=4e-5)
        vfit_magnetics_for_shot(payload, shot=41672, tstart=0.24, tend=0.34, dt=4e-5)
        apply_default_constraint_uncertainties(payload)

        pf_data = np.asarray(get_path(payload, "pf_active.coil.0.current.data"), dtype=float)
        pf_error = np.asarray(get_path(payload, "pf_active.coil.0.current.data_error_upper"), dtype=float)
        np.testing.assert_allclose(
            pf_error,
            np.abs(DEFAULT_CONSTRAINT_UNCERTAINTIES["pf_active_current"] * pf_data),
        )
        self.assertEqual(
            len(get_path(payload, "pf_active.coil.0.current.time")),
            len(get_path(payload, "pf_active.coil.0.current.data")),
        )

        tf_data = np.asarray(get_path(payload, "tf.b_field_tor_vacuum_r.data"), dtype=float)
        tf_error = np.asarray(get_path(payload, "tf.b_field_tor_vacuum_r.data_error_upper"), dtype=float)
        np.testing.assert_allclose(
            tf_error,
            np.abs(DEFAULT_CONSTRAINT_UNCERTAINTIES["tf_b_field_tor_vacuum_r"] * tf_data),
        )
        self.assertEqual(
            len(get_path(payload, "tf.b_field_tor_vacuum_r.time")),
            len(get_path(payload, "tf.b_field_tor_vacuum_r.data")),
        )

        ip_data = np.asarray(get_path(payload, "magnetics.ip.0.data"), dtype=float)
        ip_error = np.asarray(get_path(payload, "magnetics.ip.0.data_error_upper"), dtype=float)
        np.testing.assert_allclose(
            ip_error,
            np.abs(DEFAULT_CONSTRAINT_UNCERTAINTIES["magnetics_ip"] * ip_data),
        )

        dia_data = np.asarray(get_path(payload, "magnetics.diamagnetic_flux.0.data"), dtype=float)
        dia_error = np.asarray(get_path(payload, "magnetics.diamagnetic_flux.0.data_error_upper"), dtype=float)
        np.testing.assert_allclose(
            dia_error,
            np.abs(DEFAULT_CONSTRAINT_UNCERTAINTIES["magnetics_diamagnetic_flux"] * dia_data),
        )
        self.assertEqual(
            len(get_path(payload, "magnetics.diamagnetic_flux.0.time")),
            len(get_path(payload, "magnetics.diamagnetic_flux.0.data")),
        )

    def test_magnetics_sensor_groups_receive_group_specific_uncertainties(self):
        from vaft.machine_mapping import (
            DEFAULT_CONSTRAINT_UNCERTAINTIES,
            apply_default_constraint_uncertainties,
            vfit_magnetics_for_shot,
        )

        payload = {}
        vfit_magnetics_for_shot(payload, shot=41672, tstart=0.24, tend=0.34, dt=4e-5)
        apply_default_constraint_uncertainties(payload)

        for probe_index, probe in enumerate(get_path(payload, "magnetics.b_field_pol_probe")):
            radial = float(probe["position"]["r"])
            vertical = float(probe["position"]["z"])
            if radial < 0.09:
                relative = DEFAULT_CONSTRAINT_UNCERTAINTIES["magnetics_bpol_inboard"]
            elif abs(vertical) > 0.8:
                relative = DEFAULT_CONSTRAINT_UNCERTAINTIES["magnetics_bpol_side"]
            else:
                relative = DEFAULT_CONSTRAINT_UNCERTAINTIES["magnetics_bpol_outboard"]

            field_data = np.asarray(probe["field"]["data"], dtype=float)
            field_error = np.asarray(probe["field"]["data_error_upper"], dtype=float)
            np.testing.assert_allclose(field_error, np.abs(relative * field_data))
            self.assertEqual(len(probe["field"]["time"]), len(field_data))

        for flux_index, flux_loop in enumerate(get_path(payload, "magnetics.flux_loop")):
            radial = float(flux_loop["position"][0]["r"])
            if radial < 0.15:
                relative = DEFAULT_CONSTRAINT_UNCERTAINTIES["magnetics_flux_loop_inboard"]
            else:
                relative = DEFAULT_CONSTRAINT_UNCERTAINTIES["magnetics_flux_loop_outboard"]

            flux_data = np.asarray(flux_loop["flux"]["data"], dtype=float)
            flux_error = np.asarray(flux_loop["flux"]["data_error_upper"], dtype=float)
            np.testing.assert_allclose(flux_error, np.abs(relative * flux_data))
            self.assertEqual(len(flux_loop["flux"]["time"]), len(flux_data))

    def test_builders_remain_transient_free_until_uncertainty_helper_runs(self):
        from helpers import path_exists
        from vaft.machine_mapping import (
            vfit_magnetics_for_shot,
            vfit_pf_active_for_shot,
            vfit_tf_dynamic,
            vfit_tf_static,
        )

        payload = {}
        vfit_pf_active_for_shot(payload, shot=41672, tstart=0.24, tend=0.34, dt=4e-5)
        vfit_tf_static(payload)
        vfit_tf_dynamic(payload, shot=41672, tstart=0.24, tend=0.34, dt=4e-5)
        vfit_magnetics_for_shot(payload, shot=41672, tstart=0.24, tend=0.34, dt=4e-5)

        self.assertFalse(path_exists(payload, "pf_active.coil.0.current.data_error_upper"))
        self.assertFalse(path_exists(payload, "tf.b_field_tor_vacuum_r.data_error_upper"))
        self.assertFalse(path_exists(payload, "magnetics.ip.0.data_error_upper"))


if __name__ == "__main__":
    unittest.main()
