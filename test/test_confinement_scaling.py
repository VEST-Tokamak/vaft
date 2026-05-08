import unittest

import numpy as np

from vaft.formula.equilibrium import (
    confinement_factor_ITER89P,
    confinement_time_from_engineering_parameters,
)


class ConfinementScalingTests(unittest.TestCase):
    def setUp(self):
        # SI inputs expected by the public API
        self.I_p = 1.2e6      # [A]
        self.B_t = 0.55       # [T]
        self.P_loss = 4.0e6   # [W]
        self.n_e = 4.2e19     # [m^-3]
        self.M = 2.0          # [amu]
        self.R = 0.85         # [m]
        self.epsilon = 0.62   # [-]
        self.kappa = 1.95     # [-]
        self.tau_E_exp = 0.024
        self.tau_E_ITER89P = 0.018

    def test_h98y2_regression_matches_manual_expression(self):
        got = confinement_time_from_engineering_parameters(
            I_p=self.I_p,
            B_t=self.B_t,
            P_loss=self.P_loss,
            n_e=self.n_e,
            M=self.M,
            R=self.R,
            epsilon=self.epsilon,
            kappa=self.kappa,
            scaling="H98y2",
        )

        ip_ma = self.I_p * 1e-6
        p_mw = self.P_loss * 1e-6
        n_19 = self.n_e * 1e-19
        expected = (
            0.0562
            * ip_ma**0.93
            * self.R**1.97
            * self.epsilon**0.58
            * self.kappa**0.78
            * n_19**0.41
            * self.B_t**0.15
            * self.M**0.19
            * p_mw**-0.69
        )
        self.assertAlmostEqual(got, expected, places=12)

    def test_nstx2006h_no_longer_depends_on_R_epsilon_kappa_M(self):
        base = confinement_time_from_engineering_parameters(
            I_p=self.I_p,
            B_t=self.B_t,
            P_loss=self.P_loss,
            n_e=self.n_e,
            M=self.M,
            R=self.R,
            epsilon=self.epsilon,
            kappa=self.kappa,
            scaling="NSTX2006H",
        )

        varied = confinement_time_from_engineering_parameters(
            I_p=self.I_p,
            B_t=self.B_t,
            P_loss=self.P_loss,
            n_e=self.n_e,
            M=3.0,
            R=1.15,
            epsilon=0.35,
            kappa=1.3,
            scaling="NSTX2006H",
        )

        self.assertAlmostEqual(base, varied, places=12)

    def test_nstx2006l_regression_matches_manual_expression(self):
        got = confinement_time_from_engineering_parameters(
            I_p=self.I_p,
            B_t=self.B_t,
            P_loss=self.P_loss,
            n_e=self.n_e,
            M=self.M,
            R=self.R,
            epsilon=self.epsilon,
            kappa=self.kappa,
            scaling="NSTX2006L",
        )

        ip_ma = self.I_p * 1e-6
        p_mw = self.P_loss * 1e-6
        n_19 = self.n_e * 1e-19
        expected = 4.73e-4 * ip_ma**1.01 * self.B_t**0.70 * n_19**0.07 * p_mw**-0.37
        self.assertAlmostEqual(got, expected, places=12)

    def test_st_multi_machine_matches_manual_expression_volume_density(self):
        n_vol = 0.88 * self.n_e
        got = confinement_time_from_engineering_parameters(
            I_p=self.I_p,
            B_t=self.B_t,
            P_loss=self.P_loss,
            n_e=n_vol,
            M=self.M,
            R=self.R,
            epsilon=self.epsilon,
            kappa=self.kappa,
            scaling="Kurskiev2022",
            input_density_definition="volume_avg",
        )

        ip_ma = self.I_p * 1e-6
        p_mw = self.P_loss * 1e-6
        n_19 = n_vol * 1e-19
        expected = (
            0.066
            * ip_ma**0.53
            * self.B_t**1.05
            * p_mw**-0.58
            * n_19**0.65
            * self.R**2.66
            * self.kappa**0.78
        )
        self.assertAlmostEqual(got, expected, places=12)

    def test_density_conversion_requires_explicit_factor(self):
        with self.assertRaises(ValueError):
            confinement_time_from_engineering_parameters(
                I_p=self.I_p,
                B_t=self.B_t,
                P_loss=self.P_loss,
                n_e=self.n_e,
                M=self.M,
                R=self.R,
                epsilon=self.epsilon,
                kappa=self.kappa,
                scaling="Kurskiev2022",
                input_density_definition="line_avg",
            )

    def test_density_conversion_line_to_volume_matches_direct_volume_input(self):
        converted = confinement_time_from_engineering_parameters(
            I_p=self.I_p,
            B_t=self.B_t,
            P_loss=self.P_loss,
            n_e=self.n_e,
            M=self.M,
            R=self.R,
            epsilon=self.epsilon,
            kappa=self.kappa,
            scaling="Kurskiev2022",
            input_density_definition="line_avg",
            line_to_volume_factor=0.88,
        )
        direct = confinement_time_from_engineering_parameters(
            I_p=self.I_p,
            B_t=self.B_t,
            P_loss=self.P_loss,
            n_e=0.88 * self.n_e,
            M=self.M,
            R=self.R,
            epsilon=self.epsilon,
            kappa=self.kappa,
            scaling="Kurskiev2022",
            input_density_definition="volume_avg",
        )
        self.assertAlmostEqual(converted, direct, places=12)

    def test_invalid_density_definition_raises_clear_error(self):
        with self.assertRaises(ValueError):
            confinement_time_from_engineering_parameters(
                I_p=self.I_p,
                B_t=self.B_t,
                P_loss=self.P_loss,
                n_e=self.n_e,
                M=self.M,
                R=self.R,
                epsilon=self.epsilon,
                kappa=self.kappa,
                scaling="ITER89P",
                input_density_definition="foobar",
            )

    def test_backward_compatible_default_call_still_works(self):
        got = confinement_time_from_engineering_parameters(
            I_p=self.I_p,
            B_t=self.B_t,
            P_loss=self.P_loss,
            n_e=self.n_e,
            M=self.M,
            R=self.R,
            epsilon=self.epsilon,
            kappa=self.kappa,
            scaling="ITER89P",
        )
        self.assertTrue(got > 0.0)

    def test_confinement_factor_iter89p_scalar(self):
        got = confinement_factor_ITER89P(
            tau_E_exp=self.tau_E_exp,
            tau_E_ITER89P=self.tau_E_ITER89P,
        )
        self.assertAlmostEqual(got, self.tau_E_exp / self.tau_E_ITER89P, places=12)

    def test_confinement_factor_iter89p_vectorized(self):
        tau_exp = np.array([0.020, 0.024, 0.030], dtype=float)
        tau_iter89p = np.array([0.010, 0.018, 0.020], dtype=float)
        got = confinement_factor_ITER89P(tau_E_exp=tau_exp, tau_E_ITER89P=tau_iter89p)
        expected = tau_exp / tau_iter89p
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
