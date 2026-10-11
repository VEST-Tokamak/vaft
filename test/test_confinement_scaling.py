import unittest

import numpy as np

from vaft.formula.equilibrium import (
    confinement_factor_ITER89P,
    confinement_scaling_basis,
    confinement_time_from_engineering_parameters,
    goldston_l_mode_confinement_time_from_I_P_R_a_kappa,
    neo_alcator_confinement_time_from_n_a_R_q,
    ohmic_l_mode_confinement_time_from_tau_ohmic_tau_aux,
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
        # 0.141 is the engineering-unit prefactor stored in _SCALING_COEFS
        # (SI-published Kaye 2006 coefficient converted to Ip[MA]/n[1e19]/P[MW];
        # see the conversion_note in vaft/formula/constants.py).
        expected = 0.141 * ip_ma**1.01 * self.B_t**0.70 * n_19**0.07 * p_mw**-0.37
        self.assertAlmostEqual(got, expected, places=12)

    def test_st_multi_machine_matches_manual_expression_line_density(self):
        # Kurskiev2022 is declared with density_definition == "line_avg".
        got = confinement_time_from_engineering_parameters(
            I_p=self.I_p,
            B_t=self.B_t,
            P_loss=self.P_loss,
            n_e=self.n_e,
            M=self.M,
            R=self.R,
            epsilon=self.epsilon,
            kappa=self.kappa,
            scaling="Kurskiev2022",
        )

        ip_ma = self.I_p * 1e-6
        p_mw = self.P_loss * 1e-6
        n_19 = self.n_e * 1e-19
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
        # Kurskiev2022 expects line-averaged density; passing volume-averaged
        # density without an explicit conversion factor must fail loudly.
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
                input_density_definition="volume_avg",
            )

    def test_density_conversion_volume_to_line_matches_direct_line_input(self):
        # line_to_volume_factor = n_vol / n_line, so a volume-averaged input of
        # 0.88 * n_e with factor 0.88 must equal a direct line-averaged n_e.
        converted = confinement_time_from_engineering_parameters(
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
            line_to_volume_factor=0.88,
        )
        direct = confinement_time_from_engineering_parameters(
            I_p=self.I_p,
            B_t=self.B_t,
            P_loss=self.P_loss,
            n_e=self.n_e,
            M=self.M,
            R=self.R,
            epsilon=self.epsilon,
            kappa=self.kappa,
            scaling="Kurskiev2022",
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


    def test_iter97l_regression_matches_manual_expression(self):
        got = confinement_time_from_engineering_parameters(
            I_p=self.I_p, B_t=self.B_t, P_loss=self.P_loss, n_e=self.n_e, M=self.M,
            R=self.R, epsilon=self.epsilon, kappa=self.kappa, scaling="ITER97L",
        )
        # Kaye et al., NF 37 (1997) 1303: I_p MA, n 1e19 m^-3, P MW; (R/a)^0.06 = eps^-0.06.
        expected = (0.023 * 1.2**0.96 * self.B_t**0.03 * self.R**1.83 * (1 / self.epsilon)**0.06
                    * self.kappa**0.64 * 4.2**0.40 * self.M**0.20 * 4.0**-0.73)
        self.assertAlmostEqual(got, expected, places=12)

    def test_neo_alcator_in_the_papers_cgs_units(self):
        # n = 1e13 cm^-3, a = 30 cm, R = 40 cm, q = 3 (Goldston 1984 eq. 3).
        expected = 7.1e-22 * 1e13 * 30.0**1.04 * 40.0**2.04 * 3.0**0.5
        got = neo_alcator_confinement_time_from_n_a_R_q(1e19, 0.30, 0.40, 3.0)
        self.assertIsInstance(got, float)
        self.assertAlmostEqual(got, expected, places=14)
        # Linear in density, the defining property of the LOC regime.
        arr = neo_alcator_confinement_time_from_n_a_R_q(np.array([1e19, 2e19]), 0.30, 0.40, 3.0)
        np.testing.assert_allclose(arr, [expected, 2 * expected], rtol=1e-14)

    def test_goldston_l_mode_in_the_papers_units(self):
        # I_p = 1e5 A, P = 1e5 W, R = 40 cm, a = 30 cm, kappa = 1.8 (Goldston 1984 eq. 6).
        expected = 6.4e-8 * 1e5 * 1e5**-0.5 * 40.0**1.75 * 30.0**-0.37 * 1.8**0.5
        got = goldston_l_mode_confinement_time_from_I_P_R_a_kappa(1e5, 1e5, 0.40, 0.30, 1.8)
        self.assertAlmostEqual(got, expected, places=14)
        # tau ~ P^-1/2: quadrupling the power halves the confinement time.
        self.assertAlmostEqual(
            goldston_l_mode_confinement_time_from_I_P_R_a_kappa(1e5, 4e5, 0.40, 0.30, 1.8), expected / 2,
            places=14)

    def test_ohmic_l_mode_quadrature(self):
        self.assertAlmostEqual(ohmic_l_mode_confinement_time_from_tau_ohmic_tau_aux(0.002, 0.002),
                               0.002 / np.sqrt(2.0), places=15)
        # The smaller time dominates.
        got = ohmic_l_mode_confinement_time_from_tau_ohmic_tau_aux(np.array([1e-3]), np.array([1.0]))
        np.testing.assert_allclose(got, [1e-3], rtol=1e-6)

    def test_ohmic_and_l_mode_scalings_refuse_non_positive_inputs(self):
        with self.assertRaises(ValueError):
            neo_alcator_confinement_time_from_n_a_R_q(1e19, 0.30, 0.40, 0.0)
        with self.assertRaises(ValueError):
            goldston_l_mode_confinement_time_from_I_P_R_a_kappa(1e5, -1.0, 0.40, 0.30, 1.8)
        with self.assertRaises(ValueError):
            ohmic_l_mode_confinement_time_from_tau_ohmic_tau_aux(np.nan, 0.002)

    def test_every_scaling_declares_its_energy_and_power_basis(self):
        # Issue #1713's assignments.
        expected = {"ITER89P": "global", "ITER97L": "thermal", "H98y2": "thermal",
                    "NSTX2006H": "thermal", "NSTX2006L": "global"}
        for name, basis in expected.items():
            self.assertEqual(confinement_scaling_basis(name).energy_basis, basis)
        from vaft.formula.constants import _SCALING_COEFS
        for name in [*_SCALING_COEFS, "NeoAlcator", "Goldston84L", "Goldston84OhmicL"]:
            b = confinement_scaling_basis(name)
            self.assertIn(b.energy_basis, ("thermal", "global", "unaudited"))
            self.assertIn(b.power_basis, ("p_loss", "p_abs", "p_heat", "none", "unaudited"))
            self.assertTrue(b.energy_source and b.power_source)
        self.assertEqual(confinement_scaling_basis("Kurskiev2022").power_basis, "p_abs")
        # Unchecked power definitions are declared unaudited, never guessed.
        self.assertEqual(confinement_scaling_basis("ITER89P").power_basis, "unaudited")
        with self.assertRaises(KeyError):
            confinement_scaling_basis("not_a_scaling")


class ITPA20ScalingTests(unittest.TestCase):
    """ITPA20 and ITPA20-IL, Verdoolaege et al., Nucl. Fusion 61 (2021) 076006."""

    # The paper's ITER baseline inductive Q = 10 point (Sec. 4, before table 6).
    ITER = dict(I_p=15e6, B_t=5.3, n_e=10.3e19, P_loss=87e6, R=6.2, delta=0.48,
                kappa=1.7, epsilon=0.32, M=2.5)

    def _tau(self, scaling, **changes):
        return confinement_time_from_engineering_parameters(scaling=scaling, **{**self.ITER, **changes})

    def test_iter_predictions_reproduce_the_paper(self):
        # Eq. (7) ITPA20: 3.07 s at P_l,th = 87 MW, 2.79 s at 100 MW; eq. (5)
        # ITPA20-IL: 2.90 s and 2.65 s. Both quoted with two decimals from the
        # unrounded fits (tables 16 and 10); the printed, rounded coefficients
        # stored here give 3.082, 2.807, 2.891 and 2.643 s (within 0.7 %).
        for scaling, p_loss, paper in (("ITPA20", 87e6, 3.07), ("ITPA20", 100e6, 2.79),
                                       ("ITPA20_IL", 87e6, 2.90), ("ITPA20_IL", 100e6, 2.65)):
            self.assertAlmostEqual(self._tau(scaling, P_loss=p_loss), paper, delta=0.007 * paper)
        # Same point, IPB98(y,2): the paper's 3.62 s.
        self.assertAlmostEqual(self._tau("H98y2"), 3.62, delta=0.005)

    def test_values_match_the_hand_written_power_laws(self):
        ip, bt, n, p, r, d, k, e, m = 1.2, 0.55, 4.2, 4.0, 0.85, 0.35, 1.8, 0.62, 2.0
        kw = dict(I_p=ip * 1e6, B_t=bt, n_e=n * 1e19, P_loss=p * 1e6, R=r, delta=d, kappa=k, epsilon=e, M=m)
        itpa20 = (0.053 * ip**0.98 * bt**0.22 * n**0.24 * p**-0.669 * r**1.71 * (1 + d)**0.36
                  * k**0.80 * e**0.35 * m**0.20)
        itpa20_il = (0.067 * ip**1.29 * bt**-0.13 * n**0.147 * p**-0.644 * r**1.19 * (1 + d)**0.56
                     * k**0.67 * m**0.30)
        np.testing.assert_allclose(
            confinement_time_from_engineering_parameters(scaling="ITPA20", **kw), itpa20, rtol=1e-12)
        np.testing.assert_allclose(
            confinement_time_from_engineering_parameters(scaling="ITPA20_IL", **kw), itpa20_il, rtol=1e-12)

    def test_exponent_signs_and_the_missing_epsilon_of_itpa20_il(self):
        # Doubling one input scales tau by 2**alpha: the sign of every exponent.
        for scaling, signs in (("ITPA20", {"I_p": 1, "B_t": 1, "n_e": 1, "P_loss": -1, "R": 1,
                                           "kappa": 1, "epsilon": 1, "M": 1}),
                               ("ITPA20_IL", {"I_p": 1, "B_t": -1, "n_e": 1, "P_loss": -1, "R": 1,
                                              "kappa": 1, "epsilon": 0, "M": 1})):
            base = self._tau(scaling)
            for name, sign in signs.items():
                ratio = self._tau(scaling, **{name: 2 * self.ITER[name]}) / base
                self.assertEqual(int(np.sign(np.log(ratio).round(12))), sign, (scaling, name))
        # Triangularity enters as (1 + delta): delta = 0 is allowed, a negative
        # delta lowers tau, 1 + delta <= 0 is refused.
        self.assertAlmostEqual(self._tau("ITPA20", delta=0.0) / self._tau("ITPA20"), 1.48**-0.36, places=12)
        self.assertLess(self._tau("ITPA20_IL", delta=-0.3), self._tau("ITPA20_IL", delta=0.0))
        with self.assertRaises(ValueError):
            self._tau("ITPA20", delta=-1.0)

    def test_a_triangularity_scaling_requires_delta_and_others_ignore_it(self):
        kw = {key: value for key, value in self.ITER.items() if key != "delta"}
        for scaling in ("ITPA20", "ITPA20_IL"):
            with self.assertRaisesRegex(ValueError, "delta"):
                confinement_time_from_engineering_parameters(scaling=scaling, **kw)
        self.assertEqual(confinement_time_from_engineering_parameters(scaling="H98y2", **kw),
                         self._tau("H98y2", delta=-5.0))

    def test_basis_is_thermal_on_the_thermal_loss_power(self):
        for scaling in ("ITPA20", "ITPA20_IL"):
            basis = confinement_scaling_basis(scaling)
            self.assertEqual((basis.energy_basis, basis.power_basis), ("thermal", "p_loss"))
            self.assertIn("radiation", basis.power_source)


if __name__ == "__main__":
    unittest.main()
