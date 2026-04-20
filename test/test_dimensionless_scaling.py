import unittest

import numpy as np

from vaft.formula.equilibrium import (
    calc_beta_t,
    calc_inverse_aspect_ratio,
    calc_nu_star,
    calc_omega_i_tau_E,
    calc_q_cyl,
    calc_rho_star,
    check_kadomtsev_constraint,
    coulomb_logarithm,
    coulomb_logarithm_from_n_T,
    kadomtsev_constraint_from_engineering_exponents,
    line_to_volume_avg_density,
    nu_star_from_n_T_B_R_epsilon_kappa_I,
    omega_i_tau_E_from_B_tau_E_M,
    q_cyl_from_B_R_epsilon_kappa_I,
    rho_star_from_M_T_B_R_epsilon,
    beta_t_from_n_T_B,
)


class DimensionlessScalingTests(unittest.TestCase):
    def test_iter_reference_case_matches_expected_verdoolaege_values(self):
        i_p = 15e6
        b_t = 5.3
        n_line = 10.3e19
        n_vol = line_to_volume_avg_density(n_line)
        t_eV = 8.6e3
        r_geo = 6.2
        epsilon = 0.32
        kappa = 1.7
        m_eff = 2.5

        rho_star = rho_star_from_M_T_B_R_epsilon(
            M_eff_amu=m_eff,
            T_eV=t_eV,
            B_t_T=b_t,
            R_geo_m=r_geo,
            epsilon=epsilon,
        )
        beta_t = beta_t_from_n_T_B(n_m3=n_vol, T_eV=t_eV, B_t_T=b_t)
        nu_star = nu_star_from_n_T_B_R_epsilon_kappa_I(
            n_m3=n_vol,
            T_eV=t_eV,
            B_t_T=b_t,
            R_geo_m=r_geo,
            epsilon=epsilon,
            kappa_a=kappa,
            I_p_A=i_p,
        )
        q_cyl = q_cyl_from_B_R_epsilon_kappa_I(
            B_t_T=b_t,
            R_geo_m=r_geo,
            epsilon=epsilon,
            kappa_a=kappa,
            I_p_A=i_p,
        )

        self.assertAlmostEqual(rho_star, 0.0020, delta=2.0e-4)
        self.assertAlmostEqual(beta_t, 2.24, delta=0.1)
        self.assertAlmostEqual(nu_star, 0.014, delta=0.002)
        self.assertAlmostEqual(q_cyl, 1.94, delta=0.05)

    def test_coulomb_logarithm_matches_manual_expression(self):
        n = 9.06e19
        t = 8.6e3
        expected = 30.9 - np.log(np.sqrt(n) / t)
        got = coulomb_logarithm_from_n_T(n, t)
        self.assertAlmostEqual(got, float(expected), places=12)

    def test_line_to_volume_avg_density_default_factor(self):
        n_line = 10.3e19
        got = line_to_volume_avg_density(n_line)
        self.assertAlmostEqual(got, 0.88 * n_line, places=6)

    def test_calc_inverse_aspect_ratio(self):
        a = 1.984
        r = 6.2
        got = calc_inverse_aspect_ratio(a, r)
        self.assertAlmostEqual(got, a / r, places=12)

    def test_invalid_inputs_raise_value_error(self):
        with self.assertRaises(ValueError):
            coulomb_logarithm_from_n_T(1.0e19, 0.0)
        with self.assertRaises(ValueError):
            line_to_volume_avg_density(0.0)
        with self.assertRaises(ValueError):
            calc_inverse_aspect_ratio(-1.0, 6.2)
        with self.assertRaises(ValueError):
            rho_star_from_M_T_B_R_epsilon(2.5, 8.6e3, 5.3, 6.2, 0.0)
        with self.assertRaises(ValueError):
            beta_t_from_n_T_B(9.06e19, 8.6e3, -5.3)
        with self.assertRaises(ValueError):
            q_cyl_from_B_R_epsilon_kappa_I(5.3, 6.2, 0.32, 1.7, 0.0)
        with self.assertRaises(ValueError):
            nu_star_from_n_T_B_R_epsilon_kappa_I(9.06e19, 8.6e3, 5.3, 6.2, 0.32, 1.7, -1.0)
        with self.assertRaises(ValueError):
            omega_i_tau_E_from_B_tau_E_M(5.3, 0.0, 2.5)
        with self.assertRaises(ValueError):
            check_kadomtsev_constraint(0.0, 0.0, 0.0, 0.0, 0.0, tol=0.0)

    def test_vectorized_inputs_are_supported(self):
        n = np.array([9.06e19, 7.0e19], dtype=float)
        t = np.array([8.6e3, 7.0e3], dtype=float)
        b = np.array([5.3, 4.8], dtype=float)
        r = np.array([6.2, 5.8], dtype=float)
        eps = np.array([0.32, 0.30], dtype=float)
        kappa = np.array([1.7, 1.6], dtype=float)
        i_p = np.array([15e6, 12e6], dtype=float)
        m_eff = np.array([2.5, 2.2], dtype=float)
        tau = np.array([3.0, 2.0], dtype=float)

        rho = rho_star_from_M_T_B_R_epsilon(m_eff, t, b, r, eps)
        beta = beta_t_from_n_T_B(n, t, b)
        q = q_cyl_from_B_R_epsilon_kappa_I(b, r, eps, kappa, i_p)
        nu = nu_star_from_n_T_B_R_epsilon_kappa_I(n, t, b, r, eps, kappa, i_p)
        omega_tau = omega_i_tau_E_from_B_tau_E_M(b, tau, m_eff)

        self.assertEqual(rho.shape, (2,))
        self.assertEqual(beta.shape, (2,))
        self.assertEqual(q.shape, (2,))
        self.assertEqual(nu.shape, (2,))
        self.assertEqual(omega_tau.shape, (2,))

        self.assertTrue(np.all(np.isfinite(rho)))
        self.assertTrue(np.all(np.isfinite(beta)))
        self.assertTrue(np.all(np.isfinite(q)))
        self.assertTrue(np.all(np.isfinite(nu)))
        self.assertTrue(np.all(np.isfinite(omega_tau)))

    def test_kadomtsev_constraint_value_and_checker(self):
        # Exactly satisfies: 4*aR - 8*an - aI - 3*aP - 5*aB - 5 = 0
        a_i, a_b, a_p, a_n, a_r = 0.0, 0.0, 1.0, 0.0, 2.0
        alpha_k = kadomtsev_constraint_from_engineering_exponents(a_i, a_b, a_p, a_n, a_r)
        self.assertAlmostEqual(alpha_k, 0.0, places=12)
        self.assertTrue(check_kadomtsev_constraint(a_i, a_b, a_p, a_n, a_r))

        self.assertFalse(check_kadomtsev_constraint(1.0, 0.5, 0.2, 0.1, 1.0))

    def test_aliases_match_canonical_functions(self):
        n = np.array([9.06e19, 8.0e19], dtype=float)
        t = np.array([8.6e3, 7.5e3], dtype=float)
        b = np.array([5.3, 5.0], dtype=float)
        r = np.array([6.2, 6.0], dtype=float)
        eps = np.array([0.32, 0.31], dtype=float)
        kappa = np.array([1.7, 1.6], dtype=float)
        i_p = np.array([15e6, 14e6], dtype=float)
        m = np.array([2.5, 2.3], dtype=float)
        tau = np.array([3.2, 2.8], dtype=float)

        np.testing.assert_allclose(
            coulomb_logarithm(n, t),
            coulomb_logarithm_from_n_T(n, t),
        )
        np.testing.assert_allclose(
            calc_rho_star(m, t, b, r, eps),
            rho_star_from_M_T_B_R_epsilon(m, t, b, r, eps),
        )
        np.testing.assert_allclose(
            calc_beta_t(n, t, b),
            beta_t_from_n_T_B(n, t, b),
        )
        np.testing.assert_allclose(
            calc_q_cyl(b, r, eps, kappa, i_p),
            q_cyl_from_B_R_epsilon_kappa_I(b, r, eps, kappa, i_p),
        )
        np.testing.assert_allclose(
            calc_nu_star(n, t, b, r, eps, kappa, i_p),
            nu_star_from_n_T_B_R_epsilon_kappa_I(n, t, b, r, eps, kappa, i_p),
        )
        np.testing.assert_allclose(
            calc_omega_i_tau_E(b, tau, m),
            omega_i_tau_E_from_B_tau_E_M(b, tau, m),
        )

    def test_omega_i_tau_E_matches_manual_si_calculation(self):
        b = 5.3
        tau = 3.5
        m_eff = 2.5
        z_i = 1.0
        qe = 1.602176634e-19
        m_p = 1.67262192e-27
        expected = z_i * qe * b * tau / (m_eff * m_p)
        got = omega_i_tau_E_from_B_tau_E_M(b, tau, m_eff, z_i)
        self.assertAlmostEqual(got, expected, places=10)

    def test_beta_t_fraction_output(self):
        n = 9.06e19
        t = 8.6e3
        b = 5.3
        beta_percent = beta_t_from_n_T_B(n, t, b, output="percent")
        beta_fraction = beta_t_from_n_T_B(n, t, b, output="fraction")
        self.assertAlmostEqual(beta_fraction, beta_percent / 100.0, places=12)


if __name__ == "__main__":
    unittest.main()
