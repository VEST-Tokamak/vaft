"""Synthetic fixtures for canonical and legacy OMAS payload validation."""

from __future__ import annotations

from copy import deepcopy


_CANONICAL_MINIMAL = {
    "dataset_description": {
        "ids_properties": {"homogeneous_time": 2},
        "data_entry": {
            "machine": "VEST",
            "pulse": 41672,
            "run": 1,
            "user": "tester",
        },
    },
    "pf_active": {
        "ids_properties": {"homogeneous_time": 1},
        "time": [0.24, 0.28, 0.32],
        "coil": [
            {
                "name": "PF1",
                "identifier": "PF1",
                "current": {"data": [0.0, 10.0, 20.0]},
                "element": [
                    {
                        "turns_with_sign": 1,
                        "geometry": {
                            "rectangle": {
                                "r": 0.53,
                                "z": 0.0,
                                "width": 0.0172,
                                "height": 2.4,
                            }
                        },
                    }
                ],
            }
        ],
    },
    "magnetics": {
        "ids_properties": {"homogeneous_time": 1},
        "time": [0.24, 0.28, 0.32],
        "ip": [{"data": [0.0, 5_000.0, 6_000.0]}],
        "diamagnetic_flux": [{"data": [0.0, 0.01, 0.02]}],
        "flux_loop": [
            {
                "flux": {"data": [0.0, 0.1, 0.2]},
                "position": [{"r": 0.25, "z": -0.1}],
            }
        ],
        "b_field_pol_probe": [
            {
                "field": {"data": [0.0, 0.3, 0.4]},
                "position": {"r": 0.3, "z": 0.2},
            }
        ],
    },
    "tf": {
        "ids_properties": {"homogeneous_time": 1},
        "time": [0.24, 0.28, 0.32],
        "coil": [{"current": {"data": [1_000.0, 1_100.0, 1_200.0]}}],
        "b_field_tor_vacuum_r": {"data": [0.05, 0.055, 0.06]},
        "r0": 0.4,
    },
    "barometry": {
        "ids_properties": {"homogeneous_time": 1},
        "gauge": [
            {
                "name": "PKR-251 Main Gauge",
                "pressure": {
                    "time": [0.24, 0.28, 0.32],
                    "data": [0.1, 0.2, 0.3],
                },
            }
        ],
    },
    "spectrometer_uv": {
        "ids_properties": {"homogeneous_time": 1},
        "time": [0.24, 0.28, 0.32],
        "channel": [
            {
                "name": "H alpha Filterscope",
                "processed_line": [
                    {
                        "label": "H-alpha_6563",
                        "wavelength_central": 656.3e-9,
                        "intensity": {"data": [1.0, 1.2, 1.1]},
                    }
                ],
            }
        ],
    },
    "thomson_scattering": {
        "ids_properties": {"homogeneous_time": 1},
        "time": [0.25, 0.3, 0.35],
        "channel": [
            {
                "name": "Polychrometer 1R1",
                "position": {"r": 0.475, "z": 0.0},
                "t_e": {"data": [12.0, 13.0, 14.0]},
                "n_e": {"data": [1.0e18, 1.1e18, 1.2e18]},
            }
        ],
    },
    "charge_exchange": {
        "ids_properties": {"homogeneous_time": 1},
        "time": [0.25, 0.3, 0.35],
        "channel": [
            {
                "name": "Ion Doppler Spectroscopy",
                "position": {"r": {"data": 0.39}},
                "ion": [
                    {
                        "label": "C3+",
                        "intensity": {"data": [5.0, 5.5, 6.0]},
                        "velocity_tor": {"data": [1_000.0, 1_100.0, 1_200.0]},
                        "t_i": {"data": [10.0, 11.0, 12.0]},
                    }
                ],
            }
        ],
    },
}


_LEGACY_MAGNETICS = {
    "magnetics": {
        "ids_properties": {"homogeneous_time": 1},
        "time": [0.24, 0.28, 0.32],
    },
    "flux_loop": {
        "loop": [
            {
                "flux": {"data": [0.0, 0.1, 0.2]},
            }
        ]
    },
    "b_field_pol_probe": {
        "probe": [
            {
                "field": {"data": [0.0, 0.3, 0.4]},
            }
        ]
    },
    "rogowski_coil": {
        "coil": [
            {
                "current": {"data": [0.0, 5_000.0, 6_000.0]},
            }
        ]
    },
}


def canonical_minimal_fixture():
    """Return a deep copy of the canonical minimal migration fixture."""
    return deepcopy(_CANONICAL_MINIMAL)


def legacy_magnetics_fixture():
    """Return a deep copy of a legacy non-canonical magnetics payload."""
    return deepcopy(_LEGACY_MAGNETICS)
