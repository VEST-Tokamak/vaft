"""Canonical OMAS IDS contract specs for the migration test layer."""

from __future__ import annotations


CANONICAL_IDS_SPECS = {
    "dataset_description": {
        "required_paths": [
            "dataset_description.data_entry.machine",
            "dataset_description.data_entry.pulse",
            "dataset_description.data_entry.run",
            "dataset_description.ids_properties.homogeneous_time",
        ],
        "expected_values": {
            "dataset_description.ids_properties.homogeneous_time": 2,
        },
        "series": [],
    },
    "pf_active": {
        "required_paths": [
            "pf_active.ids_properties.homogeneous_time",
            "pf_active.time",
            "pf_active.coil.0.name",
            "pf_active.coil.0.identifier",
            "pf_active.coil.0.current.data",
            "pf_active.coil.0.element.0.turns_with_sign",
            "pf_active.coil.0.element.0.geometry.rectangle.r",
            "pf_active.coil.0.element.0.geometry.rectangle.z",
            "pf_active.coil.0.element.0.geometry.rectangle.width",
            "pf_active.coil.0.element.0.geometry.rectangle.height",
        ],
        "expected_values": {
            "pf_active.ids_properties.homogeneous_time": 1,
        },
        "series": [
            {
                "data_path": "pf_active.coil.0.current.data",
                "time_paths": [
                    "pf_active.coil.0.current.time",
                    "pf_active.time",
                ],
            },
        ],
    },
    "magnetics": {
        "required_paths": [
            "magnetics.ids_properties.homogeneous_time",
            "magnetics.time",
            "magnetics.ip.0.data",
            "magnetics.diamagnetic_flux.0.data",
            "magnetics.flux_loop.0.flux.data",
            "magnetics.flux_loop.0.position.0.r",
            "magnetics.flux_loop.0.position.0.z",
            "magnetics.b_field_pol_probe.0.field.data",
            "magnetics.b_field_pol_probe.0.position.r",
            "magnetics.b_field_pol_probe.0.position.z",
        ],
        "expected_values": {
            "magnetics.ids_properties.homogeneous_time": 1,
        },
        "series": [
            {
                "data_path": "magnetics.ip.0.data",
                "time_paths": [
                    "magnetics.ip.0.time",
                    "magnetics.time",
                ],
            },
            {
                "data_path": "magnetics.diamagnetic_flux.0.data",
                "time_paths": [
                    "magnetics.diamagnetic_flux.0.time",
                    "magnetics.time",
                ],
            },
            {
                "data_path": "magnetics.flux_loop.0.flux.data",
                "time_paths": [
                    "magnetics.flux_loop.0.flux.time",
                    "magnetics.time",
                ],
            },
            {
                "data_path": "magnetics.b_field_pol_probe.0.field.data",
                "time_paths": [
                    "magnetics.b_field_pol_probe.0.field.time",
                    "magnetics.time",
                ],
            },
        ],
    },
    "tf": {
        "required_paths": [
            "tf.ids_properties.homogeneous_time",
            "tf.time",
            "tf.coil.0.current.data",
            "tf.b_field_tor_vacuum_r.data",
            "tf.r0",
        ],
        "expected_values": {
            "tf.ids_properties.homogeneous_time": 1,
        },
        "series": [
            {
                "data_path": "tf.coil.0.current.data",
                "time_paths": [
                    "tf.coil.0.current.time",
                    "tf.time",
                ],
            },
            {
                "data_path": "tf.b_field_tor_vacuum_r.data",
                "time_paths": [
                    "tf.b_field_tor_vacuum_r.time",
                    "tf.time",
                ],
            },
        ],
    },
    "barometry": {
        "required_paths": [
            "barometry.ids_properties.homogeneous_time",
            "barometry.gauge.0.name",
            "barometry.gauge.0.pressure.time",
            "barometry.gauge.0.pressure.data",
        ],
        "expected_values": {
            "barometry.ids_properties.homogeneous_time": 1,
        },
        "series": [
            {
                "data_path": "barometry.gauge.0.pressure.data",
                "time_paths": [
                    "barometry.gauge.0.pressure.time",
                    "barometry.time",
                ],
            },
        ],
    },
    "spectrometer_uv": {
        "required_paths": [
            "spectrometer_uv.ids_properties.homogeneous_time",
            "spectrometer_uv.time",
            "spectrometer_uv.channel.0.name",
            "spectrometer_uv.channel.0.processed_line.0.label",
            "spectrometer_uv.channel.0.processed_line.0.wavelength_central",
            "spectrometer_uv.channel.0.processed_line.0.intensity.data",
        ],
        "expected_values": {
            "spectrometer_uv.ids_properties.homogeneous_time": 1,
        },
        "series": [
            {
                "data_path": "spectrometer_uv.channel.0.processed_line.0.intensity.data",
                "time_paths": [
                    "spectrometer_uv.channel.0.processed_line.0.intensity.time",
                    "spectrometer_uv.time",
                ],
            },
        ],
    },
    "thomson_scattering": {
        "required_paths": [
            "thomson_scattering.ids_properties.homogeneous_time",
            "thomson_scattering.time",
            "thomson_scattering.channel.0.position.r",
            "thomson_scattering.channel.0.position.z",
            "thomson_scattering.channel.0.t_e.data",
            "thomson_scattering.channel.0.n_e.data",
        ],
        "expected_values": {
            "thomson_scattering.ids_properties.homogeneous_time": 1,
        },
        "series": [
            {
                "data_path": "thomson_scattering.channel.0.t_e.data",
                "time_paths": [
                    "thomson_scattering.channel.0.t_e.time",
                    "thomson_scattering.time",
                ],
            },
            {
                "data_path": "thomson_scattering.channel.0.n_e.data",
                "time_paths": [
                    "thomson_scattering.channel.0.n_e.time",
                    "thomson_scattering.time",
                ],
            },
        ],
    },
    "charge_exchange": {
        "required_paths": [
            "charge_exchange.ids_properties.homogeneous_time",
            "charge_exchange.time",
            "charge_exchange.channel.0.position.r.data",
            "charge_exchange.channel.0.ion.0.label",
            "charge_exchange.channel.0.ion.0.intensity.data",
            "charge_exchange.channel.0.ion.0.velocity_tor.data",
            "charge_exchange.channel.0.ion.0.t_i.data",
        ],
        "expected_values": {
            "charge_exchange.ids_properties.homogeneous_time": 1,
        },
        "series": [
            {
                "data_path": "charge_exchange.channel.0.ion.0.intensity.data",
                "time_paths": [
                    "charge_exchange.channel.0.ion.0.intensity.time",
                    "charge_exchange.time",
                ],
            },
            {
                "data_path": "charge_exchange.channel.0.ion.0.velocity_tor.data",
                "time_paths": [
                    "charge_exchange.channel.0.ion.0.velocity_tor.time",
                    "charge_exchange.time",
                ],
            },
            {
                "data_path": "charge_exchange.channel.0.ion.0.t_i.data",
                "time_paths": [
                    "charge_exchange.channel.0.ion.0.t_i.time",
                    "charge_exchange.time",
                ],
            },
        ],
    },
}


SAMPLE_FILE_IDS = {
    "41672.json": (
        "dataset_description",
        "pf_active",
        "magnetics",
        "tf",
        "barometry",
    ),
    "39915.json": (
        "pf_active",
        "magnetics",
        "tf",
        "spectrometer_uv",
    ),
    "thomson_scattering.json": ("thomson_scattering",),
    "vfit_ion_doppler_single.json": ("charge_exchange",),
    "vfit_ion_doppler_profile.json": ("charge_exchange",),
}


LEGACY_PATHS = (
    "flux_loop.loop.0.flux.data",
    "b_field_pol_probe.probe.0.field.data",
    "rogowski_coil.coil.0.current.data",
)
