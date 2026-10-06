# Reference portfolio audit (#1712)

Server scan: 2026-10-06T07:34:55+00:00 over /srv/vest.filedb, /home/user1/runs/campaign/filedb.

## Shot x stage matrix

| shot | FileDB | diagnostics | eddy | efit | chease | mhd_linear | thomson | ces | core_profiles | electron_efit | kinetic_efit | camera_visible | shotlog | legacy:camera_visible | legacy:camera_visible_fluctuation | legacy:hard_x_rays | legacy:shotlog | legacy:soft_x_rays | packaged-only |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 39915 | campaign | ok 09-30 | ok 09-30 | ok 09-30 |  |  | ok 09-30 | n/a 09-30 | ok 09-30 | ok 09-30 | n/a 09-30 |  |  |  |  |  |  |  |  |
| 39915 | packaged-only |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | dataset_description wall |
| 39915 | production | ok 10-01 | ok 10-02 | STALE 09-02 | STALE 09-02 | STALE 09-03 |  |  |  |  |  |  | ok 09-18 |  |  |  | raw | STALE |  |
| 39916 | campaign | ok 09-30 | ok 09-30 | ok 09-30 |  |  | ok 09-30 | n/a 09-30 | ok 09-30 | ok 09-30 | n/a 09-30 |  |  |  |  |  |  |  |  |
| 39916 | production | ok 10-01 | ok 10-02 | STALE 09-03 |  |  |  |  |  |  |  |  | ok 09-18 |  |  |  | raw | STALE |  |
| 40600 | packaged-only |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | dataset_description wall |
| 40600 | production | ok 10-02 | ok 10-02 | STALE 09-03 |  |  |  |  |  |  |  |  | ok 09-18 |  | STALE | deferred | raw | STALE |  |
| 41524 | packaged-only |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | dataset_description em_coupling wall |
| 41524 | production | ok 10-02 | ok 10-02 | STALE 09-03 |  |  |  |  |  |  |  |  | ok 09-18 |  |  |  | raw | STALE |  |
| 41672 | packaged-only |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | dataset_description em_coupling wall |
| 41672 | production | ok 10-03 | ok 10-03 | STALE 09-03 |  |  |  |  |  |  |  |  | ok 09-18 |  |  |  | raw | STALE |  |
| 42962 | campaign | ok 09-30 | ok 09-30 | ok 09-30 |  |  | ok 09-30 | n/a 09-30 | ok 09-30 | ok 09-30 | n/a 09-30 |  |  |  |  |  |  |  |  |
| 42962 | production | ok 10-03 | ok 10-03 | STALE 09-03 |  |  |  |  |  |  |  | ok 10-04 | ok 09-18 | raw |  |  | raw |  |  |
| 45531 | packaged-only |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | dataset_description wall |
| 45531 | production | ok 09-18 | ok 09-18 |  |  |  |  |  |  |  |  |  | ok 09-18 |  |  |  | raw | STALE |  |
| 48224 | packaged-only |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | charge_exchange core_profiles dataset_description thomson_scattering wall |
| 48224 | production | ok 10-06 | ok 10-06 | FAIL 09-14 |  |  |  |  |  |  |  |  | ok 09-18 |  |  |  | raw | STALE |  |
| 48226 | production | ok 10-01 | ok 10-01 | STALE 09-14 |  |  | ok 09-15 | ok 09-15 | n/a 09-15 | n/a 09-15 | n/a 09-15 | ok 10-05 | ok 09-18 | raw |  |  | raw | STALE |  |

## Per shot

| shot | stage | lineage | IDS | FileDB | output date | state | reason | packaged |
|---|---|---|---|---|---|---|---|---|
| 39915 | diagnostics | - | magnetics | ~/runs/campaign/filedb | 2026-09-30 | available | magnetics channel status 'partial'; stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 39915 | diagnostics | - | pf_active | ~/runs/campaign/filedb | 2026-09-30 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 39915 | diagnostics | - | tf | ~/runs/campaign/filedb | 2026-09-30 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 39915 | diagnostics | - | barometry | ~/runs/campaign/filedb | 2026-09-30 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 39915 | diagnostics | - | spectrometer_uv | ~/runs/campaign/filedb | 2026-09-30 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 39915 | diagnostics | - | langmuir_probes | ~/runs/campaign/filedb | 2026-09-30 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit |  |
| 39915 | diagnostics | - | ec_launchers | ~/runs/campaign/filedb | 2026-09-30 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 39915 | eddy | - | pf_passive | ~/runs/campaign/filedb | 2026-09-30 | available | provenance gap: manifest records no VAFT commit | yes |
| 39915 | efit | magnetic | equilibrium | ~/runs/campaign/filedb | 2026-09-30 | available | provenance gap: manifest records no VAFT commit | yes |
| 39915 | thomson | - | thomson_scattering | ~/runs/campaign/filedb | 2026-09-30 | available | provenance gap: manifest records no VAFT commit |  |
| 39915 | ces | - | charge_exchange | ~/runs/campaign/filedb | 2026-09-30 | input-unavailable | stage manifest status 'unavailable': FileNotFoundError: [Errno 2] No such file or directory: '/srv/vest.diagnostic/IDS_39915.mat'; unavailable: ces; VEST charge exchange; machine era vest-pre-43017-pf1906 |  |
| 39915 | core_profiles | - | core_profiles | ~/runs/campaign/filedb | 2026-09-30 | available | provenance gap: manifest records no VAFT commit |  |
| 39915 | electron_efit | - | equilibrium | ~/runs/campaign/filedb | 2026-09-30 | available | provenance gap: manifest records no VAFT commit | yes |
| 39915 | kinetic_efit | - | equilibrium | ~/runs/campaign/filedb | 2026-09-30 | input-unavailable | stage manifest status 'unavailable': this shot's profiles were built without a measured ion temperature, so it belongs to the other lineage, not kinetic_efit; unavailable: kinetic_efit; VEST kinetic efit; machine era vest-pre-43017-pf1906 | yes |
| 39915 | diagnostics | - | magnetics | /srv/vest.filedb | 2026-10-01 | available | magnetics channel status 'partial'; stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 39915 | diagnostics | - | pf_active | /srv/vest.filedb | 2026-10-01 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 39915 | diagnostics | - | tf | /srv/vest.filedb | 2026-10-01 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 39915 | diagnostics | - | barometry | /srv/vest.filedb | 2026-10-01 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 39915 | diagnostics | - | spectrometer_uv | /srv/vest.filedb | 2026-10-01 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 39915 | diagnostics | - | langmuir_probes | /srv/vest.filedb | 2026-10-01 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit |  |
| 39915 | diagnostics | - | ec_launchers | /srv/vest.filedb | 2026-10-01 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 39915 | eddy | - | pf_passive | /srv/vest.filedb | 2026-10-02 | available | provenance gap: manifest records no VAFT commit | yes |
| 39915 | efit | magnetic | equilibrium | /srv/vest.filedb | 2026-09-02 | regeneration-required | upstream eddy (2026-10-02) is newer | yes |
| 39915 | chease | magnetic | equilibrium | /srv/vest.filedb | 2026-09-02 | regeneration-required | upstream efit (magnetic) is itself stale | yes |
| 39915 | mhd_linear | magnetic | mhd_linear | /srv/vest.filedb | 2026-09-03 | regeneration-required | upstream chease (magnetic) is itself stale |  |
| 39915 | mhd_linear | magnetic | ntms | /srv/vest.filedb | 2026-09-03 | regeneration-required | upstream chease (magnetic) is itself stale |  |
| 39915 | shotlog | - | pulse_schedule | /srv/vest.filedb | 2026-09-18 | available | provenance gap: manifest records no VAFT commit |  |
| 39915 | legacy:shotlog | - | pulse_schedule | /srv/vest.filedb | - | available | raw input; composed OMAS product present (omas/shotlog/39915) |  |
| 39915 | legacy:soft_x_rays | - | soft_x_rays | /srv/vest.filedb | - | regeneration-required | raw input; no valid soft_x_rays OMAS product yet |  |
| 39915 | - | - | dataset_description | - | - | available | machine description; composed from vaft.machine_mapping | yes |
| 39915 | - | - | wall | - | - | available | machine description; composed from vaft.machine_mapping | yes |
| 39916 | diagnostics | - | magnetics | ~/runs/campaign/filedb | 2026-09-30 | available | magnetics channel status 'partial'; stage manifest status 'partial'; provenance gap: manifest records no VAFT commit |  |
| 39916 | diagnostics | - | pf_active | ~/runs/campaign/filedb | 2026-09-30 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit |  |
| 39916 | diagnostics | - | tf | ~/runs/campaign/filedb | 2026-09-30 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit |  |
| 39916 | diagnostics | - | barometry | ~/runs/campaign/filedb | 2026-09-30 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit |  |
| 39916 | diagnostics | - | spectrometer_uv | ~/runs/campaign/filedb | 2026-09-30 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit |  |
| 39916 | diagnostics | - | langmuir_probes | ~/runs/campaign/filedb | 2026-09-30 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit |  |
| 39916 | diagnostics | - | ec_launchers | ~/runs/campaign/filedb | 2026-09-30 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit |  |
| 39916 | eddy | - | pf_passive | ~/runs/campaign/filedb | 2026-09-30 | available | provenance gap: manifest records no VAFT commit |  |
| 39916 | efit | magnetic | equilibrium | ~/runs/campaign/filedb | 2026-09-30 | available | provenance gap: manifest records no VAFT commit |  |
| 39916 | thomson | - | thomson_scattering | ~/runs/campaign/filedb | 2026-09-30 | available | provenance gap: manifest records no VAFT commit |  |
| 39916 | ces | - | charge_exchange | ~/runs/campaign/filedb | 2026-09-30 | input-unavailable | stage manifest status 'unavailable': FileNotFoundError: [Errno 2] No such file or directory: '/srv/vest.diagnostic/IDS_39916.mat'; unavailable: ces; VEST charge exchange; machine era vest-pre-43017-pf1906 |  |
| 39916 | core_profiles | - | core_profiles | ~/runs/campaign/filedb | 2026-09-30 | available | provenance gap: manifest records no VAFT commit |  |
| 39916 | electron_efit | - | equilibrium | ~/runs/campaign/filedb | 2026-09-30 | available | provenance gap: manifest records no VAFT commit |  |
| 39916 | kinetic_efit | - | equilibrium | ~/runs/campaign/filedb | 2026-09-30 | input-unavailable | stage manifest status 'unavailable': this shot's profiles were built without a measured ion temperature, so it belongs to the other lineage, not kinetic_efit; unavailable: kinetic_efit; VEST kinetic efit; machine era vest-pre-43017-pf1906 |  |
| 39916 | diagnostics | - | magnetics | /srv/vest.filedb | 2026-10-01 | available | magnetics channel status 'partial'; stage manifest status 'partial'; provenance gap: manifest records no VAFT commit |  |
| 39916 | diagnostics | - | pf_active | /srv/vest.filedb | 2026-10-01 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit |  |
| 39916 | diagnostics | - | tf | /srv/vest.filedb | 2026-10-01 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit |  |
| 39916 | diagnostics | - | barometry | /srv/vest.filedb | 2026-10-01 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit |  |
| 39916 | diagnostics | - | spectrometer_uv | /srv/vest.filedb | 2026-10-01 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit |  |
| 39916 | diagnostics | - | langmuir_probes | /srv/vest.filedb | 2026-10-01 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit |  |
| 39916 | diagnostics | - | ec_launchers | /srv/vest.filedb | 2026-10-01 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit |  |
| 39916 | eddy | - | pf_passive | /srv/vest.filedb | 2026-10-02 | available | provenance gap: manifest records no VAFT commit |  |
| 39916 | efit | magnetic | equilibrium | /srv/vest.filedb | 2026-09-03 | regeneration-required | upstream eddy (2026-10-02) is newer |  |
| 39916 | shotlog | - | pulse_schedule | /srv/vest.filedb | 2026-09-18 | available | provenance gap: manifest records no VAFT commit |  |
| 39916 | legacy:shotlog | - | pulse_schedule | /srv/vest.filedb | - | available | raw input; composed OMAS product present (omas/shotlog/39916) |  |
| 39916 | legacy:soft_x_rays | - | soft_x_rays | /srv/vest.filedb | - | regeneration-required | raw input; no valid soft_x_rays OMAS product yet |  |
| 40600 | diagnostics | - | magnetics | /srv/vest.filedb | 2026-10-02 | available | magnetics channel status 'partial'; stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 40600 | diagnostics | - | pf_active | /srv/vest.filedb | 2026-10-02 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 40600 | diagnostics | - | tf | /srv/vest.filedb | 2026-10-02 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 40600 | diagnostics | - | barometry | /srv/vest.filedb | 2026-10-02 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 40600 | diagnostics | - | spectrometer_uv | /srv/vest.filedb | 2026-10-02 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 40600 | diagnostics | - | langmuir_probes | /srv/vest.filedb | 2026-10-02 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 40600 | diagnostics | - | ec_launchers | /srv/vest.filedb | 2026-10-02 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 40600 | eddy | - | pf_passive | /srv/vest.filedb | 2026-10-02 | available | provenance gap: manifest records no VAFT commit |  |
| 40600 | efit | magnetic | equilibrium | /srv/vest.filedb | 2026-09-03 | regeneration-required | upstream eddy (2026-10-02) is newer |  |
| 40600 | shotlog | - | pulse_schedule | /srv/vest.filedb | 2026-09-18 | available | provenance gap: manifest records no VAFT commit |  |
| 40600 | legacy:camera_visible_fluctuation | - | camera_visible | /srv/vest.filedb | - | regeneration-required | raw input; no valid camera_visible_fluctuation OMAS product yet | yes |
| 40600 | legacy:hard_x_rays | - | hard_x_rays | /srv/vest.filedb | - | deferred | raw input; no FileDB stage composes it yet |  |
| 40600 | legacy:shotlog | - | pulse_schedule | /srv/vest.filedb | - | available | raw input; composed OMAS product present (omas/shotlog/40600) |  |
| 40600 | legacy:soft_x_rays | - | soft_x_rays | /srv/vest.filedb | - | regeneration-required | raw input; no valid soft_x_rays OMAS product yet |  |
| 40600 | - | - | dataset_description | - | - | available | machine description; composed from vaft.machine_mapping | yes |
| 40600 | - | - | wall | - | - | available | machine description; composed from vaft.machine_mapping | yes |
| 41524 | diagnostics | - | magnetics | /srv/vest.filedb | 2026-10-02 | available | magnetics channel status 'partial'; stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 41524 | diagnostics | - | pf_active | /srv/vest.filedb | 2026-10-02 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 41524 | diagnostics | - | tf | /srv/vest.filedb | 2026-10-02 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 41524 | diagnostics | - | barometry | /srv/vest.filedb | 2026-10-02 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 41524 | diagnostics | - | spectrometer_uv | /srv/vest.filedb | 2026-10-02 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 41524 | diagnostics | - | langmuir_probes | /srv/vest.filedb | 2026-10-02 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 41524 | diagnostics | - | ec_launchers | /srv/vest.filedb | 2026-10-02 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 41524 | eddy | - | pf_passive | /srv/vest.filedb | 2026-10-02 | available | provenance gap: manifest records no VAFT commit | yes |
| 41524 | efit | magnetic | equilibrium | /srv/vest.filedb | 2026-09-03 | regeneration-required | upstream eddy (2026-10-02) is newer | yes |
| 41524 | shotlog | - | pulse_schedule | /srv/vest.filedb | 2026-09-18 | available | provenance gap: manifest records no VAFT commit |  |
| 41524 | legacy:shotlog | - | pulse_schedule | /srv/vest.filedb | - | available | raw input; composed OMAS product present (omas/shotlog/41524) |  |
| 41524 | legacy:soft_x_rays | - | soft_x_rays | /srv/vest.filedb | - | regeneration-required | raw input; no valid soft_x_rays OMAS product yet |  |
| 41524 | - | - | dataset_description | - | - | available | machine description; composed from vaft.machine_mapping | yes |
| 41524 | - | - | em_coupling | - | - | available | machine description; composed from vaft.machine_mapping | yes |
| 41524 | - | - | wall | - | - | available | machine description; composed from vaft.machine_mapping | yes |
| 41672 | diagnostics | - | magnetics | /srv/vest.filedb | 2026-10-03 | available | magnetics channel status 'partial'; stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 41672 | diagnostics | - | pf_active | /srv/vest.filedb | 2026-10-03 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 41672 | diagnostics | - | tf | /srv/vest.filedb | 2026-10-03 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 41672 | diagnostics | - | barometry | /srv/vest.filedb | 2026-10-03 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 41672 | diagnostics | - | spectrometer_uv | /srv/vest.filedb | 2026-10-03 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 41672 | diagnostics | - | langmuir_probes | /srv/vest.filedb | 2026-10-03 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 41672 | diagnostics | - | ec_launchers | /srv/vest.filedb | 2026-10-03 | available | stage manifest status 'partial'; provenance gap: manifest records no VAFT commit | yes |
| 41672 | eddy | - | pf_passive | /srv/vest.filedb | 2026-10-03 | available | provenance gap: manifest records no VAFT commit | yes |
| 41672 | efit | magnetic | equilibrium | /srv/vest.filedb | 2026-09-03 | regeneration-required | upstream eddy (2026-10-03) is newer | yes |
| 41672 | shotlog | - | pulse_schedule | /srv/vest.filedb | 2026-09-18 | available | provenance gap: manifest records no VAFT commit |  |
| 41672 | legacy:shotlog | - | pulse_schedule | /srv/vest.filedb | - | available | raw input; composed OMAS product present (omas/shotlog/41672) |  |
| 41672 | legacy:soft_x_rays | - | soft_x_rays | /srv/vest.filedb | - | regeneration-required | raw input; no valid soft_x_rays OMAS product yet |  |
| 41672 | - | - | dataset_description | - | - | available | machine description; composed from vaft.machine_mapping | yes |
| 41672 | - | - | em_coupling | - | - | available | machine description; composed from vaft.machine_mapping | yes |
| 41672 | - | - | wall | - | - | available | machine description; composed from vaft.machine_mapping | yes |
| 42962 | diagnostics | - | magnetics | ~/runs/campaign/filedb | 2026-09-30 | available | provenance gap: manifest records no VAFT commit |  |
| 42962 | diagnostics | - | pf_active | ~/runs/campaign/filedb | 2026-09-30 | available | provenance gap: manifest records no VAFT commit |  |
| 42962 | diagnostics | - | tf | ~/runs/campaign/filedb | 2026-09-30 | available | provenance gap: manifest records no VAFT commit |  |
| 42962 | diagnostics | - | barometry | ~/runs/campaign/filedb | 2026-09-30 | available | provenance gap: manifest records no VAFT commit |  |
| 42962 | diagnostics | - | spectrometer_uv | ~/runs/campaign/filedb | 2026-09-30 | available | provenance gap: manifest records no VAFT commit |  |
| 42962 | diagnostics | - | langmuir_probes | ~/runs/campaign/filedb | 2026-09-30 | available | provenance gap: manifest records no VAFT commit |  |
| 42962 | diagnostics | - | ec_launchers | ~/runs/campaign/filedb | 2026-09-30 | available | provenance gap: manifest records no VAFT commit |  |
| 42962 | eddy | - | pf_passive | ~/runs/campaign/filedb | 2026-09-30 | available | provenance gap: manifest records no VAFT commit |  |
| 42962 | efit | magnetic | equilibrium | ~/runs/campaign/filedb | 2026-09-30 | available | provenance gap: manifest records no VAFT commit |  |
| 42962 | thomson | - | thomson_scattering | ~/runs/campaign/filedb | 2026-09-30 | available | provenance gap: manifest records no VAFT commit |  |
| 42962 | ces | - | charge_exchange | ~/runs/campaign/filedb | 2026-09-30 | input-unavailable | stage manifest status 'unavailable': FileNotFoundError: [Errno 2] No such file or directory: '/srv/vest.diagnostic/IDS_42962.mat'; unavailable: ces; VEST charge exchange; machine era vest-pre-43017-pf1906 |  |
| 42962 | core_profiles | - | core_profiles | ~/runs/campaign/filedb | 2026-09-30 | available | provenance gap: manifest records no VAFT commit |  |
| 42962 | electron_efit | - | equilibrium | ~/runs/campaign/filedb | 2026-09-30 | available | provenance gap: manifest records no VAFT commit |  |
| 42962 | kinetic_efit | - | equilibrium | ~/runs/campaign/filedb | 2026-09-30 | input-unavailable | stage manifest status 'unavailable': this shot's profiles were built without a measured ion temperature, so it belongs to the other lineage, not kinetic_efit; unavailable: kinetic_efit; VEST kinetic efit; machine era vest-pre-43017-pf1906 |  |
| 42962 | diagnostics | - | magnetics | /srv/vest.filedb | 2026-10-03 | available | provenance gap: manifest records no VAFT commit |  |
| 42962 | diagnostics | - | pf_active | /srv/vest.filedb | 2026-10-03 | available | provenance gap: manifest records no VAFT commit |  |
| 42962 | diagnostics | - | tf | /srv/vest.filedb | 2026-10-03 | available | provenance gap: manifest records no VAFT commit |  |
| 42962 | diagnostics | - | barometry | /srv/vest.filedb | 2026-10-03 | available | provenance gap: manifest records no VAFT commit |  |
| 42962 | diagnostics | - | spectrometer_uv | /srv/vest.filedb | 2026-10-03 | available | provenance gap: manifest records no VAFT commit |  |
| 42962 | diagnostics | - | langmuir_probes | /srv/vest.filedb | 2026-10-03 | available | provenance gap: manifest records no VAFT commit |  |
| 42962 | diagnostics | - | ec_launchers | /srv/vest.filedb | 2026-10-03 | available | provenance gap: manifest records no VAFT commit |  |
| 42962 | eddy | - | pf_passive | /srv/vest.filedb | 2026-10-03 | available | provenance gap: manifest records no VAFT commit |  |
| 42962 | efit | magnetic | equilibrium | /srv/vest.filedb | 2026-09-03 | regeneration-required | upstream eddy (2026-10-03) is newer |  |
| 42962 | camera_visible | - | camera_visible | /srv/vest.filedb | 2026-10-04 | available | provenance gap: manifest records no VAFT commit |  |
| 42962 | shotlog | - | pulse_schedule | /srv/vest.filedb | 2026-09-18 | available | provenance gap: manifest records no VAFT commit |  |
| 42962 | legacy:camera_visible | - | camera_visible | /srv/vest.filedb | - | available | raw input; composed OMAS product present (omas/camera_visible/42962) |  |
| 42962 | legacy:shotlog | - | pulse_schedule | /srv/vest.filedb | - | available | raw input; composed OMAS product present (omas/shotlog/42962) |  |
| 45531 | diagnostics | - | magnetics | /srv/vest.filedb | 2026-09-18 | available | provenance gap: manifest records no VAFT commit | yes |
| 45531 | diagnostics | - | pf_active | /srv/vest.filedb | 2026-09-18 | available | provenance gap: manifest records no VAFT commit | yes |
| 45531 | diagnostics | - | tf | /srv/vest.filedb | 2026-09-18 | available | provenance gap: manifest records no VAFT commit | yes |
| 45531 | diagnostics | - | barometry | /srv/vest.filedb | 2026-09-18 | available | provenance gap: manifest records no VAFT commit | yes |
| 45531 | diagnostics | - | spectrometer_uv | /srv/vest.filedb | 2026-09-18 | available | provenance gap: manifest records no VAFT commit | yes |
| 45531 | diagnostics | - | langmuir_probes | /srv/vest.filedb | 2026-09-18 | available | provenance gap: manifest records no VAFT commit | yes |
| 45531 | diagnostics | - | ec_launchers | /srv/vest.filedb | 2026-09-18 | available | provenance gap: manifest records no VAFT commit | yes |
| 45531 | eddy | - | pf_passive | /srv/vest.filedb | 2026-09-18 | available | provenance gap: manifest records no VAFT commit |  |
| 45531 | shotlog | - | pulse_schedule | /srv/vest.filedb | 2026-09-18 | available | provenance gap: manifest records no VAFT commit |  |
| 45531 | legacy:shotlog | - | pulse_schedule | /srv/vest.filedb | - | available | raw input; composed OMAS product present (omas/shotlog/45531) |  |
| 45531 | legacy:soft_x_rays | - | soft_x_rays | /srv/vest.filedb | - | regeneration-required | raw input; no valid soft_x_rays OMAS product yet | yes |
| 45531 | - | - | dataset_description | - | - | available | machine description; composed from vaft.machine_mapping | yes |
| 45531 | - | - | wall | - | - | available | machine description; composed from vaft.machine_mapping | yes |
| 48224 | diagnostics | - | magnetics | /srv/vest.filedb | 2026-10-06 | available | provenance gap: manifest records no VAFT commit |  |
| 48224 | diagnostics | - | pf_active | /srv/vest.filedb | 2026-10-06 | available | provenance gap: manifest records no VAFT commit |  |
| 48224 | diagnostics | - | tf | /srv/vest.filedb | 2026-10-06 | available | provenance gap: manifest records no VAFT commit |  |
| 48224 | diagnostics | - | barometry | /srv/vest.filedb | 2026-10-06 | available | provenance gap: manifest records no VAFT commit |  |
| 48224 | diagnostics | - | spectrometer_uv | /srv/vest.filedb | 2026-10-06 | available | provenance gap: manifest records no VAFT commit |  |
| 48224 | diagnostics | - | langmuir_probes | /srv/vest.filedb | 2026-10-06 | available | provenance gap: manifest records no VAFT commit |  |
| 48224 | diagnostics | - | ec_launchers | /srv/vest.filedb | 2026-10-06 | available | provenance gap: manifest records no VAFT commit |  |
| 48224 | eddy | - | pf_passive | /srv/vest.filedb | 2026-10-06 | available | provenance gap: manifest records no VAFT commit |  |
| 48224 | efit | magnetic | equilibrium | /srv/vest.filedb | 2026-09-14 | validation-failed | stage manifest status 'no_output': EFIT output unavailable: completed_no_gfiles: returncode=0; gfiles=0 | yes |
| 48224 | shotlog | - | pulse_schedule | /srv/vest.filedb | 2026-09-18 | available | provenance gap: manifest records no VAFT commit |  |
| 48224 | legacy:shotlog | - | pulse_schedule | /srv/vest.filedb | - | available | raw input; composed OMAS product present (omas/shotlog/48224) |  |
| 48224 | legacy:soft_x_rays | - | soft_x_rays | /srv/vest.filedb | - | regeneration-required | raw input; no valid soft_x_rays OMAS product yet |  |
| 48224 | - | - | charge_exchange | - | - | server-unavailable | packaged only; no canonical server product found | yes |
| 48224 | - | - | core_profiles | - | - | server-unavailable | packaged only; no canonical server product found | yes |
| 48224 | - | - | dataset_description | - | - | available | machine description; composed from vaft.machine_mapping | yes |
| 48224 | - | - | thomson_scattering | - | - | server-unavailable | packaged only; no canonical server product found | yes |
| 48224 | - | - | wall | - | - | available | machine description; composed from vaft.machine_mapping | yes |
| 48226 | diagnostics | - | magnetics | /srv/vest.filedb | 2026-10-01 | available | provenance gap: manifest records no VAFT commit |  |
| 48226 | diagnostics | - | pf_active | /srv/vest.filedb | 2026-10-01 | available | provenance gap: manifest records no VAFT commit |  |
| 48226 | diagnostics | - | tf | /srv/vest.filedb | 2026-10-01 | available | provenance gap: manifest records no VAFT commit |  |
| 48226 | diagnostics | - | barometry | /srv/vest.filedb | 2026-10-01 | available | provenance gap: manifest records no VAFT commit |  |
| 48226 | diagnostics | - | spectrometer_uv | /srv/vest.filedb | 2026-10-01 | available | provenance gap: manifest records no VAFT commit |  |
| 48226 | diagnostics | - | langmuir_probes | /srv/vest.filedb | 2026-10-01 | available | provenance gap: manifest records no VAFT commit |  |
| 48226 | diagnostics | - | ec_launchers | /srv/vest.filedb | 2026-10-01 | available | provenance gap: manifest records no VAFT commit |  |
| 48226 | eddy | - | pf_passive | /srv/vest.filedb | 2026-10-01 | available | provenance gap: manifest records no VAFT commit |  |
| 48226 | efit | magnetic | equilibrium | /srv/vest.filedb | 2026-09-14 | regeneration-required | upstream eddy (2026-10-01) is newer |  |
| 48226 | thomson | - | thomson_scattering | /srv/vest.filedb | 2026-09-15 | available | provenance gap: manifest records no VAFT commit |  |
| 48226 | ces | - | charge_exchange | /srv/vest.filedb | 2026-09-15 | available | provenance gap: manifest records no VAFT commit |  |
| 48226 | core_profiles | - | core_profiles | /srv/vest.filedb | 2026-09-15 | input-unavailable | stage manifest status 'unavailable': unavailable: core_profiles; missing: core_profiles:t=298.0ms, core_profiles:t=299.0ms, core_profiles:t=300.0ms, core_profiles:t=301.0ms |  |
| 48226 | electron_efit | - | equilibrium | /srv/vest.filedb | 2026-09-15 | input-unavailable | stage manifest status 'unavailable': no core_profiles slice to reconstruct at; unavailable: electron_efit; VEST electron efit; machine era vest-45967-plus-pf2507 |  |
| 48226 | kinetic_efit | - | equilibrium | /srv/vest.filedb | 2026-09-15 | input-unavailable | stage manifest status 'unavailable': this shot's profiles were built without a measured ion temperature, so it belongs to the other lineage, not kinetic_efit; unavailable: kinetic_efit; VEST kinetic efit; machine era vest-45967-plus-pf2507 |  |
| 48226 | camera_visible | - | camera_visible | /srv/vest.filedb | 2026-10-05 | available | provenance gap: manifest records no VAFT commit |  |
| 48226 | shotlog | - | pulse_schedule | /srv/vest.filedb | 2026-09-18 | available | provenance gap: manifest records no VAFT commit |  |
| 48226 | legacy:camera_visible | - | camera_visible | /srv/vest.filedb | - | available | raw input; composed OMAS product present (omas/camera_visible/48226) |  |
| 48226 | legacy:shotlog | - | pulse_schedule | /srv/vest.filedb | - | available | raw input; composed OMAS product present (omas/shotlog/48226) |  |
| 48226 | legacy:soft_x_rays | - | soft_x_rays | /srv/vest.filedb | - | regeneration-required | raw input; no valid soft_x_rays OMAS product yet |  |

## Portfolio (registry IDS)

| IDS | source | occ | mapping | stages | server available on | packaged on | state | note |
|---|---|---|---|---|---|---|---|---|
| barometry | main | 0 | implemented | diagnostics | 39915, 39916, 40600, 41524, 41672, 42962, 45531, 48224, 48226 | 39915, 40600, 41524, 41672, 45531 | available |  |
| bremsstrahlung_visible | - | 0 | not_implemented | - | - | - | not-reference-qualified | mapping not implemented |
| camera_visible | camera-visible-fluctuation | 0 | implemented | camera_visible_fluctuation | - | 40600 | regeneration-required | producer exists; no valid server product on an audited shot |
| camera_visible | main | 0 | implemented | camera_visible | 42962, 48226 | 40600 | available |  |
| charge_exchange | main | 0 | implemented | ces | 48226 | 48224 | available |  |
| coils_non_axisymmetric | main/chease | 1 | implemented | gpec_ideal | - | - | deferred | stage deferred to #95 |
| core_profiles | kinetic-efit/neoclassical | 0 | producer | neoclassical | - | 48224 | regeneration-required | producer exists; no valid server product on an audited shot |
| core_profiles | main | 0 | producer | core_profiles | 39915, 39916, 42962 | 48224 | available |  |
| core_transport | kinetic-efit/neoclassical | 0 | producer | neoclassical | - | - | regeneration-required | producer exists; no valid server product on an audited shot |
| dataset_description | - | 0 | implemented | - | 39915, 40600, 41524, 41672, 45531, 48224 | 39915, 40600, 41524, 41672, 45531, 48224 | available |  |
| disruption | - | 0 | not_implemented | - | - | - | not-reference-qualified | mapping not implemented |
| ec_launchers | main | 0 | implemented | diagnostics | 39915, 39916, 40600, 41524, 41672, 42962, 45531, 48224, 48226 | 39915, 40600, 41524, 41672, 45531 | available |  |
| em_coupling | - | 0 | implemented | - | 41524, 41672 | 41524, 41672 | available |  |
| equilibrium | electron-efit | 0 | producer | electron_efit | 39915, 39916, 42962 | 39915, 41524, 41672, 48224 | available |  |
| equilibrium | kinetic-efit | 0 | producer | kinetic_efit | - | 39915, 41524, 41672, 48224 | regeneration-required | producer exists; no valid server product on an audited shot |
| equilibrium | main | 0 | producer | efit | 39915, 39916, 42962 | 39915, 41524, 41672, 48224 | available |  |
| equilibrium | main/chease | 0 | producer | chease | - | 39915, 41524, 41672, 48224 | regeneration-required | producer exists; no valid server product on an audited shot |
| gas_injection | - | 0 | not_implemented | - | - | - | not-reference-qualified | mapping not implemented |
| gas_pumping | - | 0 | not_implemented | - | - | - | not-reference-qualified | mapping not implemented |
| hard_x_rays | - | 0 | partial | - | - | - | deferred | no FileDB stage produces it yet |
| interferometer | - | 0 | implemented | - | - | - | deferred | no FileDB stage produces it yet |
| langmuir_probes | main | 0 | implemented | diagnostics | 39915, 39916, 40600, 41524, 41672, 42962, 45531, 48224, 48226 | 40600, 41524, 41672, 45531 | available |  |
| magnetics | impa | 0 | implemented | impa | - | 39915, 40600, 41524, 41672, 45531 | regeneration-required | producer exists; no valid server product on an audited shot |
| magnetics | main | 0 | implemented | diagnostics | 39915, 39916, 40600, 41524, 41672, 42962, 45531, 48224, 48226 | 39915, 40600, 41524, 41672, 45531 | available |  |
| mhd_linear | main/chease | 0 | producer | gpec_ideal, mhd_linear | - | - | regeneration-required | producer exists; no valid server product on an audited shot |
| mhd_linear | main/chease | 1 | producer | gpec_ideal, mhd_linear | - | - | regeneration-required | producer exists; no valid server product on an audited shot |
| nbi | - | 0 | partial | - | - | - | deferred | no FileDB stage produces it yet |
| ntms | main/chease | 0 | producer | mhd_linear | - | - | regeneration-required | producer exists; no valid server product on an audited shot |
| pf_active | main | 0 | implemented | diagnostics | 39915, 39916, 40600, 41524, 41672, 42962, 45531, 48224, 48226 | 39915, 40600, 41524, 41672, 45531 | available |  |
| pf_passive | main | 0 | implemented | eddy | 39915, 39916, 40600, 41524, 41672, 42962, 45531, 48224, 48226 | 39915, 41524, 41672 | available |  |
| pulse_schedule | main | 0 | implemented | shotlog | 39915, 39916, 40600, 41524, 41672, 42962, 45531, 48224, 48226 | - | available |  |
| soft_x_rays | main | 0 | implemented | soft_x_rays | - | 45531 | regeneration-required | producer exists; no valid server product on an audited shot |
| spectrometer_uv | main | 0 | implemented | diagnostics | 39915, 39916, 40600, 41524, 41672, 42962, 45531, 48224, 48226 | 39915, 40600, 41524, 41672, 45531 | available |  |
| tf | main | 0 | implemented | diagnostics | 39915, 39916, 40600, 41524, 41672, 42962, 45531, 48224, 48226 | 39915, 40600, 41524, 41672, 45531 | available |  |
| thomson_scattering | main | 0 | implemented | thomson | 39915, 39916, 42962, 48226 | 48224 | available |  |
| wall | - | 0 | implemented | - | 39915, 40600, 41524, 41672, 45531, 48224 | 39915, 40600, 41524, 41672, 45531, 48224 | available |  |
