# Issue #266: 6 kW ECH launch condition from CAD

`extract_launcher_geometry.py` reduces the EC officer's `MID 6KW ECH ASSY.stp` to the
launch condition stored in `vest.yaml` under `0: ec_launchers: beams: ech_6kw`. The STEP
file was received on 2026-09-21 and has sha256 `a4f7b9dd…711c`. It is 9.4 MB and is not in
the repository; ask on #266 for a copy.

The file is an AP214 export from Inventor 2026 and contains all nine parts together with the
assembly placement, so the `.ipt`/`.iam` originals are not needed. It can be read on macOS
with `cadquery-ocp`.

| Quantity | Value | How it is derived |
|---|---|---|
| Machine axis | from the `MIDDLE SHIELD PART` cylinder (r = 800 mm, z = ±587 mm); z = 0 is the midpoint of its axial extent, and the script checks that the port holes at −360/0/+360 mm are symmetric about it | CAD |
| Launch point | R = 0.8021 m, z = −0.360 m | Point where the port-tube end surface crosses the bore axis. The end is cut on an r = 800 mm cylinder tilted about 9° and offset from the machine axis, so on the bore axis it sits about 2 mm behind the wall |
| Bore axis | misses the machine axis by 0.005 mm; axial component 0 | CAD |
| Direction | (k_R, k_phi, k_Z) = (−1, 0, 0), so both steering angles are 0 | CAD |
| Polarization | TE10 E-field vertical: the WR340 narrow wall spans Z and the broad wall spans phi | CAD |
| Quartz window | R = 0.873–0.883 m on axis | CAD |
| WR284 end | R = 0.893 m | CAD |
| phi | 210° | Port map (`5ML10`, clock 5, #718); the CAD has no toroidal datum |

The following points are provisional until the EC officer confirms them:

- The z value assumes that the shield's local +Y axis points up and that the shield centre is
  the midplane. Lower tier −360 mm agrees with the port name.
- The installation is assumed to apply to every shot that has ECH power recorded (from about
  shot 29500 on).
