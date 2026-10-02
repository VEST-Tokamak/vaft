# TES patch

TES is not open source, and VAFT ships no build of it. If you build `rtes`
yourself, apply `tes_limiter_and_powell.patch` first. It touches about twenty
lines of `TES/find_psiab.cpp` and `nr/powell.cpp` (issue #1469):

- the limiting-point check used `if(j=max_index)`, an assignment;
- X-point and magnetic-axis refinement could run away (an unbounded `powell`);
- `powell` ended the whole process at its iteration cap.

How to apply it and rebuild: see the TES section of `install/README.md`.
