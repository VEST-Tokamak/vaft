# Reduced-family crash after input correction

On 2026-09-08 LLDB launched the pinned AppleClang NICE binary against
`/tmp/nice-666-corrected-v5/families/core`, bypassing only the adapter's
pre-execution family guard for this controlled diagnostic run. Meshing
completed (111 ms), the new COCOS manager reported 11, and VacTH reached
its diagnostic-output path.

Observed stop:

```text
EXC_BAD_ACCESS (code=1, address=0xfffffffffffffff8)
frame #0: nice_recon`VacTHSolver::DebugPlotData() + 996
```

The pinned `src/vacth_solver.cpp:4539` function unconditionally accesses
`_b_meas[_nbprobe - 1]` (line 4592) and the corresponding last flux
measurement at line 4651. An absent family therefore indexes -1; zero residuals
printed
before this crash do not mean a successful reconstruction. The adapter rejects
empty B-pol or flux-loop families before launching this standalone path.
No upstream patch or revision change was made. LLDB required a local debug
launch outside the sandbox; the initial sandboxed launch could not start.
