# Measure a Python process's peak memory on Windows without psutil; the transfer pilot peaks at 0.36 GB

**Symptom.** After two background jobs were killed for low RAM (see
`2026-10-19-2350_background-shell-killed-for-low-ram.md`), the question was how much the transfer pilot needs.
A first probe printed 0 MB everywhere: `ctypes.windll.psapi.GetProcessMemoryInfo(GetCurrentProcess(), ...)`
failed silently, because without `restype` the 64-bit pseudo-handle is truncated and the return value was
not checked.

**Fix.** Declare the types and check the return:

```python
K32 = ctypes.WinDLL("kernel32", use_last_error=True)
K32.GetCurrentProcess.restype = wt.HANDLE
K32.K32GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
K32.K32GetProcessMemoryInfo.restype = wt.BOOL
if not K32.K32GetProcessMemoryInfo(K32.GetCurrentProcess(), ctypes.byref(c), c.cb):
    raise ctypes.WinError(ctypes.get_last_error())
```

(`PMC` is PROCESS_MEMORY_COUNTERS; `PeakWorkingSetSize` = peak RAM, `PeakPagefileUsage` = peak commit.)

**Measured (transfer_pilot_v2, 2026-10-20).** Imports 176 MB; reality plus 12 worlds with ARF 207 MB; peak
361 MB RAM and 442 MB committed, during an exact MLP ground truth. One process needs about 0.5 GB, three about
1.3 GB. The pressure came from elsewhere: Chrome 3.5 GB (50 processes, including pages opened for review),
VS Code 2.4 GB, six Claude sessions 2.1 GB, Edge 0.9 GB, 24 `Rterm` processes 0.5 GB; commit 32.2/37.0 GB.
