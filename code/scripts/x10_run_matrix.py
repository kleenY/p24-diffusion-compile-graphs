"""S4-X X10: scripts/run_matrix.py UNCHANGED (executed with runpy; same arguments, same ordering / cache modes / --formal self-check), but each child process
`python -m p24.bench` becomes `python -m scripts.x10_bench` (SD-Turbo) with per-process gates and stderr / pinned-count records (scripts/x10_common.py).
Usage (from code/, under taskset -c 0-15): python -B -m scripts.x10_run_matrix --arms E0,G1,C1,C1+G1 --keys 512x512x1x0,... --procs 5 --out <dir> [run_matrix args]
Environment: X10_LOG_DIR (records dir), X10_MEM_MIN_MB, X10_MEM_DROP_MB, P24_PINNED_FALLBACK=1.
"""
import pathlib
import runpy
import sys

from scripts import x10_common as X

if "-h" not in sys.argv and "--help" not in sys.argv:
    X.install_subprocess_guard()
runpy.run_path(str(pathlib.Path(__file__).with_name("run_matrix.py")), run_name="__main__")
