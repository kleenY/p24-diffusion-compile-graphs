"""S4-X X10: scripts/prewarm_s2.py UNCHANGED (runpy), children redirected to SD-Turbo as in scripts/x10_run_matrix.py.  The pre-warm process of every
(compiled arm, key) is also the COLD-EMPTY-cache measurement (compile_s / first_call_*_s / first_req_s with an empty per-arm cache dir).
Usage (from code/): python -B -m scripts.x10_prewarm --arms C1,C1+G1 --keys 512x512x1x0,... --out <dir> --warm-root "$P24_CACHE/x10/warm_s2"
"""
import pathlib
import runpy
import sys

from scripts import x10_common as X

if "-h" not in sys.argv and "--help" not in sys.argv:
    X.install_subprocess_guard()
runpy.run_path(str(pathlib.Path(__file__).with_name("prewarm_s2.py")), run_name="__main__")
