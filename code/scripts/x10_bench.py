"""S4-X X10: p24.bench.main() UNCHANGED, with the model class swapped to SD-Turbo (scripts/x10_common.py) and --model-dir defaulting to the locked
SD-Turbo snapshot.  Same CLI as `python -m p24.bench` (arm / H / W / B / cfg / reps / warm / comp-reps / out / proc-id / cold-tier / profile / nsys-window).
Usage: python -B -m scripts.x10_bench --arm E0 --H 512 --W 512 --B 1 --cfg 0 --reps 20 --out <dir> --proc-id 0
"""
import sys

from scripts import x10_common as X

if "-h" not in sys.argv and "--help" not in sys.argv:
    X.patch_runners()
    if "--model-dir" not in sys.argv:
        sys.argv += ["--model-dir", X.model_dir()]
from p24 import bench  # noqa: E402

if __name__ == "__main__":
    bench.main()
