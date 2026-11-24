"""S4-X X2: device-level GPU0 memory sampler (separate process, NOT in the timed path).  Preregistration goals/S4X_X2_whole_process_memory_bound_prereg_v1.md sec 2 ("NVML >=10 Hz").

Samples ONLY GPU0 (UUID GPU-722679a6-175b-cb56-b764-e8124d262c41; any other --uuid is refused, the other GPU is never opened) and writes a CSV  t_unix,used_mb[,procs]  :
  t_unix   time.time() at the sample (same clock as the per-request records of scripts/serve_trace_S4X_X2.py)
  used_mb  device memory used in MiB (NVML nvmlDeviceGetMemoryInfo.used / 2**20) = what nvidia-smi shows: CUDA contexts of ALL processes on GPU0 + allocator memory (graph pools included)
  procs    (every --proc-every-th sample, else empty) "pid:used_mb|pid:used_mb" of the compute processes on GPU0 if the driver reports it ("na" when the driver reports N/A)
Backends (--backend auto = first that works): pynvml (not installed in p24_diff) -> ctypes on libnvidia-ml.so.1 (no dependency; used here) -> nvidia-smi streaming (-lms) -> fake (CPU tests only).
Stop: SIGTERM / SIGINT, or the --stop-file appears, or --max-s elapsed.  At exit <csv without .csv>.summary.json is written (backend, n, achieved rate, max gap, peak, ...).
Summarize an existing CSV over a window:  --summarize FILE.csv [--t0 T --t1 T] [--out X.json]  (peak / median / baseline before t0 ...).

Usage  : taskset -c 31 "$PROJ_PYTHON" -B scripts/nvml_sampler_S4X_X2.py --out X.nvml.csv --hz 20 --stop-file X.stop &
Probe  : ... --probe-s 2 --out P.csv   (samples for 2 s then exits; the devblock does this once at the start to check the sampler and to record the idle baseline)
Selftest (no GPU, no NVML call): --selftest   (loads libnvidia-ml symbols only; runs the fake backend)
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

GPU0_UUID = "GPU-722679a6-175b-cb56-b764-e8124d262c41"
MIB = 1024 * 1024


class NvmlError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------------------------ backends
class _Mem(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class _ProcInfo(ctypes.Structure):               # nvmlProcessInfo_v2_t / nvmlProcessInfo_t of the R535 header (24 bytes)
    _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong), ("gpuInstanceId", ctypes.c_uint), ("computeInstanceId", ctypes.c_uint)]


NVML_VALUE_NOT_AVAILABLE = 0xFFFFFFFFFFFFFFFF
NVML_ERROR_INSUFFICIENT_SIZE = 7
NVML_SYMBOLS = ("nvmlInit_v2", "nvmlShutdown", "nvmlDeviceGetHandleByUUID", "nvmlDeviceGetMemoryInfo", "nvmlDeviceGetUUID", "nvmlErrorString")


def load_libnvml(path: Optional[str] = None):
    """ctypes handle of libnvidia-ml (symbol presence checked, nothing initialised)."""
    last = None
    for name in ([path] if path else ["libnvidia-ml.so.1", "libnvidia-ml.so"]):
        try:
            lib = ctypes.CDLL(name)
            missing = [s for s in NVML_SYMBOLS if not hasattr(lib, s)]
            if missing:
                raise NvmlError(f"{name}: missing symbols {missing}")
            return lib
        except OSError as e:
            last = e
    raise NvmlError(f"libnvidia-ml not loadable: {last!r}")


class CtypesNvml:
    name = "ctypes-nvml"

    def __init__(self, uuid: str, lib=None, with_procs: bool = True):
        if uuid != GPU0_UUID:
            raise NvmlError(f"refused: only GPU0 ({GPU0_UUID}) may be sampled, got {uuid}")
        self.lib = lib or load_libnvml()
        self._chk(self.lib.nvmlInit_v2(), "nvmlInit_v2")
        self._inited = True
        self.h = ctypes.c_void_p()
        self._chk(self.lib.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(self.h)), "nvmlDeviceGetHandleByUUID")
        buf = ctypes.create_string_buffer(96)
        self._chk(self.lib.nvmlDeviceGetUUID(self.h, buf, 96), "nvmlDeviceGetUUID")
        if buf.value.decode() != GPU0_UUID:
            raise NvmlError(f"handle UUID {buf.value.decode()} != GPU0")
        self.mem = _Mem()
        self._procs_fn = getattr(self.lib, "nvmlDeviceGetComputeRunningProcesses_v3", None) if with_procs else None
        self._proc_fail = 0
        self.total_mb = None

    def _chk(self, rc, what):
        if rc != 0:
            try:
                self.lib.nvmlErrorString.restype = ctypes.c_char_p
                msg = self.lib.nvmlErrorString(rc).decode()
            except Exception:  # noqa: BLE001
                msg = "?"
            raise NvmlError(f"{what} failed rc={rc} ({msg})")

    def used_mb(self) -> float:
        self._chk(self.lib.nvmlDeviceGetMemoryInfo(self.h, ctypes.byref(self.mem)), "nvmlDeviceGetMemoryInfo")
        if self.total_mb is None:
            self.total_mb = self.mem.total / MIB
        return self.mem.used / MIB

    def procs(self) -> str:
        if self._procs_fn is None or self._proc_fail >= 3:
            return ""
        try:
            n = ctypes.c_uint(0)
            rc = self._procs_fn(self.h, ctypes.byref(n), None)
            if rc == 0 and n.value == 0:
                return ""
            if rc not in (0, NVML_ERROR_INSUFFICIENT_SIZE):
                raise NvmlError(f"procs rc={rc}")
            cap = n.value + 4
            arr = (_ProcInfo * cap)()
            n = ctypes.c_uint(cap)
            rc = self._procs_fn(self.h, ctypes.byref(n), arr)
            if rc != 0:
                raise NvmlError(f"procs rc={rc}")
            return "|".join(f"{arr[i].pid}:" + ("na" if arr[i].usedGpuMemory == NVML_VALUE_NOT_AVAILABLE else f"{arr[i].usedGpuMemory / MIB:.0f}") for i in range(n.value))
        except Exception:  # noqa: BLE001  per-process data are optional: give up after 3 failures
            self._proc_fail += 1
            return ""

    def close(self):
        if getattr(self, "_inited", False):
            self.lib.nvmlShutdown()
            self._inited = False


class PynvmlBackend:
    name = "pynvml"

    def __init__(self, uuid: str, with_procs: bool = True):
        if uuid != GPU0_UUID:
            raise NvmlError(f"refused: only GPU0 ({GPU0_UUID}) may be sampled, got {uuid}")
        import pynvml  # noqa: PLC0415
        self.n = pynvml
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByUUID(uuid)
        u = pynvml.nvmlDeviceGetUUID(self.h)
        u = u.decode() if isinstance(u, bytes) else u
        if u != GPU0_UUID:
            raise NvmlError(f"handle UUID {u} != GPU0")
        self.with_procs = with_procs
        self.total_mb = None

    def used_mb(self) -> float:
        m = self.n.nvmlDeviceGetMemoryInfo(self.h)
        self.total_mb = m.total / MIB
        return m.used / MIB

    def procs(self) -> str:
        if not self.with_procs:
            return ""
        try:
            ps = self.n.nvmlDeviceGetComputeRunningProcesses(self.h)
            return "|".join(f"{p.pid}:" + ("na" if p.usedGpuMemory is None else f"{p.usedGpuMemory / MIB:.0f}") for p in ps)
        except Exception:  # noqa: BLE001
            self.with_procs = False
            return ""

    def close(self):
        try:
            self.n.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


class SmiBackend:
    """nvidia-smi -lms streaming; one sample per output line (timestamp = read time)."""
    name = "nvidia-smi"
    streaming = True

    def __init__(self, uuid: str, interval_ms: int = 50, with_procs: bool = False):
        if uuid != GPU0_UUID:
            raise NvmlError(f"refused: only GPU0 ({GPU0_UUID}) may be sampled, got {uuid}")
        self.p = subprocess.Popen(["nvidia-smi", "-i", uuid, "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", str(interval_ms)],
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
        self.total_mb = None

    def used_mb(self) -> float:
        line = self.p.stdout.readline()
        if not line:
            raise NvmlError("nvidia-smi stream ended")
        return float(line.strip())

    def procs(self) -> str:
        return ""

    def close(self):
        self.p.terminate()


class FakeBackend:
    """CPU-only test backend: used = base + ramp (MiB) -- deterministic, no GPU / NVML access."""
    name = "fake"

    def __init__(self, uuid: str, base: float = 1000.0, slope: float = 400.0, top: float = 5000.0, with_procs: bool = True):
        if uuid != GPU0_UUID:
            raise NvmlError(f"refused: only GPU0 ({GPU0_UUID}) may be sampled, got {uuid}")
        self.t0 = time.time()
        self.base, self.slope, self.top, self.with_procs = base, slope, top, with_procs
        self.total_mb = 24564.0

    def used_mb(self) -> float:
        f = os.environ.get("X2_FAKE_NVML_FILE")          # tests: a fake serve process writes the value the fake device should show
        if f and os.path.exists(f):
            try:
                return float(Path(f).read_text().strip())
            except ValueError:
                pass
        return min(self.top, self.base + self.slope * (time.time() - self.t0))

    def procs(self) -> str:
        return "4242:%d" % int(self.used_mb() - self.base) if self.with_procs else ""

    def close(self):
        pass


def make_backend(kind: str, uuid: str, with_procs: bool, interval_ms: int):
    errs = []
    order = {"auto": ["pynvml", "ctypes", "smi"], "pynvml": ["pynvml"], "ctypes": ["ctypes"], "smi": ["smi"], "fake": ["fake"]}[kind]
    for k in order:
        try:
            if k == "pynvml":
                return PynvmlBackend(uuid, with_procs)
            if k == "ctypes":
                return CtypesNvml(uuid, with_procs=with_procs)
            if k == "smi":
                return SmiBackend(uuid, interval_ms)
            if k == "fake":
                return FakeBackend(uuid, with_procs=with_procs)
        except NvmlError as e:
            if "refused" in str(e):
                raise
            errs.append(f"{k}: {e}")
        except Exception as e:  # noqa: BLE001  ImportError, OSError, ...
            errs.append(f"{k}: {e!r}")
    raise NvmlError("no NVML backend available: " + "; ".join(errs))


# ------------------------------------------------------------------------------------------------------------------ summary
def _pct(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    i = min(len(sorted_vals) - 1, max(0, int(math.ceil(q * len(sorted_vals))) - 1))
    return sorted_vals[i]


def read_csv(path) -> List[tuple]:
    rows = []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split(",", 2)
            if len(parts) < 2 or not parts[1]:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1]), parts[2] if len(parts) > 2 else ""))
            except ValueError:
                continue
    return rows


def parse_procs(s: str) -> dict:
    out = {}
    for item in filter(None, s.split("|")):
        pid, _, mb = item.partition(":")
        try:
            out[int(pid)] = float(mb)
        except ValueError:
            out[int(pid) if pid.isdigit() else -1] = float("nan")
    return out


def summarize_rows(rows: List[tuple], t0: Optional[float] = None, t1: Optional[float] = None) -> dict:
    """Peak / median / rate over the window [t0, t1] (whole file when None); baseline = median of the samples before t0 (idle GPU0 before the process started)."""
    inwin = [r for r in rows if (t0 is None or r[0] >= t0) and (t1 is None or r[0] <= t1)]
    pre = [r[1] for r in rows if t0 is not None and r[0] < t0]
    out = dict(n_samples=len(inwin), t0=t0, t1=t1, n_pre_samples=len(pre), baseline_used_mb=(sorted(pre)[len(pre) // 2] if pre else None))
    if not inwin:
        out.update(peak_used_mb=None)
        return out
    vals = [r[1] for r in inwin]
    ts = [r[0] for r in inwin]
    sv = sorted(vals)
    dur = ts[-1] - ts[0]
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    k = max(range(len(vals)), key=vals.__getitem__)
    out.update(peak_used_mb=vals[k], peak_t_unix=ts[k], peak_t_rel_s=(ts[k] - (t0 if t0 is not None else ts[0])), median_used_mb=sv[len(sv) // 2], p95_used_mb=_pct(sv, 0.95),
               min_used_mb=sv[0], first_used_mb=vals[0], last_used_mb=vals[-1], duration_s=dur, achieved_hz=(len(vals) - 1) / dur if dur > 0 else None,
               max_gap_s=max(gaps) if gaps else None, median_gap_s=(sorted(gaps)[len(gaps) // 2] if gaps else None))
    per: dict = {}
    for _, _, ps in inwin:
        for pid, mb in parse_procs(ps).items():
            if not math.isnan(mb):
                per[pid] = max(per.get(pid, 0.0), mb)
    out["per_pid_peak_mb"] = {str(k2): v for k2, v in sorted(per.items())}
    return out


# ------------------------------------------------------------------------------------------------------------------ run
def run(a) -> int:
    be = make_backend(a.backend, a.uuid, with_procs=a.proc_every > 0, interval_ms=int(1000 / a.hz))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stop = {"flag": False}

    def _h(signum, frame):  # noqa: ARG001
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _h)
    signal.signal(signal.SIGINT, _h)
    period = 1.0 / a.hz
    t_start = time.time()
    n = n_err = 0
    streaming = getattr(be, "streaming", False)
    with open(out, "w", buffering=1) as f:
        f.write("t_unix,used_mb,procs\n")
        nxt = time.perf_counter()
        while not stop["flag"]:
            now = time.time()
            if (a.stop_file and os.path.exists(a.stop_file)) or now - t_start > a.max_s or (a.probe_s and now - t_start >= a.probe_s):
                break
            try:
                u = be.used_mb()
                t = time.time()
                ps = be.procs() if (a.proc_every > 0 and n % a.proc_every == 0) else ""
                f.write(f"{t:.3f},{u:.1f},{ps}\n")
                n += 1
            except Exception as e:  # noqa: BLE001  a missing sample is a gap, never a stop; give up after 200 errors without any sample
                n_err += 1
                if n_err > 200 and n == 0:
                    print(f"sampler giving up: {e!r}", file=sys.stderr)
                    be.close()
                    return 3
            if not streaming:
                nxt += period
                dt = nxt - time.perf_counter()
                if dt > 0:
                    time.sleep(dt)
                else:
                    nxt = time.perf_counter()
    be.close()
    rows = read_csv(out)
    s = summarize_rows(rows)
    s.update(backend=be.name, uuid=a.uuid, requested_hz=a.hz, n_errors=n_err, total_mb=getattr(be, "total_mb", None), csv=str(out), pid=os.getpid())
    Path(str(out).rsplit(".csv", 1)[0] + ".summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({k: s.get(k) for k in ("backend", "n_samples", "achieved_hz", "max_gap_s", "peak_used_mb", "baseline_used_mb", "n_errors")}), flush=True)
    return 0 if s["n_samples"] else 3


def selftest() -> int:
    load_libnvml()                # symbol presence only; nothing initialised, no GPU touched
    assert ctypes.sizeof(_ProcInfo) == 24 and ctypes.sizeof(_Mem) == 24
    import tempfile
    d = Path(tempfile.mkdtemp())
    ns = argparse.Namespace(backend="fake", uuid=GPU0_UUID, hz=20.0, out=str(d / "t.csv"), stop_file=None, max_s=30, probe_s=1.5, proc_every=5)
    rc = run(ns)
    s = json.loads((d / "t.summary.json").read_text())
    assert rc == 0 and s["achieved_hz"] >= 10 and s["peak_used_mb"] > 1000, s
    try:
        make_backend("fake", "GPU-00000000-0000-0000-0000-000000000000", True, 50)
        raise AssertionError("any non-GPU0 UUID must be refused")
    except NvmlError as e:
        assert "refused" in str(e)
    print("selftest ok:", json.dumps(dict(libnvml_symbols=len(NVML_SYMBOLS), achieved_hz=round(s["achieved_hz"], 1), n=s["n_samples"], peak=s["peak_used_mb"])))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", help="CSV path (t_unix,used_mb,procs)")
    ap.add_argument("--uuid", default=GPU0_UUID, help="refused unless it is the GPU0 UUID")
    ap.add_argument("--hz", type=float, default=20.0, help="target sampling rate (>= 10 required by the card; default 20)")
    ap.add_argument("--backend", default=os.environ.get("X2_NVML_BACKEND", "auto"), choices=["auto", "pynvml", "ctypes", "smi", "fake"], help="default: env X2_NVML_BACKEND (tests use fake) or auto")
    ap.add_argument("--proc-every", type=int, default=10, help="per-process query every N samples (0 = never)")
    ap.add_argument("--stop-file", default=None)
    ap.add_argument("--max-s", type=float, default=7200.0)
    ap.add_argument("--probe-s", type=float, default=0.0, help="sample for this many seconds and exit (sampler probe)")
    ap.add_argument("--summarize", default=None, help="summarize this CSV (no sampling)")
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.summarize:
        s = summarize_rows(read_csv(a.summarize), a.t0, a.t1)
        txt = json.dumps(s, indent=1)
        if a.out:
            Path(a.out).write_text(txt)
        print(txt)
        return 0
    if not a.out:
        ap.error("--out is required")
    if a.hz < 10:
        ap.error("--hz must be >= 10 (card)")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
