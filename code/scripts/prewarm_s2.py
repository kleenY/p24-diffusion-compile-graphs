"""Pre-warm the per-arm compile caches for S2 (protocol sec 5 "Pre-warming"; plan 9.4/2.1).
For every (arm, key) one p24.bench process (reps 1, warm 0, comp-reps 0) runs into <warm-root>/<arm with + -> _><dir-suffix>/ .
<warm-root> defaults to $P24_CACHE/warm_dev (development runs never write warm_s2); the formal pre-warm passes
--warm-root $P24_CACHE/warm_s2 (review M1).
A pre-warm process must start and end with NO other compute process on GPU0 (Inductor fixes some kernel configurations at
first run and stores them in the cache). Before each process the driver waits until GPU0 has no other compute process (poll
--poll-s, at most --max-wait-s). On timeout the process is SKIPPED, not run: a line {"skipped": "gpu0 busy ...", arm, key}
goes into <out>/prewarm_log.jsonl and the driver continues with the next entry (redo skipped entries later with the same
command; finished entries have a CSV in <out> and are not repeated unless --redo).
Each log line records start/end time, GPU0 other-process count at start/end (from the bench meta; `contended` = either > 0),
host load, the Inductor FX-graph / AOT-autograd cache counters of that process, first_req_s and the return code. At the end the
driver prints the lists of skipped and contended entries. --dir-suffix _r2 rebuilds an arm into a new directory
(<arm>_r2) when a finished entry turned out to be contended.
Cache-independent arms (E0, E1, G1, G2) are skipped (nothing to warm) unless --include-cache-free.
Processes run pinned to --pin (default 0-15, same as formal runs) via taskset.

Usage: python -m scripts.prewarm_s2 --arms C1,C1+G1 --keys 256x256x1x0 --out results/S2/prewarm --max-wait-s 300 \
           --warm-root "$P24_CACHE/warm_s2"
"""
import argparse, json, os, pathlib, subprocess, sys, time

from p24.bench import gpu0_other_procs

CACHE_FREE = {"E0", "E1", "G1", "G2"}

ap = argparse.ArgumentParser()
ap.add_argument("--arms", required=True)
ap.add_argument("--keys", required=True, help="comma list of HxWxBxcfg")
ap.add_argument("--out", required=True)
ap.add_argument("--max-wait-s", type=int, default=1800)
ap.add_argument("--poll-s", type=int, default=30)
ap.add_argument("--pin", default="0-15", help="taskset CPU list; '' disables pinning")
ap.add_argument("--include-cache-free", action="store_true")
ap.add_argument("--warm-root", default=None, help="parent of the per-arm warm dirs (default $P24_CACHE/warm_dev)")
ap.add_argument("--dir-suffix", default="", help="appended to the per-arm dir name, e.g. _r2 (rebuild into a new dir)")
ap.add_argument("--redo", action="store_true", help="run an entry again even if its CSV already exists in --out")
a = ap.parse_args()

out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
P24_CACHE = pathlib.Path(os.environ["P24_CACHE"])
warm_root = pathlib.Path(a.warm_root) if a.warm_root else P24_CACHE / "warm_dev"
skipped, contended, finished = [], [], 0
log = open(out / "prewarm_log.jsonl", "a")
keys = [tuple(int(v) for v in k.split("x")) for k in a.keys.split(",")]


def tag_of(arm, H, W, B, cfg, pid):
    return f"{arm}_{H}x{W}_b{B}_{'cfg' if cfg else 'nocfg'}_p{pid}"


for arm in a.arms.split(","):
    if arm in CACHE_FREE and not a.include_cache_free:
        log.write(json.dumps({"arm": arm, "skipped": "cache-independent arm"}) + "\n"); log.flush()
        continue
    cdir = warm_root / (arm.replace("+", "_") + a.dir_suffix)
    cdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=str(cdir / "inductor"), TRITON_CACHE_DIR=str(cdir / "triton"))
    for (H, W, B, cfg) in keys:
        pid = 0
        while (out / f"{tag_of(arm, H, W, B, cfg, pid)}.csv").exists():
            pid += 1
        if pid > 0 and not a.redo and any(
                json.loads(l).get("arm") == arm and json.loads(l).get("key") == [H, W, B, cfg]
                and json.loads(l).get("rc") == 0 and json.loads(l).get("cache_dir") == str(cdir)
                for l in open(out / "prewarm_log.jsonl") if l.strip()):
            continue  # this (arm, key) already finished into this cache dir
        t_w0 = time.time()
        while gpu0_other_procs() != 0:
            if time.time() - t_w0 > a.max_wait_s:
                break
            time.sleep(a.poll_s)
        waited = round(time.time() - t_w0, 1)
        if gpu0_other_procs() != 0:  # review M1: never pre-warm while GPU0 is shared; skip and record
            rec = {"arm": arm, "key": [H, W, B, cfg], "cache_dir": str(cdir), "skipped": "gpu0 busy beyond max-wait-s",
                   "waited_s": waited, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            log.write(json.dumps(rec) + "\n"); log.flush()
            skipped.append((arm, f"{H}x{W}x{B}x{cfg}"))
            print(json.dumps(rec), flush=True)
            continue
        contended_start = False
        cmd = ([] if not a.pin else ["taskset", "-c", a.pin]) + [
            sys.executable, "-B", "-m", "p24.bench", "--arm", arm, "--H", str(H), "--W", str(W), "--B", str(B),
            "--cfg", str(cfg), "--reps", "1", "--warm", "0", "--comp-reps", "0", "--out", str(out),
            "--proc-id", str(pid), "--cold-tier", "prewarm"]
        t0 = time.time()
        rec = {"arm": arm, "key": [H, W, B, cfg], "cache_dir": str(cdir), "start": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "waited_s": waited, "contended_start": contended_start, "gpu0_other_procs_before_launch": gpu0_other_procs(),
               "loadavg_before": os.getloadavg()}
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        rec.update(rc=r.returncode, secs=round(time.time() - t0, 1), end=time.strftime("%Y-%m-%dT%H:%M:%S"))
        mf = out / f"{tag_of(arm, H, W, B, cfg, pid)}.meta.json"
        if mf.exists():
            m = json.loads(mf.read_text())
            d = m.get("diag", {})
            rec.update(status=m.get("status"), first_req_s=m.get("first_req_s"),
                       gpu0_other_procs_start=m.get("gpu0_other_procs"), gpu0_other_procs_end=m.get("gpu0_other_procs_end"),
                       loadavg_start=m.get("loadavg"), loadavg_end=m.get("loadavg_end"),
                       fx_cache_hit=d.get("fx_cache_hit"), fx_cache_miss=d.get("fx_cache_miss"),
                       aot_cache_hit=d.get("aot_cache_hit"), aot_cache_miss=d.get("aot_cache_miss"),
                       dyn_unique_graphs=d.get("dyn_unique_graphs"), error=(m.get("error") or "")[:300])
        else:
            rec["stderr_tail"] = r.stderr[-600:]
        rec["contended"] = int(bool(rec.get("gpu0_other_procs_start") or rec.get("gpu0_other_procs_end")
                                    or rec.get("gpu0_other_procs_before_launch") or contended_start))
        log.write(json.dumps(rec) + "\n"); log.flush()
        finished += 1
        # p24.bench exits 0 after recording a failure as data, so also look at the recorded status (2.0b: two C1 768x768x4
        # entries failed with a CUDA error while the summary showed nothing)
        failed = r.returncode != 0 or rec.get("status") != "ok"
        if rec["contended"] or failed:
            contended.append((arm, f"{H}x{W}x{B}x{cfg}", "contended" if rec["contended"] else
                              (f"rc={r.returncode}" if r.returncode != 0 else f"status={rec.get('status')}")))
        print(json.dumps({k: rec.get(k) for k in ("arm", "key", "rc", "secs", "first_req_s", "fx_cache_hit",
                                                   "fx_cache_miss", "contended")}), flush=True)

summary = {"prewarm_done": True, "finished": finished, "skipped": skipped, "contended_or_failed": contended,
           "warm_root": str(warm_root), "dir_suffix": a.dir_suffix}
log.write(json.dumps(summary) + "\n"); log.flush()
print(json.dumps(summary))
