"""Dependency-aware job driver that walks the modification plan unattended.

Tasks are defined in run/tasks.py (re-imported every cycle, so tasks can be added while the driver
runs). A task starts when all its dependencies are done and enough CPU / GPU slots are free. A task
is done when its `done()` check passes (output markers), so the driver can be killed and restarted
at any time without repeating finished work. A task whose command line is already running (e.g.
launched by hand) is treated as running, never duplicated.

CPU budget = CPU_TOTAL minus running HBV MATLAB processes (HBV is launched by MATLAB/parallel.sh).

RUN: setsid nohup python run/driver.py > $OUT_ROOT/driver/driver.out 2>&1 &
"""
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import experiment as E  # noqa: E402

CPU_TOTAL = int(os.environ.get("HSP_CPU_TOTAL", 120))
GPU_SLOTS = int(os.environ.get("HSP_GPU_SLOTS", 4))
POLL = 60
MAX_TRIES = 2
STATE = E.DRIVER_DIR
STATE.mkdir(parents=True, exist_ok=True)


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(STATE / "driver.log", "a") as f:
        f.write(line + "\n")


def running_cmdlines():
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()
    return out


def hbv_procs(lines):
    return sum(1 for l in lines if "MATLAB" in l and "workflow_TEST_cal_parallel_HBV(" in l)


def main():
    procs = {}       # name -> (Popen, task)
    tries = json.loads((STATE / "tries.json").read_text()) if (STATE / "tries.json").exists() else {}
    last_status = None
    while True:
        import run.tasks as tasks_mod
        importlib.reload(tasks_mod)
        tasks = {t.name: t for t in tasks_mod.build_tasks()}

        # reap finished children
        for name, (p, t) in list(procs.items()):
            rc = p.poll()
            if rc is None:
                continue
            del procs[name]
            ok = t.done()
            log(f"END   {name} rc={rc} done={ok}")
            if not ok:
                tries[name] = tries.get(name, 0) + 1
                (STATE / "tries.json").write_text(json.dumps(tries, indent=1))

        lines = running_cmdlines()
        cpu_free = CPU_TOTAL - hbv_procs(lines) - sum(t.cpu for _, t in procs.values())
        gpu_free = GPU_SLOTS - sum(t.gpu for _, t in procs.values())
        status = {"done": [], "running": list(procs), "external": [], "waiting": [], "failed": [], "blocked": []}

        for name, t in tasks.items():
            if name in procs:
                continue
            if t.done():
                status["done"].append(name)
                continue
            cmd = t.command()
            if cmd is not None and any(cmd in l for l in lines):
                status["external"].append(name)
                cpu_free -= t.cpu
                gpu_free -= t.gpu
                continue
            if tries.get(name, 0) >= MAX_TRIES:
                status["failed"].append(name)
                continue
            if not all(d in tasks and tasks[d].done() for d in t.deps):
                status["waiting"].append(name)
                continue
            if cmd is None:
                status["blocked"].append(name)     # script not written yet / inputs not ready
                continue
            if t.cpu > cpu_free or t.gpu > gpu_free:
                status["waiting"].append(name)
                continue
            logf = open(STATE / f"{name}.log", "a")
            logf.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {cmd}\n")
            logf.flush()
            env = {**os.environ, "OMP_NUM_THREADS": str(max(1, t.cpu)), "MKL_NUM_THREADS": str(max(1, t.cpu))}
            p = subprocess.Popen(cmd, shell=True, stdout=logf, stderr=subprocess.STDOUT, cwd=str(E.PIPELINE_ROOT),
                                 env=env, start_new_session=True)
            procs[name] = (p, t)
            cpu_free -= t.cpu
            gpu_free -= t.gpu
            log(f"START {name} (cpu {t.cpu}, gpu {t.gpu}): {cmd}")
            status["running"].append(name)

        summary = {k: len(v) for k, v in status.items()}
        (STATE / "status.json").write_text(json.dumps({"time": time.strftime('%Y-%m-%d %H:%M:%S'),
                                                       "counts": summary, **status}, indent=1))
        if summary != last_status:
            log(f"status {summary}; cpu_free {cpu_free} gpu_free {gpu_free}")
            last_status = summary
        if not procs and not status["waiting"] and not status["external"] and not status["blocked"]:
            log("nothing left to run" + (f"; FAILED: {status['failed']}" if status["failed"] else ""))
            break
        time.sleep(POLL)


if __name__ == "__main__":
    main()
