"""Logging, progress, and run provenance.

    import logs
    logs.start("dose")          # tees everything to logs/<tag>/<stamp>_dose.log

    for i, x in logs.progress(items, "extracting"):
        ...

    logs.result("dose", {"slope_pos": 0.19, "slope_neg": -0.29})

WHY THIS EXISTS

On a rented GPU you are paying per minute, and a run that prints nothing for
forty minutes gives you no way to tell a slow job from a hung one. Progress
with an ETA tells you whether to wait or kill it.

More importantly: printed numbers vanish when the pod is destroyed. Every run
here writes a log file AND a results.jsonl entry stamped with the model, layer,
git commit, and GPU. Six months later, when you are writing this up and want to
know which commit produced which figure, that record is the only thing that
answers it.
"""

import atexit
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import config

_start = None
_tee = None


class _Tee:
    """Duplicate stdout to a file without swallowing it."""

    def __init__(self, path):
        self.f = open(path, "a", buffering=1, encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def gpu_name():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"


def meta():
    return {
        "time": datetime.now().isoformat(timespec="seconds"),
        "model": config.MODEL,
        "tag": config.MODEL_TAG,
        "layer": config.LAYER,
        "C": config.C,
        "device": config.device(),
        "gpu": gpu_name(),
        "git": git_sha(),
    }


def start(name):
    """Begin a logged run. Safe to call more than once."""
    global _start, _tee
    _start = time.time()
    d = os.path.join("logs", config.MODEL_TAG)
    os.makedirs(d, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(d, f"{stamp}_{name}.log")
    if _tee is None:
        _tee = _Tee(path)
        sys.stdout = _tee
        atexit.register(_close)
    m = meta()
    print(f"\n{'=' * 70}")
    print(f"  {name}   {m['time']}")
    print(f"  {m['model']}  layer {m['layer']}  C {m['C']}")
    print(f"  {m['gpu']}  git {m['git']}")
    print(f"  log -> {path}")
    print(f"{'=' * 70}", flush=True)
    return path


def _close():
    global _tee
    if _tee is not None:
        if _start:
            _tee.write(f"\n[total {(time.time() - _start) / 60:.1f} min]\n")
        sys.stdout = _tee.out
        _tee.close()
        _tee = None


def fmt(sec):
    if sec < 90:
        return f"{sec:.0f}s"
    if sec < 5400:
        return f"{sec / 60:.1f}m"
    return f"{sec / 3600:.1f}h"


def progress(items, label="", every=None):
    """Wrap an iterable, printing rate and ETA.

    Prints at most ~20 times regardless of length, so a 5000-item loop does not
    drown the log.
    """
    items = list(items)
    n = len(items)
    if every is None:
        every = max(1, -(-n // 20))  # ceiling division: at most ~20 prints
    t0 = time.time()
    for i, x in enumerate(items, 1):
        yield i, x
        if i % every == 0 or i == n:
            el = time.time() - t0
            rate = i / el if el else 0
            eta = (n - i) / rate if rate else 0
            bar = "#" * int(20 * i / n)
            print(f"  {label} [{bar:<20}] {i}/{n}  "
                  f"{rate:.1f}/s  elapsed {fmt(el)}  eta {fmt(eta)}",
                  flush=True)


def vram():
    """Peak GPU memory so far, in GB. Returns None on CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e9, 2)
    except Exception:
        pass
    return None


def result(name, values):
    """Append a result record to results.jsonl under the model tag.

    Printed numbers die with the pod. This does not.
    """
    rec = meta()
    rec["experiment"] = name
    rec["vram_peak_gb"] = vram()
    if _start:
        rec["minutes"] = round((time.time() - _start) / 60, 2)
    rec["values"] = {k: (float(v) if hasattr(v, "__float__") else v)
                     for k, v in values.items()}
    path = config.data_path("results.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"\n  recorded -> {path}")
    return rec


def show_results(name=None):
    """Print every recorded run, optionally filtered by experiment."""
    path = config.data_path("results.jsonl")
    if not os.path.exists(path):
        print("no results recorded yet")
        return
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if name:
        rows = [r for r in rows if r["experiment"] == name]
    for r in rows:
        print(f"\n{r['time']}  {r['experiment']}  "
              f"L{r['layer']}  git {r['git']}")
        for k, v in r["values"].items():
            print(f"    {k:28} {v}")


if __name__ == "__main__":
    show_results(sys.argv[1] if len(sys.argv) > 1 else None)
