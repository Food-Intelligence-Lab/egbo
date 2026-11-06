"""Small helpers shared by the parallel experiment runners (run_agent_pilot.py, run_batched.py).

watch_parent: ProcessPoolExecutor initializer. Each worker polls its parent's PID and exits if
    the parent has died, so stopping a runner (even with SIGKILL) never leaves orphaned workers
    that keep calling an API and writing results.
write_once: write a result file exactly once. If two runs of the same task race (e.g. a job was
    restarted while an old one was still running), the first to claim the file keeps it and the
    second discards its result instead of overwriting.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path


def watch_parent(parent_pid: int, interval: float = 2.0) -> None:
    def _loop():
        while True:
            if os.getppid() != parent_pid:
                os._exit(1)
            time.sleep(interval)
    threading.Thread(target=_loop, daemon=True).start()


def write_once(path: Path, text: str, companions: dict | None = None) -> bool:
    """Atomically write `text` to `path` unless it already exists; also write `companions`
    ({path: text}) first, only if the main file is claimed. Returns False if skipped."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = Path(str(path) + ".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    try:
        os.close(fd)
        if path.exists():
            return False
        for cpath, ctext in (companions or {}).items():
            ctmp = Path(str(cpath) + ".tmp"); ctmp.write_text(ctext); ctmp.replace(cpath)
        tmp = Path(str(path) + ".tmp"); tmp.write_text(text); tmp.replace(path)
        return True
    finally:
        lock.unlink(missing_ok=True)
