import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable


def is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class ProcessManager:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def pid_path(self, name: str) -> Path:
        return self.run_dir / f"{name}.pid"

    def log_path(self, name: str) -> Path:
        return self.run_dir / f"{name}.log"

    def read_pid(self, name: str) -> int | None:
        path = self.pid_path(name)
        if not path.exists():
            return None
        try:
            return int(path.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            return None

    def is_alive(self, name: str) -> bool:
        pid = self.read_pid(name)
        if pid is None:
            return False
        if is_running(pid):
            return True
        self.pid_path(name).unlink(missing_ok=True)
        return False

    def spawn(self, name: str, cmd: list[str], cwd: Path | None = None) -> int:
        log_path = self.log_path(name)
        with open(log_path, "a", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.pid_path(name).write_text(str(proc.pid), encoding="utf-8")
        return proc.pid

    def kill(self, name: str, timeout: float = 15.0) -> bool:
        pid = self.read_pid(name)
        if pid is None:
            return False
        if not is_running(pid):
            self.pid_path(name).unlink(missing_ok=True)
            return False

        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not is_running(pid):
                self.pid_path(name).unlink(missing_ok=True)
                return True
            time.sleep(0.2)

        if is_running(pid):
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.5)

        alive = is_running(pid)
        if not alive:
            self.pid_path(name).unlink(missing_ok=True)
        return not alive

    def wait_all_dead(self, names: list[str], timeout: float = 30.0) -> list[str]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            alive = [name for name in names if self.is_alive(name)]
            if not alive:
                return []
            time.sleep(0.3)
        return [name for name in names if self.is_alive(name)]

    def list_names(self, prefix: str = "") -> list[str]:
        names = []
        for path in self.run_dir.glob("*.pid"):
            name = path.stem
            if prefix and not name.startswith(prefix):
                continue
            names.append(name)
        return sorted(names)


def dispatch_command(parser, handlers: dict[str, Callable]):
    if len(sys.argv) <= 1:
        parser.print_help()
        raise SystemExit(0)

    first = sys.argv[1]
    if first not in handlers:
        parser.print_help()
        raise SystemExit(2)

    args = parser.parse_args()
    handlers[args.command](args)
