"""Sandbox backends. The sandbox is part of the method, not plumbing.

DockerSandbox  - real isolation: no network, CPU/mem/pids limits, read-only root FS, non-root,
                 all capabilities dropped, fresh throwaway container per command.
                 Optional gVisor (`runtime="runsc"`).
LocalSandbox   - DEV ONLY. A subprocess with scrubbed env + timeout. NOT isolated (no network
                 block, host FS visible). Arbitrary shell is disabled. Refused for real LLMs
                 unless explicitly overridden on the CLI.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

MAX_OUT = 20_000


@dataclass
class ExecResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    duration: float = 0.0


def _clip(s: str) -> str:
    return s if len(s) <= MAX_OUT else s[:MAX_OUT] + "\n...[truncated]"


class Sandbox(ABC):
    isolated: bool = False
    name: str = "abstract"

    @abstractmethod
    def run_python(self, args: list[str], workdir: Path, timeout: int = 20,
                   readonly: bool = False) -> ExecResult:
        """Run `python <args>` with workdir mounted as /work (cwd)."""

    @abstractmethod
    def run_shell(self, cmd: str, workdir: Path, timeout: int = 20) -> ExecResult:
        """Run a shell command string (agent `shell` tool)."""


class LocalSandbox(Sandbox):
    isolated = False
    name = "local"

    def __init__(self, allow_shell: bool = False):
        self.allow_shell = allow_shell

    def _env(self) -> dict[str, str]:
        keep = {"PATH", "SYSTEMROOT", "TEMP", "TMP", "COMSPEC"}
        env = {k: v for k, v in os.environ.items() if k.upper() in keep}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return env

    def _run(self, argv: list[str] | str, workdir: Path, timeout: int, shell: bool = False):
        t0 = time.time()
        try:
            p = subprocess.run(argv, cwd=str(workdir), env=self._env(), capture_output=True,
                               text=True, timeout=timeout, shell=shell, errors="replace")
            return ExecResult(p.returncode, _clip(p.stdout), _clip(p.stderr), False,
                              time.time() - t0)
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"")
            out = out.decode(errors="replace") if isinstance(out, bytes) else out
            return ExecResult(124, _clip(out), "TIMEOUT", True, time.time() - t0)

    def run_python(self, args, workdir, timeout=20, readonly=False):
        # -E ignores PYTHON* env vars, -s skips user site-packages.
        return self._run([sys.executable, "-E", "-s", *args], workdir, timeout)

    def run_shell(self, cmd, workdir, timeout=20):
        if not self.allow_shell:
            return ExecResult(126, "", "shell tool is disabled in the non-isolated local "
                              "sandbox; use the Docker sandbox.", False, 0.0)
        return self._run(cmd, workdir, timeout, shell=True)


class DockerSandbox(Sandbox):
    isolated = True
    name = "docker"

    def __init__(self, image: str = "python:3.11-slim", memory: str = "512m", cpus: float = 1.0,
                 pids: int = 128, runtime: str | None = None):
        self.image, self.memory, self.cpus, self.pids, self.runtime = (
            image, memory, cpus, pids, runtime)

    @staticmethod
    def available() -> bool:
        if not shutil.which("docker"):
            return False
        try:
            return subprocess.run(["docker", "info"], capture_output=True,
                                  timeout=15).returncode == 0
        except Exception:
            return False

    def _docker_argv(self, workdir: Path, readonly: bool, inner: list[str]) -> list[str]:
        try:  # container user is non-root; let it write the mounted dir (best effort)
            if os.name != "nt":
                os.chmod(workdir, 0o777)
        except OSError:
            pass
        argv = ["docker", "run", "--rm", "--network", "none",
                "--memory", self.memory, "--memory-swap", self.memory,
                "--cpus", str(self.cpus), "--pids-limit", str(self.pids),
                "--read-only", "--tmpfs", "/tmp:rw,size=64m",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--user", "65534:65534", "-e", "PYTHONDONTWRITEBYTECODE=1",
                "-v", f"{workdir.resolve()}:/work:{'ro' if readonly else 'rw'}", "-w", "/work"]
        if self.runtime:
            argv += ["--runtime", self.runtime]
        return argv + [self.image, *inner]

    def _run(self, argv: list[str], timeout: int) -> ExecResult:
        t0 = time.time()
        name = f"sicas-{os.getpid()}-{int(t0 * 1000) % 10**9}"
        argv = argv[:2] + ["--name", name] + argv[2:]
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout + 10,
                               errors="replace")
            return ExecResult(p.returncode, _clip(p.stdout), _clip(p.stderr), False,
                              time.time() - t0)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", name], capture_output=True)
            return ExecResult(124, "", "TIMEOUT", True, time.time() - t0)

    def run_python(self, args, workdir, timeout=20, readonly=False):
        inner = ["timeout", str(timeout), "python", *args]
        return self._run(self._docker_argv(workdir, readonly, inner), timeout)

    def run_shell(self, cmd, workdir, timeout=20):
        inner = ["timeout", str(timeout), "sh", "-c", cmd]
        return self._run(self._docker_argv(workdir, False, inner), timeout)


def get_sandbox(kind: str = "auto", runtime: str | None = None) -> Sandbox:
    if kind in ("auto", "docker") and DockerSandbox.available():
        return DockerSandbox(runtime=runtime)
    if kind == "docker":
        raise RuntimeError("Docker requested but not available (is the daemon running?)")
    return LocalSandbox()
