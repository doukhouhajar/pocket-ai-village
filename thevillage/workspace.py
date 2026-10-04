from __future__ import annotations
import asyncio
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

MAX_OUT = 1500


def _clip(s: str) -> str:
    return s if len(s) <= MAX_OUT else s[:MAX_OUT] + f"\n...[{len(s) - MAX_OUT} more chars]"


@dataclass
class ShellResult:
    exit_code: int
    output: str


class Workspace:
    def __init__(self, root: Path, backend: str = "docker", image: str = "python:3.11-slim", timeout: float = 20):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if backend not in ("docker", "local"):
            raise ValueError(backend)
        if backend == "docker" and not shutil.which("docker"):
            raise RuntimeError("docker not found; pass shell=local only on a disposable machine")
        self.backend, self.image, self.timeout = backend, image, timeout


    def path(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if not p.is_relative_to(self.root):
            raise PermissionError(f"{rel!r} is outside your workspace")
        return p

    async def bash(self, cmd: str) -> ShellResult:
        if self.backend == "docker":
            argv = [
                "docker", "run", "--rm", "--network", "none", "--memory", "512m", "--cpus", "1",
                "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/work",
                "-v", f"{self.root}:/work", "-w", "/work", self.image, "bash", "-c", cmd,
            ]
            env = None
        else:
            argv = ["bash", "-c", cmd]
            env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(self.root)}
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=self.root, env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), self.timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return ShellResult(124, f"timed out after {self.timeout}s")
        return ShellResult(proc.returncode or 0, _clip(out.decode(errors="replace")))

    def write(self, rel: str, content: str) -> str:
        p = self.path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {len(content)} chars to {rel}"

    def read(self, rel: str) -> str:
        p = self.path(rel)
        if not p.is_file():
            raise FileNotFoundError(f"{rel} does not exist")
        return _clip(p.read_text(errors="replace"))

    def delete(self, rel: str) -> str:
        p = self.path(rel)
        if not p.exists():
            raise FileNotFoundError(f"{rel} does not exist")
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        return f"deleted {rel} (permanent)"

    def listing(self) -> str:
        files = sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file())
        return ", ".join(files) or "(empty)"
