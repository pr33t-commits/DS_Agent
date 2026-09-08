"""One fresh, resource-limited Docker container per Python invocation."""
import subprocess
import tempfile
import uuid
import shutil
import sys
import os
import signal
import threading
from pathlib import Path


class SubprocessExecutor:
    """Portable execution in the selected Python environment; NOT a security sandbox."""

    def __init__(self, inputs: Path, python: str = sys.executable, timeout: int = 45):
        self.inputs = inputs.resolve()
        self.python = python
        self.timeout = timeout

    def check(self):
        try:
            subprocess.run([self.python, "-c", "import pandas, numpy, scipy"],
                           check=True, capture_output=True, timeout=30)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Python executable not found: {self.python}") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("Analysis dependencies missing in executor Python. Install pandas, numpy and scipy. "
                               + exc.stderr.decode(errors="replace")) from exc

    def run(self, code: str) -> dict:
        if len(code) > 24000:
            return {"ok": False, "error": "Code exceeds 24000 characters."}
        prefix = (f"DATA_PATH = {str(self.inputs / 'data.csv')!r}\n"
                  f"COLUMNS_PATH = {str(self.inputs / 'columns.csv')!r}\n")
        captured = bytearray()
        truncated = False
        with tempfile.TemporaryDirectory(prefix="ds-python-") as work:
            process = subprocess.Popen(
                [self.python, "-I", "-u", "-c", prefix + code], cwd=work,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                start_new_session=os.name != "nt")

            def drain():
                nonlocal truncated
                while chunk := process.stdout.read(4096):
                    room = max(0, 16000 - len(captured))
                    captured.extend(chunk[:room])
                    truncated |= len(chunk) > room

            reader = threading.Thread(target=drain, daemon=True)
            reader.start()
            timed_out = False
            try:
                process.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   capture_output=True, timeout=10, check=False)
                    if process.poll() is None:
                        process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            reader.join(timeout=5)
            result = {"ok": not timed_out and process.returncode == 0,
                      "exit_code": process.returncode,
                      "output": captured.decode(errors="replace"), "truncated": truncated}
            if timed_out:
                result["error"] = f"Python timed out after {self.timeout}s."
            return result


class DockerExecutor:
    def __init__(self, inputs: Path, image: str = "ds-understanding-python:local", timeout: int = 45):
        self.inputs = inputs.resolve()
        self.image = image
        self.timeout = timeout

    def check(self):
        docker = shutil.which("docker")
        if docker is None:
            raise RuntimeError(
                "Docker was not found on PATH. Install and start Docker Desktop, "
                "then open a new PowerShell window and run `docker version`."
            )
        try:
            subprocess.run([docker, "version"], check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, timeout=15)
            subprocess.run([docker, "image", "inspect", self.image], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=15)
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode(errors="replace").strip() if exc.stderr else ""
            if "image inspect" in " ".join(map(str, exc.cmd)):
                raise RuntimeError(
                    f"Docker image '{self.image}' was not found. Build it from the project root with: "
                    f"docker build -t {self.image} sandbox"
                ) from exc
            raise RuntimeError(f"Docker is installed but unavailable. Start Docker Desktop. {detail}") from exc

    def run(self, code: str) -> dict:
        if len(code) > 24000:
            return {"ok": False, "error": "Code exceeds 24000 characters."}
        code = "DATA_PATH = '/inputs/data.csv'\nCOLUMNS_PATH = '/inputs/columns.csv'\n" + code
        name = "ds-understanding-" + uuid.uuid4().hex
        command = ["docker", "run", "--rm", "--name", name, "--network=none",
                   "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                   "--memory=1g", "--memory-swap=1g", "--cpus=1", "--pids-limit=64",
                   "--log-driver=none", "--ulimit", "fsize=1048576:1048576",
                   "--tmpfs", "/tmp:rw,noexec,nosuid,size=128m",
                   "--mount", f"type=bind,source={self.inputs},target=/inputs,readonly",
                   "-i", self.image]
        # Spooling stdout/stderr in container bounds output before Docker streams it.
        wrapper = (
            "import contextlib, traceback\n"
            "with open('/tmp/result', 'w+') as result:\n"
            " with contextlib.redirect_stdout(result), contextlib.redirect_stderr(result):\n"
            "  try:\n"
            f"   exec(compile({code!r}, '<agent>', 'exec'), {{'__name__': '__main__'}})\n"
            "  except BaseException:\n"
            "   traceback.print_exc()\n"
            "   failed = True\n"
            "  else: failed = False\n"
            " result.seek(0)\n"
            " text = result.read(16001)\n"
            " print(text[:16000])\n"
            " if len(text) > 16000: print('[OUTPUT TRUNCATED: print smaller aggregates]')\n"
            " raise SystemExit(1 if failed else 0)\n"
        )
        # Files also avoid holding arbitrary subprocess output in host RAM.
        with tempfile.TemporaryFile() as output:
            try:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=output,
                                           stderr=subprocess.STDOUT)
                try:
                    process.communicate(wrapper.encode(), timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()
                    return {"ok": False, "error": f"Python timed out after {self.timeout}s."}
                output.seek(0)
                raw = output.read(16001)
                return {"ok": process.returncode == 0, "exit_code": process.returncode,
                        "output": raw[:16000].decode(errors="replace"), "truncated": len(raw) > 16000}
            finally:
                subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=15, check=False)
