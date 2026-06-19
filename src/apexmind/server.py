"""Launch / stop a local llama.cpp server as a context manager."""
from __future__ import annotations

import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import requests


def _wait_healthy(host: str, port: int, timeout_s: int) -> None:
    url = f"http://{host}:{port}/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"llama-server did not become healthy on {host}:{port} within {timeout_s}s")


@contextmanager
def llama_server(
    server_bin: str,
    model_path: Path,
    host: str = "127.0.0.1",
    port: int = 8081,
    n_gpu_layers: int = 99,
    ctx: int = 8192,
    startup_timeout_s: int = 240,
    extra_args: Optional[List[str]] = None,
    log_path: Optional[Path] = None,
):
    """Start llama-server for `model_path`, yield once healthy, then terminate it."""
    if not Path(server_bin).exists():
        raise FileNotFoundError(f"llama-server not found: {server_bin}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    args = [
        server_bin,
        "-m", str(model_path),
        "--host", host,
        "--port", str(port),
        "-ngl", str(n_gpu_layers),
        "-c", str(ctx),
        "--no-webui",
    ]
    if extra_args:
        args += extra_args

    log_file = open(log_path, "w", encoding="utf-8") if log_path else subprocess.DEVNULL
    proc = subprocess.Popen(args, stdout=log_file, stderr=subprocess.STDOUT)
    try:
        _wait_healthy(host, port, startup_timeout_s)
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        if log_path and log_file not in (None, subprocess.DEVNULL):
            log_file.close()
