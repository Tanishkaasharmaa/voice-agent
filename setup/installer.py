"""Installer — user-consented installation of dependencies and models.

Every function takes a progress callback ``(percent: float, message: str)``.
Nothing here runs without an explicit user choice from the DependencyManager.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

import requests

ProgressCB = Callable[[float, str], None]


class InstallError(Exception):
    pass


def _noop_progress(_percent: float, _message: str) -> None:
    pass


# ------------------------------------------------------------------ pip


def pip_install(packages: list[str], progress: ProgressCB = None) -> None:
    progress = progress or _noop_progress
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"] + packages
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=900, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise InstallError("pip install timed out") from exc
    if result.returncode != 0:
        raise InstallError((result.stderr or result.stdout or "")[-500:])
    progress(100.0, "Installed")


def package_installed(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


# -------------------------------------------------------------- download


def download_file(url: str, dest: Path, progress: ProgressCB = None,
                  timeout: float = 60.0) -> None:
    progress = progress or _noop_progress
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=timeout) as resp:
            if resp.status_code >= 400:
                raise InstallError(f"Download failed ({resp.status_code})")
            total = int(resp.headers.get("content-length", 0))
            written = 0
            with open(tmp, "wb") as handle:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    written += len(chunk)
                    if total:
                        progress(min(100.0, written / total * 100), "Downloading...")
                    else:
                        # No content-length: report bytes so the dialog still
                        # shows visible activity while the file grows.
                        progress(0.0, f"Downloading... {written / (1024 * 1024):.1f} MB")
        tmp.replace(dest)
        progress(100.0, "Done")
    except requests.RequestException as exc:
        tmp.unlink(missing_ok=True)
        raise InstallError(f"Download failed: {exc}") from exc


# ---------------------------------------------------------------- ollama


def ollama_binary() -> str | None:
    return shutil.which("ollama")


def download_ollama_installer(dest: Path, progress: ProgressCB = None) -> Path:
    """OllamaSetup.exe — the user is told this requires Internet."""
    url = "https://ollama.com/download/OllamaSetup.exe"
    download_file(url, dest, progress, timeout=30.0)
    return dest


def start_ollama(progress: ProgressCB = None) -> bool:
    """Launch the Ollama app (falls back to `ollama serve`)."""
    progress = progress or _noop_progress
    binary = ollama_binary()
    if not binary:
        return False
    try:
        subprocess.Popen([binary, "serve"], shell=False,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except OSError:
        return False


def pull_ollama_model(host: str, model: str, progress: ProgressCB = None) -> None:
    """Stream an Ollama model pull with progress."""
    progress = progress or _noop_progress
    try:
        with requests.post(
            f"{host.rstrip('/')}/api/pull",
            json={"model": model, "stream": True},
            timeout=(30, 7200),
            stream=True,
        ) as resp:
            if resp.status_code >= 400:
                raise InstallError(f"Pull failed ({resp.status_code}): {resp.text[:300]}")
            # decode_unicode=True only decodes when the response declares an
            # encoding; Ollama's streaming responses often don't, so lines can
            # arrive as bytes. Decode them explicitly instead.
            for raw in resp.iter_lines():
                if not raw:
                    continue
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                line = raw.strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                status = event.get("status", "")
                if status == "success":
                    progress(100.0, "Installed")
                    return
                total = event.get("total") or 0
                completed = event.get("completed") or 0
                if total:
                    progress(min(100.0, completed / total * 100), status or "Downloading...")
                else:
                    progress(0.0, status or "Downloading...")
        raise InstallError("Pull ended without success")
    except requests.RequestException as exc:
        raise InstallError(f"Pull failed: {exc}") from exc


# ---------------------------------------------------------------- generic


def verify_ollama_model(host: str, model: str) -> bool:
    try:
        resp = requests.get(f"{host.rstrip('/')}/api/tags", timeout=5.0)
        if resp.status_code != 200:
            return False
        names = {m.get("name") for m in resp.json().get("models", [])}
        return model in names
    except Exception:
        return False
