"""CPU / RAM detection and a lightweight resource sampler.

psutil is preferred but optional; when it is missing the module falls back
to platform/ctypes so core logic still works.
"""

from __future__ import annotations

import ctypes
import os
import platform
import threading
import time
from dataclasses import dataclass, field

try:  # pragma: no cover - psutil may be absent
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except Exception:  # pragma: no cover
    psutil = None  # type: ignore
    _HAS_PSUTIL = False


@dataclass
class HardwareInfo:
    cpu_name: str = ""
    cpu_cores: int = 0
    cpu_threads: int = 0
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    gpu: str = "None (CPU-only)"
    os_name: str = ""


def _ram_info_ctypes() -> tuple[float, float]:
    """Fallback RAM query via GlobalMemoryStatusEx (no psutil)."""
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            gb = 1024**3
            return stat.ullTotalPhys / gb, stat.ullAvailPhys / gb
    except Exception:
        pass
    return 0.0, 0.0


def detect_hardware() -> HardwareInfo:
    """Detect CPU/RAM. CPU-only is always assumed; no GPU detection is performed."""
    info = HardwareInfo(os_name=f"{platform.system()} {platform.release()}")
    info.cpu_cores = os.cpu_count() or 4
    try:
        info.cpu_threads = psutil.cpu_count(logical=True) or info.cpu_cores
    except Exception:
        info.cpu_threads = info.cpu_cores
    try:
        info.cpu_name = platform.processor() or "Unknown CPU"
    except Exception:
        info.cpu_name = "Unknown CPU"

    if _HAS_PSUTIL:
        try:
            vm = psutil.virtual_memory()
            info.ram_total_gb = vm.total / (1024**3)
            info.ram_available_gb = vm.available / (1024**3)
        except Exception:
            info.ram_total_gb, info.ram_available_gb = _ram_info_ctypes()
    else:
        info.ram_total_gb, info.ram_available_gb = _ram_info_ctypes()
    return info


_INTERNET_PROBES = (
    "https://ollama.com",
    "https://cloudflare.com/cdn-cgi/trace",
    "https://www.google.com/generate_204",
)


def has_internet(timeout: float = 3.0, endpoints: tuple[str, ...] | None = None,
                 rounds: int = 2) -> bool:
    """True when the machine can reach the internet (used at startup to
    decide whether Ollama cloud models can be used instead of local ones).

    Probes several well-known endpoints (any HTTP response counts as
    "online" — we only care that a route exists) and retries the full set
    ``rounds`` times, because right after boot a single request to a single
    host frequently fails while connectivity itself is fine.
    """
    try:
        import requests
    except Exception:
        return False

    probes = endpoints or _INTERNET_PROBES
    session = requests.Session()
    try:
        for _ in range(max(1, rounds)):
            for url in probes:
                try:
                    session.head(url, timeout=timeout, allow_redirects=True)
                    return True
                except Exception:
                    continue
        return False
    finally:
        session.close()


def recommended_worker_threads(ram_total_gb: float | None = None) -> int:
    """Conservative worker limit so Whisper/Ollama/Kokoro don't compete for
    every CPU thread at once on a low-resource machine."""
    if ram_total_gb is None:
        ram_total_gb = detect_hardware().ram_total_gb
    threads = os.cpu_count() or 4
    if ram_total_gb < 12:
        return max(1, min(threads // 2, 2))
    if ram_total_gb < 20:
        return max(1, min(threads // 2, 4))
    return max(1, min(threads - 2, 6))


@dataclass
class ResourceSnapshot:
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    ram_percent: float = 0.0
    cpu_percent: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def ram_critical(self) -> bool:
        # Critically low = below ~1.5 GB free, or > 90 % used.
        return (self.ram_total_gb - self.ram_used_gb) < 1.5 or self.ram_percent > 90


class ResourceMonitor:
    """Threaded sampler that publishes snapshots on a callback.

    The callback runs on the sampler thread; UI code must marshal it to the
    UI thread. Polling is kept modest (default 5 s).
    """

    def __init__(self, interval_s: float = 5.0) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._callback = None
        self._lock = threading.Lock()

    def start(self, callback) -> None:
        with self._lock:
            self._callback = callback
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                snapshot = self.sample()
                with self._lock:
                    cb = self._callback
                if cb:
                    cb(snapshot)
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    @staticmethod
    def sample() -> ResourceSnapshot:
        if _HAS_PSUTIL:
            try:
                vm = psutil.virtual_memory()
                return ResourceSnapshot(
                    ram_used_gb=vm.used / (1024**3),
                    ram_total_gb=vm.total / (1024**3),
                    ram_percent=vm.percent,
                    cpu_percent=psutil.cpu_percent(interval=None),
                )
            except Exception:
                pass
        total, avail = _ram_info_ctypes()
        used = total - avail
        return ResourceSnapshot(
            ram_used_gb=used,
            ram_total_gb=total,
            ram_percent=(used / total * 100) if total else 0.0,
            cpu_percent=0.0,
        )
