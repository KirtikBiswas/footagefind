"""Describe the machine numbers were measured on (every result is labelled with this)."""
from __future__ import annotations

import os
import platform
import re
import subprocess
import sys


def _cpu_model() -> str:
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if sys.platform == "win32":
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_Processor).Name"], capture_output=True, text=True, timeout=10)
            if out.stdout.strip():
                return out.stdout.strip()
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _ram_gib() -> float | None:
    try:
        import psutil
        return round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as f:
            kb = int(re.search(r"MemTotal:\s+(\d+)", f.read()).group(1))
        return round(kb / 2**20, 1)
    except Exception:
        return None


def _hypervisor() -> str | None:
    try:
        with open("/proc/cpuinfo") as f:
            if " hypervisor" in f.read():
                return "yes (virtual machine)"
    except Exception:
        pass
    return None


def hardware_info() -> dict:
    import onnxruntime as ort
    info = {
        "cpu_model": _cpu_model(),
        "logical_cores": os.cpu_count(),
        "ram_gib": _ram_gib(),
        "machine": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "ort_available_providers": ort.get_available_providers(),
    }
    hv = _hypervisor()
    if hv:
        info["virtualized"] = hv
    return info


def hardware_label(info: dict | None = None) -> str:
    info = info or hardware_info()
    return (f"{info['cpu_model']}, {info['logical_cores']} logical cores, {info['ram_gib']} GiB RAM, "
            f"{info['os']} ({info['machine']}), onnxruntime {info['onnxruntime']} CPUExecutionProvider")
