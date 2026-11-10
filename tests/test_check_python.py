"""scripts/check_python.py: plain-language refusal for Python x OS combinations onnxruntime has no wheels for."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("check_python", Path(__file__).parents[1] / "scripts" / "check_python.py")
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)


@pytest.mark.parametrize("ver,system,machine,mac,bad", [
    ((3, 12), "Linux", "x86_64", "", False),
    ((3, 14), "Darwin", "arm64", "15.1", False),
    ((3, 14), "Darwin", "x86_64", "15.1", True),   # Intel Mac: onnxruntime 1.23 was the last, up to Python 3.13
    ((3, 13), "Darwin", "x86_64", "14.6", False),
    ((3, 13), "Darwin", "x86_64", "12.7", True),   # macOS 12: onnxruntime 1.19 (up to Python 3.12)
    ((3, 12), "Darwin", "x86_64", "12.7", False),
    ((3, 12), "Darwin", "x86_64", "11.7", False),  # Big Sur Intel: pinned set (constraints/macos11-intel.txt)
    ((3, 13), "Darwin", "x86_64", "11.7", True),   # onnxruntime 1.19 (last for macOS 11) stops at Python 3.12
    ((3, 12), "Darwin", "arm64", "11.7", True),    # scipy arm64 wheels need macOS 12
    ((3, 12), "Darwin", "x86_64", "10.15.7", True),
    ((3, 14), "Darwin", "arm64", "13.6", True),
    ((3, 13), "Darwin", "arm64", "13.6", False),
    ((3, 15), "Windows", "AMD64", "", True),
    ((3, 10), "Linux", "x86_64", "", True),
])
def test_problem(monkeypatch, ver, system, machine, mac, bad):
    monkeypatch.setattr(cp, "sys", SimpleNamespace(version_info=ver, maxsize=sys.maxsize))
    monkeypatch.setattr(cp.platform, "system", lambda: system)
    monkeypatch.setattr(cp.platform, "machine", lambda: machine)
    monkeypatch.setattr(cp.platform, "mac_ver", lambda: (mac, ("", "", ""), ""))
    assert (cp.problem() is not None) == bad


def test_constraints_only_for_big_sur_intel(monkeypatch):
    monkeypatch.setattr(cp.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cp.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(cp.platform, "mac_ver", lambda: ("11.7.10", ("", "", ""), ""))
    path = cp.constraints()
    assert path == "constraints/macos11-intel.txt" and (Path(__file__).parents[1] / path).exists()
    monkeypatch.setattr(cp.platform, "mac_ver", lambda: ("13.6", ("", "", ""), ""))
    assert cp.constraints() is None
