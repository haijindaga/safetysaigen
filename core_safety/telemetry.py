"""File-based telemetry: the contract between a robot process and the
debug dashboard.

Design: the robot process only ever WRITES artifacts into one directory
(status.json + timestamped images/text, which the demos already produce);
the dashboard only ever READS them, plus writes params.json which the
robot process may poll. Loose coupling: either side can run without the
other, and new status keys / new artifact kinds show up in the dashboard
automatically.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


class TelemetryWriter:
    def __init__(self, directory: str | Path):
        self.dir = Path(directory).resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        self._params_mtime = 0.0
        self._params: dict = {}

    def write_status(self, **fields):
        """Atomically update status.json (any JSON-able keys)."""
        fields["wall_time"] = time.time()
        tmp = self.dir / "status.json.tmp"
        try:
            tmp.write_text(json.dumps(fields), encoding="utf-8")
            os.replace(tmp, self.dir / "status.json")
        except Exception:
            pass

    def read_params(self) -> dict:
        """Return the current dashboard overrides (cached by mtime)."""
        p = self.dir / "params.json"
        try:
            m = p.stat().st_mtime
            if m != self._params_mtime:
                self._params = json.loads(p.read_text(encoding="utf-8"))
                self._params_mtime = m
        except (OSError, ValueError):
            pass
        return self._params
