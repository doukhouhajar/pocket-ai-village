from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any

class EventLog:
    def __init__(self, path: Path | None):
        self.path = path
        self.events: list[dict[str, Any]] = []
        self._by_id: dict[str, dict[str, Any]] = {}
        self._fh = open(path, "a", encoding="utf-8") if path else None

    def emit(self, type: str, agent: str | None = None, **data: Any) -> str:
        eid = f"e{len(self.events)}"
        ev = {"id": eid, "t": round(time.time(), 3), "type": type, "agent": agent, **data}
        self.events.append(ev)
        self._by_id[eid] = ev
        if self._fh:
            self._fh.write(json.dumps(ev, default=str) + "\n")
            self._fh.flush()
        return eid

    def get(self, eid: str) -> dict[str, Any] | None:
        return self._by_id.get(eid)

    def of_type(self, type: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["type"] == type]

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None
