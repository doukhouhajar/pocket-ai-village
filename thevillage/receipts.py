from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, model_validator
from .log import EventLog

SUBSTANTIVE = {"bash", "write_file", "delete_file"}

class Claim(BaseModel):
    agent: str
    task_id: str
    status: Literal["done", "blocked"]
    summary: str = Field(min_length=1)
    receipts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _done_needs_receipts(self) -> Claim:
        if self.status == "done" and not self.receipts:
            raise ValueError("a 'done' report must cite at least one receipt")
        return self


class Check(BaseModel):
    name: str
    status: Literal["pass", "fail", "na"]
    severity: Literal["blocking", "advisory"] = "advisory"
    detail: str = ""

    @property
    def blocks(self) -> bool:
        return self.severity == "blocking" and self.status == "fail"


def receipt_checks(claim: Claim, log: EventLog) -> list[Check]:
    events = [log.get(r) for r in claim.receipts]
    missing = [r for r, e in zip(claim.receipts, events, strict=True) if e is None]
    foreign = [
        r for r, e in zip(claim.receipts, events, strict=True)
        if e is not None and (e["type"] != "tool_result" or e.get("agent") != claim.agent or e.get("task") != claim.task_id)
    ]
    found = [e for e in events if e is not None]
    substantive = [e for e in found if e.get("tool") in SUBSTANTIVE and e.get("ok")]
    checks = [
        Check(name="receipts_exist", severity="blocking",
              status="na" if not claim.receipts else ("fail" if missing else "pass"),
              detail=f"missing: {missing}" if missing else ""),
        Check(name="receipts_owned", severity="blocking",
              status="na" if not found else ("fail" if foreign else "pass"),
              detail=f"not this agent's work on this task: {foreign}" if foreign else ""),
        Check(name="receipts_substantive", severity="advisory",
              status="na" if claim.status != "done" else ("pass" if substantive else "fail"),
              detail="" if substantive else "no successful command, write or delete cited"),
    ]
    return checks
