from __future__ import annotations
import hashlib
import json
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from pydantic import BaseModel, ValidationError
from .llm import LLM, action_schema
from .log import EventLog

@dataclass
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    def render(self) -> str:
        fields = self.args.model_json_schema().get("properties", {})
        sig = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in fields.items())
        return f"- {self.name}({sig}): {self.description}"

class NoArgs(BaseModel):
    pass

class RememberArgs(BaseModel):
    notes: str

@dataclass
class Action:
    tool: str
    args: dict[str, Any]
    thought: str
    event_id: str
    valid: bool = True


OUTPUT_FORMAT = (
    "Reply with ONE JSON object and nothing else:\n"
    '{"thought": "<brief reasoning>", "tool": "<tool name>", "args": {<arguments>}}'
)

def parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None

def _seed(*parts: Any) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)

@dataclass
class Agent:
    name: str
    llm: LLM
    log: EventLog
    system_prompt: str
    run_seed: int
    window: int = 12 
    memory_cap: int = 3000  
    memory_floor: int = 600  
    prompt_sink: Callable[[dict[str, Any]], None] | None = None
    memory: str = ""
    recent: deque[str] = field(init=False)
    calls: int = 0
    invalid: int = 0

    def __post_init__(self) -> None:
        self.recent = deque(maxlen=self.window)
    # observation
    def observe(self, text: str) -> None:
        self.recent.append(text)

    # one model call
    async def _ask(self, user: str, tools: list[Tool], purpose: str) -> tuple[str, dict[str, Any] | None, str]:
        tool_block = "\n".join(t.render() for t in tools)
        system = f"{self.system_prompt}\n\nTools:\n{tool_block}\n\n{OUTPUT_FORMAT}"
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        seed = _seed(self.run_seed, self.name, self.calls)
        self.calls += 1
        comp = await self.llm.complete(messages, action_schema([t.name for t in tools]), seed)
        eid = self.log.emit(
            "llm_call", agent=self.name, purpose=purpose, seed=seed,
            prompt_tokens=comp.prompt_tokens, completion_tokens=comp.completion_tokens,
            provider=comp.provider, response=comp.text,
        )
        if self.prompt_sink:
            self.prompt_sink({"event": eid, "agent": self.name, "messages": messages})
        obj = parse_json_object(comp.text)
        return eid, obj, comp.text

    def _context(self, now: str) -> str:
        mem = self.memory.strip() or "(empty)"
        rec = "\n".join(self.recent) or "(nothing yet)"
        return f"## Memory\n{mem}\n\n## Recent events\n{rec}\n\n## Now\n{now}"

    async def step(self, now: str, tools: list[Tool], round: int | None = None) -> Action:
        by_name = {t.name: t for t in tools}
        user = self._context(now)
        error = ""
        for attempt in range(2):
            prompt = user if not error else f"{user}\n\nYour previous reply was invalid: {error}. Try again."
            call_id, obj, _ = await self._ask(prompt, tools, purpose="act")
            if obj is None:
                error = "not a JSON object"
                continue
            name, args = obj.get("tool"), obj.get("args") or {}
            if name not in by_name:
                error = f"unknown tool {name!r}"
                continue
            try:
                parsed = by_name[name].args.model_validate(args)
            except ValidationError as e:
                error = f"bad arguments for {name}: {e.errors()[0]['msg']}"
                continue
            thought = str(obj.get("thought", ""))
            eid = self.log.emit(
                "action", agent=self.name, round=round, tool=name,
                args=parsed.model_dump(), thought=thought, call=call_id, attempt=attempt,
            )
            return Action(name, parsed.model_dump(), thought, eid)
        self.invalid += 1
        eid = self.log.emit("invalid_action", agent=self.name, round=round, error=error)
        return Action("__invalid__", {}, "", eid, valid=False)

    async def consolidate(self, round: int | None = None) -> None:
        now = (
            "Session ending. Your recent events will be cleared. Write notes on anything "
            "you want to remember from them: commitments, what others did, what worked."
        )
        tools = [Tool("remember", "save notes to your memory", RememberArgs)]
        _, obj, _ = await self._ask(self._context(now), tools, purpose="consolidate")
        notes = str(((obj or {}).get("args") or {}).get("notes", "")).strip()
        if notes:
            tag = f"[round {round}] " if round is not None else ""
            self.memory = f"{self.memory}\n{tag}{notes}".strip()
        self.recent.clear()
        self.log.emit("consolidate", agent=self.name, round=round, added=len(notes), memory_len=len(self.memory))
        if len(self.memory) > self.memory_cap:
            await self._compress(round)

    async def _compress(self, round: int | None) -> None:
        target = self.memory_cap // 2
        now = (
            f"Your memory is too long ({len(self.memory)} chars). Rewrite it to about "
            f"{target} characters. Keep everything you still need."
        )
        tools = [Tool("remember", "replace your memory with this text", RememberArgs)]
        _, obj, _ = await self._ask(self._context(now), tools, purpose="compress")
        new = str(((obj or {}).get("args") or {}).get("notes", "")).strip()
        before = len(self.memory)
        if len(new) >= self.memory_floor:
            self.memory = new
        else:  #refuse a catastrophic rewrite
            self.memory = self.memory[-target:]
        self.log.emit("compress", agent=self.name, round=round, before=before, after=len(self.memory))
