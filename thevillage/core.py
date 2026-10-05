from __future__ import annotations
import asyncio
import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar
from pydantic import BaseModel
from .agent import Agent
from .bus import MessageBus
from .llm import LLM, MockPolicy
from .log import EventLog

AGENT_NAMES = ["Ali", "Ghali", "Sali", "Reem", "Mona", "Yaya", "Omar", "Yassir"]
@dataclass
class Village:
    run_dir: Path | None
    log: EventLog
    bus: MessageBus
    agents: list[Agent]
    seed: int
    rng: random.Random
    meta: dict[str, Any] = field(default_factory=dict)

    async def all_act(self, nows: dict[str, str], tools_for, round: int | None = None):
        return await asyncio.gather(
            *(a.step(nows[a.name], tools_for(a), round) for a in self.agents)
        )
    async def consolidate_all(self, round: int | None = None) -> None:
        await asyncio.gather(*(a.consolidate(round) for a in self.agents))


class Environment(ABC):
    name: ClassVar[str]
    Config: ClassVar[type[BaseModel]]
    def __init__(self, cfg: BaseModel):
        self.cfg = cfg

    @abstractmethod
    def system_prompt(self, agent: str, all_agents: list[str]) -> str: ...

    @abstractmethod
    async def run(self, v: Village) -> dict[str, Any]: ...

    @staticmethod
    @abstractmethod
    def mock_policy() -> MockPolicy: ...

    @property
    def comm(self) -> bool:
        return bool(getattr(self.cfg, "comm", True))

def build_village(
    env: Environment, llm: LLM, n_agents: int, seed: int, run_dir: Path | None, window: int = 12
) -> Village:
    rng = random.Random(seed)
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
    log = EventLog(run_dir / "events.jsonl" if run_dir else None)
    prompt_fh = open(run_dir / "prompts.jsonl", "a", encoding="utf-8") if run_dir else None

    def sink(rec: dict[str, Any]) -> None:
        if prompt_fh:
            prompt_fh.write(json.dumps(rec) + "\n")

    if n_agents > len(AGENT_NAMES):
        raise ValueError(f"{n_agents} agents requested but AGENT_NAMES has only {len(AGENT_NAMES)}")
    names = AGENT_NAMES[:n_agents]
    agents = [
        Agent(
            name=n, llm=llm, log=log, system_prompt=env.system_prompt(n, names),
            run_seed=seed, window=window, prompt_sink=sink if run_dir else None,
        )
        for n in names
    ]
    bus = MessageBus(log=log, enabled=env.comm)
    return Village(run_dir, log, bus, agents, seed, rng, meta={"prompt_fh": prompt_fh})
