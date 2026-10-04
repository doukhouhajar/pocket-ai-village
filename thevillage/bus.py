from __future__ import annotations
from dataclasses import dataclass, field
from .log import EventLog

@dataclass
class Message:
    id: str  
    channel: str
    sender: str
    text: str
    round: int | None = None

@dataclass
class MessageBus:
    log: EventLog
    enabled: bool = True
    history: list[Message] = field(default_factory=list)
    _cursor: dict[str, int] = field(default_factory=dict)

    def publish(self, sender: str, text: str, channel: str = "general", round: int | None = None) -> Message | None:
        if not self.enabled and sender != "env":
            return None
        eid = self.log.emit("message", agent=sender, channel=channel, text=text, round=round)
        msg = Message(eid, channel, sender, text, round)
        self.history.append(msg)
        return msg

    def unread(self, agent: str, channel: str = "general") -> list[Message]:
        start = self._cursor.get(agent, 0)
        new = [m for m in self.history[start:] if m.channel == channel and m.sender != agent]
        self._cursor[agent] = len(self.history)
        return new
