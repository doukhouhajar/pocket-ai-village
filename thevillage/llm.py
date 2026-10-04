from __future__ import annotations
import asyncio
import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol
from .config import Settings

@dataclass
class Completion:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    provider: str | None = None  # which upstream served it (OpenRouter reports this)

class LLM(Protocol):
    name: str
    async def complete(
        self, messages: list[dict[str, str]], schema: dict[str, Any] | None, seed: int
    ) -> Completion: ...


def action_schema(tool_names: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "thought": {"type": "string"},
            "tool": {"type": "string", "enum": tool_names},
            "args": {"type": "object"},
        },
        "required": ["thought", "tool", "args"],
    }

class OpenAICompatLLM:
    def __init__(self, settings: Settings):
        from openai import AsyncOpenAI

        self.s = settings
        self.name = settings.village_model
        self.mode = settings.structured_output
        self.client = AsyncOpenAI(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
        )
        self._sem = asyncio.Semaphore(settings.max_concurrency)
        self.fallbacks: list[str] = []

    def _kwargs(self, messages, schema, seed) -> dict[str, Any]:
        kwargs: dict[str, Any] = dict(
            model=self.s.village_model,
            messages=messages,
            temperature=self.s.temperature,
            max_tokens=self.s.max_tokens,
            seed=seed,
        )
        if schema is not None and self.mode == "json_schema":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "action", "schema": schema},
            }
        elif schema is not None and self.mode == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        if "openrouter.ai" in self.s.openai_base_url:
            provider: dict[str, Any] = {"require_parameters": self.s.openrouter_require_parameters}
            if self.s.openrouter_providers:
                provider["order"] = [p.strip() for p in self.s.openrouter_providers.split(",") if p.strip()]
                provider["allow_fallbacks"] = False
            kwargs["extra_body"] = {"provider": provider}
        return kwargs

    async def complete(self, messages, schema, seed) -> Completion:
        from openai import BadRequestError, NotFoundError

        async with self._sem:
            try:
                r = await self.client.chat.completions.create(**self._kwargs(messages, schema, seed))
            except (BadRequestError, NotFoundError) as e:
                # endpoint rejected strict schemas
                if self.mode != "json_schema":
                    raise
                self.mode = "json_object"
                self.fallbacks.append(f"json_schema -> json_object: {str(e)[:200]}")
                r = await self.client.chat.completions.create(**self._kwargs(messages, schema, seed))
        usage = r.usage
        return Completion(
            text=r.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            provider=getattr(r, "provider", None),
        )


# sees the prompt and the allowed tools and returns an action dict
MockPolicy = Callable[[list[dict[str, str]], list[str], random.Random], dict[str, Any]]

class MockLLM:
    name = "mock"

    def __init__(self, policy: MockPolicy):
        self.policy = policy

    async def complete(self, messages, schema, seed) -> Completion:
        tools = schema["properties"]["tool"]["enum"] if schema else []
        action = self.policy(messages, tools, random.Random(seed))
        return Completion(text=json.dumps(action))

def make_llm(settings: Settings, mock_policy: MockPolicy | None = None) -> LLM:
    if settings.llm_provider == "mock":
        if mock_policy is None:
            raise ValueError("mock provider needs a mock policy")
        return MockLLM(mock_policy)
    return OpenAICompatLLM(settings)
