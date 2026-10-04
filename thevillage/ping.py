from __future__ import annotations
import asyncio
import time
from .agent import parse_json_object
from .config import get_settings
from .llm import OpenAICompatLLM, action_schema


async def main() -> None:
    s = get_settings()
    if not s.openai_api_key or s.openai_api_key == "local":
        print("warning: OPENAI_API_KEY is not set")
    llm = OpenAICompatLLM(s)
    messages = [
        {"role": "system", "content": "You are a fisher. Tools:\n- catch(tons: integer)\n"
         'Reply with ONE JSON object: {"thought": "...", "tool": "catch", "args": {"tons": <int>}}'},
        {"role": "user", "content": "The lake holds 100 tons. How many tons do you catch?"},
    ]
    t0 = time.time()
    comp = await llm.complete(messages, action_schema(["catch"]), seed=0)
    dt = time.time() - t0
    obj = parse_json_object(comp.text)
    print(f"model:    {s.village_model}\nendpoint: {s.openai_base_url}\nprovider: {comp.provider}")
    print(f"format:   {llm.mode}{'  (fell back: ' + llm.fallbacks[0] + ')' if llm.fallbacks else ''}")
    print(f"latency:  {dt:.1f}s   tokens in/out: {comp.prompt_tokens}/{comp.completion_tokens}")
    print(f"reply:    {comp.text}")
    ok = bool(obj) and obj.get("tool") == "catch" and isinstance((obj.get("args") or {}).get("tons"), int)
    print("RESULT:   OK, valid action" if ok else "RESULT:   reply is not a valid action; tell Claude")


if __name__ == "__main__":
    asyncio.run(main())
