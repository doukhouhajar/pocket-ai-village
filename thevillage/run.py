# run one (environment, condition, seed)
from __future__ import annotations
import argparse
import asyncio
import json
import platform
import shutil
import time
from pathlib import Path
import yaml
from .config import get_settings
from .core import build_village
from .envs import ENVIRONMENTS
from .llm import make_llm


def parse_sets(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        k, _, v = p.partition("=")
        out[k.strip()] = yaml.safe_load(v)
    return out


def condition_name(overrides: dict) -> str:
    return "_".join(f"{k}={str(v).lower()}" for k, v in sorted(overrides.items())) or "default"


async def run_one(env_name: str, seed: int, overrides: dict, n_agents: int, mock: bool,
                  runs_dir: str | None = None, overwrite: bool = False) -> dict:
    s = get_settings()
    if mock:
        s = s.model_copy(update={"llm_provider": "mock"})
    env_cls = ENVIRONMENTS[env_name]
    cfg = env_cls.Config(**overrides)
    env = env_cls(cfg)
    root = Path(runs_dir or s.runs_dir)
    run_dir = root / env_name / condition_name(overrides) / f"seed{seed}"
    if run_dir.exists():
        if not overwrite and (run_dir / "results.json").exists():
            print(f"skip (exists): {run_dir}")
            return json.loads((run_dir / "results.json").read_text())
        shutil.rmtree(run_dir)
    llm = make_llm(s, env_cls.mock_policy())
    v = build_village(env, llm, n_agents, seed, run_dir)
    config = {
        "env": env_name, "seed": seed, "n_agents": n_agents, "condition": overrides,
        "env_config": cfg.model_dump(), "model": llm.name,
        "temperature": s.temperature, "max_tokens": s.max_tokens,
        "python": platform.python_version(), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    t0 = time.time()
    try:
        results = await env.run(v)
    finally:
        v.log.close()
        if v.meta.get("prompt_fh"):
            v.meta["prompt_fh"].close()
    calls = v.log.of_type("llm_call")
    results["wall_seconds"] = round(time.time() - t0, 1)
    results["llm_calls"] = len(calls)
    results["tokens_in"] = sum(c["prompt_tokens"] for c in calls)
    results["tokens_out"] = sum(c["completion_tokens"] for c in calls)
    results["providers"] = sorted({c["provider"] for c in calls if c.get("provider")})
    results["format_fallbacks"] = getattr(llm, "fallbacks", [])
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("env", choices=sorted(ENVIRONMENTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. comm=false")
    ap.add_argument("--agents", type=int, default=5)
    ap.add_argument("--mock", action="store_true", help="no model server; deterministic stub policy")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    res = asyncio.run(run_one(a.env, a.seed, parse_sets(a.set), a.agents, a.mock, a.runs_dir, a.overwrite))
    brief = {k: v for k, v in res.items() if not isinstance(v, (list, dict))}
    print(json.dumps(brief, indent=2, default=str))


if __name__ == "__main__":
    main()
