from __future__ import annotations
import argparse
import json
import signal
from pathlib import Path

def clip(s: str, n: int = 300) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + " ..."

def render(ev: dict, thoughts: bool) -> str | None:
    t, who = ev["type"], ev.get("agent") or ""
    rid = ev["id"]
    if t == "message":
        return f"{rid:>6}  [chat] {who}: {clip(ev['text'])}"
    if t == "action":
        tool, args = ev["tool"], ev["args"]
        if tool in ("say", "stay_silent"):  # the chat line already shows it
            return None
        line = f"{rid:>6}  {who} -> {tool}({clip(json.dumps(args), 200)})"
        if thoughts and ev.get("thought"):
            line += f"\n          thinks: {clip(ev['thought'], 400)}"
        return line
    if t == "tool_result":
        return f"{rid:>6}     {'ok ' if ev.get('ok') else 'ERR'} {clip(ev.get('output', ''), 200)}"
    if t == "invalid_action":
        return f"{rid:>6}  {who} !! invalid action: {ev.get('error')}"
    if t == "env_state":
        return (f"{rid:>6}  == month {ev['round']}: stock {ev['stock_before']} -> caught {ev['total']} "
                f"-> left {ev['remaining']} -> {'COLLAPSE' if ev['collapsed'] else ev['stock_after']}")
    if t == "claim":
        bad = [c["name"] for c in ev.get("checks", []) if c["status"] == "fail"]
        return f"{rid:>6}  {who} REPORTS {ev['status'].upper()} on {ev['task']}: {clip(ev['summary'], 150)} receipts={ev['receipts']}{' FAILED ' + str(bad) if bad else ''}"
    if t == "claim_rejected":
        return f"{rid:>6}  {who} report rejected: {ev['reason']}"
    if t == "task_outcome":
        truth = "achieved" if ev["achieved"] else ("infeasible" if not ev["feasible"] else "NOT achieved")
        return f"{rid:>6}  == {who} / {ev['task']}: claimed {ev['claimed']}, truth: {truth}{', HARM' if ev.get('harm') else ''}"
    if t == "consolidate":
        return f"{rid:>6}  {who} consolidated memory (+{ev['added']} chars, total {ev['memory_len']})"
    return None

def main() -> None:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # quiet exit when piped into head
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--agent", default=None)
    ap.add_argument("--thoughts", action="store_true", help="show each action's stated reasoning")
    a = ap.parse_args()
    d = Path(a.run_dir)
    cfg = json.loads((d / "config.json").read_text())
    print(f"# {cfg['env']}  condition={cfg['condition']}  seed={cfg['seed']}  model={cfg['model']}\n")
    for line in open(d / "events.jsonl"):
        ev = json.loads(line)
        if a.agent and ev.get("agent") not in (a.agent, None, "env"):
            continue
        out = render(ev, a.thoughts)
        if out:
            print(out)
    res = d / "results.json"
    if res.exists():
        r = json.loads(res.read_text())
        print("\n# results")
        for k, v in r.items():
            if not isinstance(v, (list, dict)):
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
