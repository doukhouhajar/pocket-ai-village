from __future__ import annotations
import itertools
import json
import math
import sys
from pathlib import Path
from thevillage.metrics import bootstrap_ci, mean, paired_sign_flip_p

KEY_METRICS = {
    "commons": ["survived_rounds", "collapsed_before_final", "efficiency", "sustainable_round_rate", "gini", "commitment_break_rate"],
    "receipts": ["overclaim_rate", "overclaim_rate_infeasible", "honest_abstention_rate", "success_rate",
                 "unnecessary_abstention_rate", "receipt_validity"],
    "principal": ["ask_recall", "ask_precision", "harm_rate_cleanup", "success_unclear", "success_clear",
                  "acted_without_asking_unclear"],
}
COMMON = ["invalid_action_rate", "llm_calls", "wall_seconds"]

OUT = Path(__file__).parent / "out"


def load(env_dir: Path) -> dict[str, dict[int, dict]]:
    data: dict[str, dict[int, dict]] = {}
    for res in sorted(env_dir.glob("*/seed*/results.json")):
        cond = res.parent.parent.name
        seed = int(res.parent.name.removeprefix("seed"))
        data.setdefault(cond, {})[seed] = json.loads(res.read_text())
    return data


def fmt(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.2f}" if isinstance(x, float) else str(x)


def summarize(env: str, runs: Path = Path("runs")) -> str:
    data = load(runs / env)
    if not data:
        return f"no runs under {runs / env}"
    metrics = KEY_METRICS.get(env, []) + COMMON
    lines = [f"# {env}", ""]
    for cond, seeds in data.items():
        lines.append(f"## {cond}  (n = {len(seeds)} seeds: {sorted(seeds)})")
        lines.append("")
        lines.append("| metric | mean | 95% bootstrap CI | per seed |")
        for m in metrics:
            vals = [seeds[s].get(m) for s in sorted(seeds)]
            num = [float(v) for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
            lo, hi = bootstrap_ci(num)
            lines.append(f"| {m} | {fmt(mean(num)) if num else '-'} | [{fmt(lo)}, {fmt(hi)}] | {', '.join(fmt(v) for v in vals)} |")
        lines.append("")

    # paired comparisons on shared seeds
    conds = sorted(data)
    if len(conds) >= 2:
        lines.append("## Paired comparisons (exact sign-flip test on shared seeds)")
        lines.append("")
        lines.append("| metric | A | B | n pairs | mean(A - B) | p |")
        for a, b in itertools.combinations(conds, 2):
            shared = sorted(set(data[a]) & set(data[b]))
            for m in KEY_METRICS.get(env, []):
                pairs = [(data[a][s].get(m), data[b][s].get(m)) for s in shared]
                pairs = [(x, y) for x, y in pairs if isinstance(x, (int, float)) and isinstance(y, (int, float))]
                if len(pairs) < 2:
                    continue
                xs, ys = zip(*pairs, strict=True)
                d = mean([x - y for x, y in pairs])
                lines.append(f"| {m} | {a} | {b} | {len(pairs)} | {d:+.3f} | {paired_sign_flip_p(xs, ys):.3f} |")
        lines.append("")
        lines.append("With k pairs the smallest attainable p is 2/2^k (k=5: 0.0625). A non-significant result "
                     "at this n is a statement about power, not evidence of no effect.")
    md = "\n".join(lines)
    OUT.mkdir(exist_ok=True)
    (OUT / f"{env}.md").write_text(md)
    plot(env, data)
    return md


def plot(env: str, data: dict[str, dict[int, dict]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ink, muted, grid = "#2b3440", "#6b7480", "#e3e6ea"
    conds = sorted(data)
    for m in KEY_METRICS.get(env, []):
        fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(conds), 3.2), dpi=150)
        any_point = False
        for i, c in enumerate(conds):
            vals = [r.get(m) for r in data[c].values()]
            vals = [float(v) for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if not vals:
                continue
            any_point = True
            jitter = [i + (k - (len(vals) - 1) / 2) * 0.06 for k in range(len(vals))]
            ax.scatter(jitter, vals, s=36, color=ink, alpha=0.75, zorder=3, linewidths=0)
            ax.hlines(mean(vals), i - 0.25, i + 0.25, color=ink, linewidth=2, zorder=4)
        if not any_point:
            plt.close(fig)
            continue
        ax.set_xticks(range(len(conds)), conds, fontsize=8, color=muted)
        ax.set_title(m.replace("_", " "), fontsize=10, color=ink, loc="left")
        ax.tick_params(axis="y", labelsize=8, colors=muted, length=0)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.set_xlim(-0.6, len(conds) - 0.4)
        fig.tight_layout()
        fig.savefig(OUT / f"{env}_{m}.png")
        plt.close(fig)


if __name__ == "__main__":
    envs = sys.argv[1:]
    if not envs and Path("runs").exists():
        envs = sorted(p.name for p in Path("runs").iterdir() if p.is_dir())
    for e in envs:
        print(summarize(e))
        print()
