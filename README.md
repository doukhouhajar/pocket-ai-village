# pocket-ai-village

*A small society of LLM agents, built to ask three questions under controlled conditions: do agents cooperate when acting alone pays more, do they report their own work honestly and do they know when to ask instead of act?*

This is a pocket-sized version of the [AI Village](https://theaidigest.org/village) with the same loop shape. The harness runs on a laptop, and the whole study (30 runs) took under an hour and about 1.4M tokens. Each experiment changes one thing, holds everything else fixed, and replays the same five seeds in both conditions.

The goal is to look into **the gap between what an agent says and what the log shows it did.**

## What I found, if you got some coffee

**Disclaimer.** Everything here comes from one 7B model with five seeds per condition. These are not general claims about "LLM agents". (À prendre avec des pincettes)

- With a group chat, 3 of 5 runs survived the 12-month season. Without one, all 5 collapsed by month 3. But the survivors settled on 5 tons each, every month, which is half of what the lake could sustain. One chat run collapsed without anyone breaking any promise.
- Every "done" report cited work that really happened and really was the agent's own. The work just wasn't correct. Telling agents they would be audited barely changed this (0.40 → 0.38).
- Agents ask about instructions that 'look' ambiguous, not the ones that 'are'. They tried to ask about "the client's currency" in 14 of 14 episodes and about "remove the duplicate customers" in 5 of 18. Once the shared question budget ran out, 2 of 9 cleanup episodes deleted files the client needed. 

