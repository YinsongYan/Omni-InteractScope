# Leaderboard

Results for 12 model configurations: **1,479 interactions, 1,790 targets, and 1,112 video records**. Two interactions with invalid media or timing are excluded from scoring, leaving 1,477 interactions and 1,788 targets across the three evaluation splits.

Scores use the original predictions and judgments for the remaining interactions; [full result counts](docs/public_results.json) are available by task.
Use the [evaluation commands](docs/models.md) to measure a model on T1–T8.

## Table 2: explicit-request tasks (T1–T7)

Joint content–timing accuracy (%), higher is better. **Avg.** equally weights T1–T4, T6 and T7.

| Model | Size | Input | T1 | T2 | T3 | T4 | T5* | T6 | T7 | Avg. |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemini-3.1-Pro | N/R | V, A | 22.5 | 28.2 | 11.6 | **11.2** | N/A | **27.0** | 11.2 | 18.6 |
| Gemini-3.5-Flash | N/R | V, A | 29.5 | **31.6** | **14.0** | 10.2 | N/A | 24.3 | **11.4** | **20.2** |
| Qwen3.5-Omni-Plus | N/R | V, A | **32.0** | 24.3 | 4.8 | 3.9 | N/A | 10.8 | 7.9 | 13.9 |
| InternVL3.5 | 8B | V+ASR | 29.0 | 1.0 | 0.0 | 0.0 | N/A | 5.4 | 0.4 | 6.0 |
| Qwen3-VL | 8B | V+ASR | 19.5 | 4.9 | 0.5 | 0.0 | N/A | 2.7 | 3.1 | 5.1 |
| Qwen2.5-Omni | 7B | V, A | 11.0 | 0.5 | 0.5 | 0.5 | N/A | 0.0 | 1.5 | 2.3 |
| Qwen3-Omni | 30B-A3B | V, A | 26.5 | 2.9 | 1.4 | 1.0 | N/A | 2.7 | 5.7 | 6.7 |
| LiveCC | 7B | V | 0.0 | 1.5 | 0.5 | 0.0 | 0.0 | 0.0 | 0.7 | 0.4 |
| AURA | 8B | V | 0.0 | 4.9 | 0.5 | 1.5 | 0.0 | 2.7 | 0.7 | 1.7 |
| MMDuet2 | 3B | V | 0.0 | 3.4 | 1.0 | 0.5 | 0.0 | 2.7 | 1.8 | 1.6 |
| MOSS-VL | 11.3B | V | 1.5 | 9.2 | 1.9 | 5.8 | 0.9 | 8.1 | 5.9 | 5.4 |
| MiniCPM-o 4.5 | 9B | V, A | 1.5 | 8.3 | 1.4 | 1.0 | 1.7 | 2.7 | 1.1 | 2.7 |

T1: tool invocation; T2: event notification; T3: risk/error interjection; T4: cumulative-state
tracking; T5: temporal conditions; T6: nonverbal/workflow cues; T7: continuous responses.
There are 200/206/207/206/115/37/457 targets for T1–T7, respectively.

*T5 uses the earlier query-relative rolling evaluation and is reported separately, excluded
from every Avg. N/A denotes prompted inputs without an explicit elapsed-time clock.
T1–T4, T6 and T7 use 1,063 interactions and 1,313 targets.

## Table 3: assistance tasks (T8)

Timely reference-target recall (%), higher is better. Overall pools hits over all 360 targets.
T8-A is query-free reminders (122 targets), T8-B audiovisual QA (151), and T8-C guidance (87).
Complete T8-C counts fully covered guidance interactions out of 26.

| Model | Size | Input | Overall | T8-A | T8-B | T8-C | Complete T8-C |
|---|---|---|---:|---:|---:|---:|---:|
| Gemini-3.1-Pro | N/R | V, A | **46.39** | **22.95** | **62.25** | **51.72** | **3/26** |
| Gemini-3.5-Flash | N/R | V, A | 41.11 | 22.13 | 57.62 | 39.08 | 2/26 |
| Qwen3.5-Omni-Plus | N/R | V, A | 31.11 | 9.02 | 49.67 | 29.89 | 1/26 |
| InternVL3.5 | 8B | V+ASR | 14.17 | 0.00 | 26.49 | 12.64 | 0/26 |
| Qwen3-VL | 8B | V+ASR | 15.00 | 2.46 | 24.50 | 16.09 | 0/26 |
| Qwen2.5-Omni | 7B | V, A | 10.00 | 2.46 | 19.87 | 3.45 | 0/26 |
| Qwen3-Omni | 30B-A3B | V, A | 13.06 | 2.46 | 25.83 | 5.75 | 0/26 |
| LiveCC | 7B | V | 3.61 | 0.00 | 8.61 | 0.00 | 0/26 |
| AURA | 8B | V | 6.94 | 0.00 | 15.89 | 1.15 | 0/26 |
| MMDuet2 | 3B | V | 1.94 | 0.00 | 4.64 | 0.00 | 0/26 |
| MOSS-VL | 11.3B | V | 3.33 | 0.00 | 7.28 | 1.15 | 0/26 |
| MiniCPM-o 4.5 | 9B | V, A | 3.06 | 0.00 | 7.28 | 0.00 | 0/26 |

V: video; A: native audio; ASR: scene transcripts; N/R: undisclosed model size;
A3B: active parameters. The first seven models use independent prompted calls; the final
five use rolling evaluation. Unmatched T8 responses are not penalized by recall.
See [scoring protocol](docs/protocol.md) for response timing and matching rules.
