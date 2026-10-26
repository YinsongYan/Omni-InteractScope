# Evaluation protocol

## Collection clock and causal inputs

For Table 2, decisions occur at the request boundary, global integer media seconds thereafter,
and clip end. A query at 2.4 seconds in a 6.3-second clip yields `2.4, 3, 4, 5, 6, 6.3`.
The first decision cannot precede the first available frame. Table 3 uses the same clock
principle and full horizon; T8-A begins at stream start. No target-dependent local clock,
scheduled event control, or target-based suffix truncation is offered by the release CLI.

Prompted calls independently receive the standing request and a sampled prefix from clip
start, at nominal 1 fps and at most 32 frames. Long prefixes are resampled over the entire
available history. There is no feedback of previous responses. Scene audio is sliced at the
current decision. V+ASR exposes only completed transcript segments. Native sessions initialize
with the request and replay pre-query history; outputs from that replay do not enter scores.
Thereafter frames and successive audio chunks advance the same media clock.

Video sampling uses actual presentation timestamps and restores the container start offset.
Audio decoding retains its original timestamps and trims/pads to clip-relative zero before
causal slicing. Tiny audio inputs may be padded with silence inside the model processor;
padding cannot advance the observation clock or consume future media. No synthetic frames
are accepted during Table 2/3 inference.

## Table 2

Targets have point windows `[reference - 2s, reference + 2s]`. Neighboring targets partition
response-assignment neighborhoods at their temporal midpoints. After the two-second merge
rule, the first assigned response is decisive: early means premature; timely but wrong content
means content failure; no assigned response means a miss. Later corrections cannot rescue a
target. T1 requires exactly the reference set of tool names, with no extras or omissions.
Other content is judged by the pinned binary GPT-4o rubric selected by the task router.

Prompted responses use decision timestamps. Native responses use adapter onset timestamps;
outputs beginning during initialization are excluded. LiveCC groups raw chunks relative to
each group's first timestamp, with a two-second inclusive gap; it does not extend the group
relative to its latest chunk. Its continuation markers are removed, adjacent verbatim chunks
are collapsed, and group text is scored at the unchanged onset. Full inference ensures the
last scored group is not cut off by an execution shortcut.

Accuracy is successful targets / all targets per task. Avg. is the equal mean of six task
accuracies, not micro accuracy over targets or a mean over interactions. T5 is excluded.

## Table 3

T8-A uses its exact annotated valid interval. T8-B/C use point ±2-second windows with the lower
bound clipped to query time. The assistance rubric compares each temporally eligible response
with the specific reference target and user context. A generic warning is insufficient for a
specific target; a guidance step must convey the currently actionable reference information.
Equivalent wording is accepted; timing is judged separately.

Matching maximizes successful target coverage with order-preserving, one-to-one response use.
Each target and collected response can be used at most once. Earlier unrelated responses do
not exhaust a target. Native responses are timestamped when the collected content becomes
available. LiveCC uses its native continuation boundaries and completion timestamps; the
accepted scorer retains an unfinished final fragment as an explicitly incomplete collected
fragment. Table 2 onset processing must not be applied to Table 3.

Overall is pooled hits / 360. T8-A/B/C use 122/151/87 targets. Complete T8-C counts interactions
with every reference guidance step covered out of 26 sequences. No unmatched-output penalty
is applied. This measures reference-target recall, not precision, interruption cost or
closed-loop user-task success.

## T5 and failure accounting

The separately displayed T5 column retains the legacy query-relative integer-step rolling
schedule, prior media sampler, no pre-query replay, and the historical final-target scan bound.
It is distinct from the main Table 2 common clock. A native adapter is required. Its point
scorer still applies the same content and ±2-second timing criteria.

All inference failures remain explicit and are retried on resume. Missing predictions prevent
full scoring. Documented terminal failures can be included as zero-success records, retaining
all target denominators. Invalid/failed content-judge responses are never negative verdicts;
they stop scoring until a valid response can be obtained. Caches accept only consistent
binary verdicts from the pinned judge model.

All scores use media time. Network round trips, model generation time, queuing and hardware
latency are not included in these timestamps.
