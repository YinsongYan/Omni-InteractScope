# Dataset format

The dataset has **one annotation file**, `annotations.json`, containing all 1,479
interactions. An interaction is one record in its `interactions` list, with one or more
reference `responses`. Evaluation splits select records from this shared annotation file.

The default dataset root is `data/Omni-InteractScope/`. The evaluation code lives
in this GitHub repository; the hosted dataset lives in the private
[Hugging Face dataset repository](https://huggingface.co/datasets/Darden7/Omni-InteractScope)
under that directory. Follow the [download instructions](../README.md#get-the-code-and-data)
to obtain annotations, query attachments, ASR caches, verification manifests, and
207 self-recorded videos. The 905 platform-sourced videos are not hosted; obtain
them using the source links in `annotations.json`, as described below.

```text
Omni-InteractScope/
  annotations.json           # the sole annotation file (1,479 records, 1,790 targets)
  videos/                    # 1,112 videos, including their original scene audio
  query_attachments/audio/   # 131 query audio assets
  query_attachments/images/  # 25 query image assets
  asr_cache/table2/          # fixed historical ASR inputs, also used by T5
  asr_cache/table3/          # fixed assistance ASR inputs
  media_metadata.json       # media timing metadata needed by the legacy T5 protocol
  external_public_videos.jsonl # file manifest for the 905 platform-sourced videos
  LICENSE.md
  README.md
  file_manifest.jsonl        # file paths, byte sizes and SHA-256 values
  SHA256SUMS
```

## Video access

For platform-sourced videos, read the following fields from each record in
`annotations.json` → `interactions`:

| Field | How to use it |
|---|---|
| `source_attribution.source` | Source platform: `douyin`, `xiaohongshu`, or `bilibili` |
| `source_attribution.original_video_url` | Open this source link to obtain and download the original video |
| `source_start_ms`, `source_end_ms` | Start and end of the evaluation clip in the source video, in milliseconds |
| `video` | Save the prepared clip at this path relative to `data/Omni-InteractScope/` |

The following command exports the links and clip boundaries to a TSV file, with one row
per evaluation video. Run it from the repository root:

```bash
python - <<'PY' > video_sources.tsv
import csv
import json
import sys
from pathlib import Path

records = json.loads(Path('data/Omni-InteractScope/annotations.json').read_text())['interactions']
writer = csv.writer(sys.stdout, delimiter='\t')
writer.writerow(['video', 'source', 'original_video_url', 'source_start_ms', 'source_end_ms'])
seen = set()
for record in records:
    source = record['source_attribution']
    if source['source'] not in {'douyin', 'xiaohongshu', 'bilibili'} or record['video'] in seen:
        continue
    url = source['original_video_url'].strip()
    if url.startswith('ttps://'):  # Repair a missing 'h' in recorded source links.
        url = 'h' + url
    writer.writerow([record['video'], source['source'], url,
                     record['source_start_ms'], record['source_end_ms']])
    seen.add(record['video'])
PY
```

The links identify source-platform pages and may require platform-specific download tools.
This command exports a download list; it does not download media. Prepare each clip using
the listed start/end times, retain its audio, and use the exact relative filename in `video`.
Annotation query and response times already refer to the resulting clip; do not subtract
the source start time from them again.

For reproducible evaluation, the prepared video files must match the byte sizes and hashes
in `file_manifest.jsonl`. A fresh platform download or a re-encoded clip may differ from the
reference file even when its content appears identical. `omni-bench verify` checks file
sizes by default; `--checksums` also checks media hashes. ASR cache keys include video
filenames and sizes, so matching filenames alone is insufficient. The source-link list
provides access to the source content; it does not guarantee byte-identical downloads.

If you already have video files matching the manifest, import them with:

```bash
python scripts/materialize_public_videos.py \
  --source /path/to/video-archive \
  --dataset data/Omni-InteractScope --checksums
```

`--source` accepts a dataset root or its `videos/` directory. The script imports files listed
in `external_public_videos.jsonl` and preserves existing destination files. It uses hard links
on the same filesystem and copies across filesystems; add `--copy` for independent copies
or `--dry-run` to check the plan. Treat hard-linked media as read-only. All media must resolve
to paths inside the dataset root; symlinks to files outside it are not supported.

## Annotation schema

The document has `schema_version: "omni-interactscope-1.1"` and an `interactions` list.
Each record also carries that schema version. Essential fields are:

| Field | Meaning |
|---|---|
| `interaction_id` | Stable interaction identifier |
| `task` | Paper task group, T1 through T8 |
| `task_type`, `sub_task_type` | Historical task/subtask labels used by evaluation routing |
| `evaluation_splits` | `["table2"]`, `["table3"]`, `["t5"]`, or `[]` for an excluded record |
| `evaluation_exclusion` | Reason for exclusion, present on the two excluded records |
| `video`, `query_audio`, `query_image` | Asset paths relative to the dataset root; absent attachments use `""` |
| `query_type`, `query_text`, `query_time_ms` | Request modality, content and clip-relative query time |
| `responses` | Reference answers with trigger times/windows, response types and trigger reasons |
| `language`, `qa_language`, `target_group` | Scene language, query/answer language and assistance user context |
| `source_attribution` | Public source information retained within the same record |
| `source_attribution.source`, `source_attribution.original_video_url` | Source platform and link for obtaining the original video |
| `source_start_ms`, `source_end_ms` | Evaluation clip boundaries in the source video, in milliseconds |

`omni_eval.dataset.load_split(root, name)` reads this one file and selects records using
`evaluation_splits`. `all` means every record, including the two exclusions; it is a loader
option, not an `all.json` file. Each scoring population's count and ordered IDs are checked
against fixed release constants. The CLI still uses `--split table2`, `--split table3` or
`--split t5`; those names select evaluation protocols and do not refer to separate files.

| Paper group | Historical `task_type` / `sub_task_type` | Full interactions |
|---|---|---:|
| T1 Tool invocation | `simultaneous_tool_calls_search_ui`, `simultaneous_tool_calls_search_generative_ui` | 200 |
| T2 Event notification | `proactive_notifying`; `visual_hints` B/F | 207 |
| T3 Risk/error interjection | `verbal_visual_interjection` | 208 |
| T4 Cumulative-state tracking | `state_dependent` | 206 |
| T5 Temporal conditions | `time_awareness` | 115 |
| T6 Nonverbal/workflow cues | `visual_hints` A/C/D/E; `proactive_agent_assistance` | 37 |
| T7 Continuous responses | `simultaneous_speech` | 207 |
| T8 Assistance | `impairment_assistance` A/B/C | 299 |

T8-A/B/C are stored as `task: "T8"` with `sub_task_type: "A"`, `"B"` or `"C"`.
They contain 122/151/26 interactions and 122/151/87 targets, respectively. A means
query-free reminders, B audiovisual question answering, and C guided interaction.
Subtask letters belong to their historical task type; an A in one type is not an A in another.

The fixed scoring populations are:

- Table 2: 1,063 interactions and 1,313 targets. T1/T2/T3/T4/T6/T7 contain
  200/206/207/206/37/207 interactions and 200/206/207/206/37/457 targets.
- Table 3: 299 interactions and 360 targets.
- T5: 115 interactions and targets, scored using the separate legacy protocol.
- Full dataset: 1,479 interactions and 1,790 targets. One T2 and one T3 interaction
  remain available but are excluded from scoring because of invalid media/annotation timing.

`query_time_ms`, `notify_trigger_time_ms`, `valid_response_window_ms` and
`long_memory_start_ms` use clip-relative milliseconds. The loader converts them once to
seconds. Missing temporal values are `null`. `source_start_ms`/`source_end_ms` describe
only the source crop and must not be subtracted from query/response timestamps. References,
trigger reasons, target times and source titles are not model inputs.

`verify` checks population membership/order, duplicate IDs, target counts, file sizes,
annotation hashes, asset references and required T5 media metadata. `--checksums` also
hashes all media. ASR caches use the released asset filenames and their current file sizes; missing
required transcripts raise errors instead of falling back to mock/annotation transcripts.

Use matching code and data versions. Inference records the digest of `annotations.json`
with each run; use a new output directory when changing the dataset or model configuration.
