"""ASR query substitution and scene transcript preparation with explicit cache misses."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..data.media import MediaSource

BACKENDS = ("qwen3-asr", "cache", "mock")
DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B-hf"
DEFAULT_ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B-hf"

# Annotation language code -> the language name Qwen3-ASR accepts. Forcing the language skips
# the model's own detection step (and its occasional Cantonese/Mandarin confusion); "multi" and
# anything unknown map to None = auto-detect.
_LANG_NAMES = {
    "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
    "yue": "Cantonese", "fr": "French", "de": "German", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian",
}

# The forced aligner covers fewer languages than the recogniser, and its Japanese/Korean word
# splitters need extra packages (nagisa / soynlp). Outside this set the timeline falls back to
# window stamps, recorded as ``stamps: "window"`` in the cache entry.
_ALIGNER_LANGS = {
    "Chinese", "English", "Cantonese", "French", "German",
    "Italian", "Portuguese", "Russian", "Spanish",
}

# Sentence boundaries for grouping aligned words into segments. The word splitter drops
# punctuation, so these are found in the *transcript*, between two aligned words.
_SENT_END = "。！？!?…；;\n"
# Repetition-collapse only (never for segmenting): '.' and ',' split English clauses, which is
# where Qwen3-ASR's loops repeat, but as a segment boundary '.' would also cut every decimal.
_CLAUSE_END = _SENT_END + ".,，、"
# Silence between two aligned words that counts as a pause the segmenter may cut at. Three
# aligner frames (80 ms each) — long enough not to fire between two syllables of one word.
_MIN_PAUSE_SEC = 0.24

# Energy-VAD constants. These define the transcript, so they are keyed into the cache (see
# ``_timeline_params``) rather than exposed as config knobs — there is no reason to tune them
# per run, and a silent change would leave stale entries that still looked valid.
_VAD = {
    "frame_sec": 0.03,        # RMS frame
    "min_silence_sec": 0.5,   # shorter gaps stay inside one speech span (they are just pauses)
    "min_speech_sec": 0.3,    # drop shorter blips: clicks, single-frame noise
    "pad_sec": 0.25,          # widen each span, so a soft onset/coda is not clipped off
    "floor_margin_db": 8.0,   # speech = this far above the track's own noise floor ...
    "peak_margin_db": 35.0,   # ... but never more than this far below its loud parts
    "min_range_db": 6.0,      # below this dynamic range the VAD abstains (see _speech_spans)
}


def language_name(code: str) -> Optional[str]:
    """Map an annotation ``language`` to a Qwen3-ASR language name (None = auto-detect).

    Only consulted when ``modality.asr_force_language`` is on. Forcing is off by default because
    an interaction's ``language`` describes the *interaction*, not every second of its audio track,
    and Qwen3-ASR obeys the constraint even when it does not fit: over the English background music
    in ``2逛街__inter_02`` (147-160 s), forced Chinese invented the dialogue ``苗苗说`` /
    ``你的爱怎么了？``, while auto-detect returned the actual lyrics. Fabricated dialogue in an
    audio-trigger benchmark is the worst possible failure — it reads exactly like a trigger.
    Auto-detect also lets the model report empty audio as empty instead of hallucinating filler.
    """
    return _LANG_NAMES.get((code or "").strip().lower())


def _collapse_repeats(text: str, min_reps: int = 3, max_period: int = 4) -> str:
    """Collapse a clause cycle repeated ``min_reps``+ times in a row down to one occurrence.

    Qwen3-ASR degenerates into a loop when a window holds little speech but plenty of other sound
    (music, machinery). Its own ``_detect_and_fix_repetitions`` catches some of these, but not the
    multi-clause cycles: a 38 s window over the background music in ``2逛街__inter_02`` returned
    ``Didn't even notice. No punches to. You wanna say so?`` about forty times over.

    Left unchecked that is not just noise in the text. The forced aligner has to place all forty
    copies somewhere, so it walks its timestamps off the end of the audio — that window produced
    segments stamped 272-297 s inside a 182.8 s span. Collapsing first fixes both.

    The result is a concatenation of substrings of ``text`` in order, so a cursor walk over it
    (:meth:`ASR._segments_from_words`) still lines up with the aligner's word list.
    """
    units, buf = [], ""
    for ch in text:
        buf += ch
        if ch in _CLAUSE_END:
            units.append(buf)
            buf = ""
    if buf:
        units.append(buf)

    out: List[str] = []
    i, n = 0, len(units)
    while i < n:
        for p in range(1, max_period + 1):
            if i + p * min_reps > n:
                continue
            cycle = [units[k].strip() for k in range(i, i + p)]
            j = i + p
            while j + p <= n and [units[k].strip() for k in range(j, j + p)] == cycle:
                j += p
            if (j - i) // p >= min_reps:
                out.extend(units[i:i + p])
                i = j
                break
        else:
            out.append(units[i])
            i += 1
    return "".join(out).strip()


def _speech_spans(track, sr: int = 16000, max_sec: float = 120.0) -> List[tuple]:
    """Energy VAD: the ``[(start, end)]`` speech regions of ``track``, in seconds from its start.

    This exists because of how forced alignment fails. The aligner distributes the recognised words
    over whatever it is given, so silence inside a window gets absorbed into the neighbouring words
    and their timestamps drift: on ``2逛街__inter_02`` a 120 s window stamped ``苗苗`` at 140.9 s for
    an utterance that a tight window places at 164.2 s. At ``temporal.tolerance_sec = 2.0`` a 24 s
    error is not a rounding issue — it leaks a trigger word half a minute early. A fixed grid only
    bounds that error by the window length; cutting *at* the silence removes its cause.

    Two further effects, both wanted: silent stretches are never transcribed at all (that is where
    the ``嗯。呜。`` hallucinations came from), and the recogniser runs on far less audio.

    Abstains — one span covering everything — when the track has too little dynamic range to
    separate speech from background, which is the honest answer for a clip that is loud throughout
    (machinery, traffic) rather than one that happens to be all speech.

    Measured over this release's 331 timeline spans (3.48 h): 2.42 h kept in 2262 windows at a 20 s
    cap, 16 spans abstained, 28 returned nothing at all. Those 28 are why the caller treats an empty
    result as "no opinion" and falls back to a grid rather than to an empty transcript.
    """
    import numpy as np

    n_frame = max(1, int(_VAD["frame_sec"] * sr))
    total = len(track) / sr
    n = len(track) // n_frame
    if n < 3:
        return [(0.0, total)]
    rms = np.sqrt((track[: n * n_frame].reshape(n, n_frame).astype("float64") ** 2).mean(axis=1))
    db = 20.0 * np.log10(rms + 1e-10)
    floor, peak = float(np.percentile(db, 20)), float(np.percentile(db, 99))
    if peak - floor < _VAD["min_range_db"]:
        return [(0.0, total)]
    voiced = db >= max(floor + _VAD["floor_margin_db"], peak - _VAD["peak_margin_db"])

    # Frame runs -> spans, bridging gaps that are mere pauses.
    spans: List[List[float]] = []
    for i, v in enumerate(voiced):
        if not v:
            continue
        t0, t1 = i * _VAD["frame_sec"], (i + 1) * _VAD["frame_sec"]
        if spans and t0 - spans[-1][1] <= _VAD["min_silence_sec"]:
            spans[-1][1] = t1
        else:
            spans.append([t0, t1])
    spans = [s for s in spans if s[1] - s[0] >= _VAD["min_speech_sec"]]
    if not spans:
        return []

    out: List[tuple] = []
    for s0, s1 in spans:
        s0, s1 = max(0.0, s0 - _VAD["pad_sec"]), min(total, s1 + _VAD["pad_sec"])
        if out and s0 <= out[-1][1]:  # padding may have made two spans touch
            out[-1] = (out[-1][0], s1)
        else:
            out.append((s0, s1))
    # A speech region longer than the window is split on the grid: continuous speech has no silence
    # to cut at, so here the window genuinely is the bound on both memory and alignment drift.
    capped: List[tuple] = []
    for s0, s1 in out:
        while s1 - s0 > max_sec:
            capped.append((s0, s0 + max_sec))
            s0 += max_sec
        capped.append((s0, s1))
    return capped


@dataclass
class TranscriptSegment:
    start_sec: float
    end_sec: float
    text: str


class ASRCacheMiss(RuntimeError):
    """Backend ``cache`` was asked for a transcript that has not been precomputed."""


class ASR:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        mock: Optional[bool] = None,
        device: str = "cpu",
        backend: str = "cache",
        window_sec: float = 120.0,
        context_chars: int = 0,
        cache_dir: str = "",
        timestamps: str = "aligner",
        aligner_model: str = DEFAULT_ALIGNER,
        segment_max_sec: float = 8.0,
        force_language: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        # Default ``cache``: the read-only side. A caller who forgets to pass a backend gets a loud
        # miss, never an unintended transcription run.
        self.backend = (backend or "cache").strip().lower()
        self.window_sec = float(window_sec)
        self.context_chars = int(context_chars)
        self.cache_dir = cache_dir
        self.timestamps = (timestamps or "aligner").strip().lower()
        self.aligner_model = aligner_model
        self.segment_max_sec = float(segment_max_sec)
        self.force_language = bool(force_language)
        if self.timestamps not in ("aligner", "window"):
            raise ValueError(f"modality.asr_timestamps must be 'aligner' or 'window', got {timestamps!r}")
        self._model = None
        self._processor = None
        self._aligner = None
        self._aligner_processor = None
        self.mock = mock
        if self.mock is None:
            self.mock = os.environ.get("OMNI_ASR_MOCK", "") == "1"
        if self.backend == "mock":
            self.mock = True
        if self.backend not in BACKENDS:
            raise ValueError(f"unknown modality.asr_backend {backend!r}; choose from {list(BACKENDS)}")
        # Counters, printed by the precompute script.
        # align_reject: windows whose forced alignment overran the audio and were downgraded to
        # window stamps. Worth watching in a precompute — a high count means asr_window_sec is too
        # large for this material.
        self.stats = {"cache_hit": 0, "computed": 0, "align_reject": 0}

    # ---- disk cache --------------------------------------------------------------
    def _params(self) -> Dict[str, Any]:
        """The transcriber's identity, shared by both job kinds.

        Deliberately **not** the backend name: ``cache`` has to land on the exact key the
        precompute wrote, and which of the two processes is asking is not a property of the
        transcript. The model name already separates ``base`` from ``Qwen/Qwen3-ASR-1.7B-hf``.
        """
        return {"model": self.model_name}

    def _timeline_params(self) -> Dict[str, Any]:
        """Extra key material for timeline jobs only.

        Kept out of :meth:`_params` so that retuning the segmentation does not invalidate the 171
        query-audio transcripts, which are whole-file and have neither windows nor timestamps.
        """
        p = {
            "window_sec": round(self.window_sec, 3),
            "context_chars": self.context_chars,
            "timestamps": self.timestamps,
        }
        if self.timestamps == "aligner":
            p["aligner"] = self.aligner_model
            p["segment_max_sec"] = round(self.segment_max_sec, 3)
            p["min_pause_sec"] = _MIN_PAUSE_SEC
            p["vad"] = dict(_VAD)  # the VAD chooses the windows, so it is part of the identity
        return p

    def _cache_key(self, kind: str, source: str, extra: Dict[str, Any]) -> str:
        # basename+size, not the absolute path: the cache survives a --videos override or a
        # move of the data root, and still changes when a clip is re-fetched at a new length.
        payload = {"kind": kind, "source": source, **self._params(), **extra}
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, key[:2], f"{key}.json")

    def _cache_read(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.cache_dir:
            return None
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None  # corrupt entry: recompute rather than crash

    def _cache_write(self, key: str, record: Dict[str, Any]) -> None:
        if not self.cache_dir:
            return
        path = self._cache_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
        os.replace(tmp, path)  # atomic: a killed precompute never leaves a half-written entry

    @staticmethod
    def _source_id(path: str) -> str:
        try:
            return f"{os.path.basename(path)}:{os.path.getsize(path)}"
        except OSError:
            return os.path.basename(path)

    def _miss(self, what: str) -> ASRCacheMiss:
        return ASRCacheMiss(
            f"modality.asr_backend='cache' but {what} is not in {self.cache_dir or '<no cache dir>'}.\n"
            f"  precompute it first:  source envs/qwen3_asr/env.sh && "
            f"omni-bench precompute-asr --split table2 --asr-cache <cache-dir>\n"
            f"  (this env cannot run Qwen3-ASR itself: it needs transformers>=5, the adapters need 4.5x)"
        )

    # ---- model loading -----------------------------------------------------------
    @property
    def _dtype(self):
        import torch

        return torch.bfloat16 if str(self.device).startswith("cuda") else torch.float32

    def _ensure_qwen3(self):
        """Load Qwen3-ASR. Only importable inside ``envs/qwen3_asr`` (transformers>=5)."""
        if self._model is None:
            from transformers import Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor

            self._processor = Qwen3ASRProcessor.from_pretrained(self.model_name)
            self._model = Qwen3ASRForConditionalGeneration.from_pretrained(
                self.model_name, dtype=self._dtype,
            ).to(self.device).eval()
        return self._model

    def _ensure_aligner(self):
        """Load the forced aligner (``Qwen3-ForcedAligner-0.6B-hf``), a token-classification head.

        Separate 0.6B checkpoint, one forward per window — no generation, so it is cheap next to
        the recogniser it timestamps.

        It needs **its own** processor, not the recogniser's: the two differ only in the chat
        template, but the aligner's is the one that joins the words with the ``<timestamp>`` pairs
        the head classifies. With the recogniser's template the input holds no ``<timestamp>`` at
        all, and ``decode_forced_alignment`` then dies on an empty mask.
        """
        if self._aligner is None:
            from transformers import Qwen3ASRForTokenClassification, Qwen3ASRProcessor

            self._aligner_processor = Qwen3ASRProcessor.from_pretrained(self.aligner_model)
            self._aligner = Qwen3ASRForTokenClassification.from_pretrained(
                self.aligner_model, dtype=self._dtype,
            ).to(self.device).eval()
        return self._aligner

    def _qwen3_transcribe(self, audio, language: Optional[str], prompt: str = "") -> str:
        """One Qwen3-ASR forward over a float32 16 kHz mono array. Greedy, like every adapter."""
        import numpy as np
        import torch

        model = self._ensure_qwen3()
        arr = np.asarray(audio, dtype="float32")
        if arr.size < 160:  # <10 ms: nothing to transcribe
            return ""
        # (device, dtype): the mel features come out of the feature extractor in float32 while the
        # weights are bf16, and BatchFeature.to casts only the floating-point entries — input_ids
        # stay integer.
        inputs = self._processor.apply_transcription_request(
            audio=[arr], language=language, prompt=prompt or None,
        ).to(self.device, self._dtype)
        n_in = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=False,
                # Budget by the audio's own length, not the configured window: the last window of a
                # span is usually short, and an oversized budget only buys a longer silence loop.
                max_new_tokens=max(48, int(arr.size / 16000 * 24)),
            )
        # ``parsed`` also runs the upstream repetition fix, which matters here: on a window that is
        # mostly silence Qwen3-ASR reliably degenerates into repeating one phrase. It only catches
        # single-clause loops, hence _collapse_repeats on top.
        parsed = self._processor.decode(out[0][n_in:], return_format="parsed")
        return _collapse_repeats((parsed.get("transcription") or "").strip())

    def _qwen3_align(self, audio, text: str, language: Optional[str]) -> List[Dict[str, Any]]:
        """Word-level timestamps for ``text`` over ``audio``: [{"text","start_time","end_time"}].

        80 ms resolution, relative to the start of ``audio``.
        """
        import numpy as np
        import torch

        aligner = self._ensure_aligner()
        proc = self._aligner_processor
        arr = np.asarray(audio, dtype="float32")
        if not text or arr.size < 160:
            return []
        inputs, word_lists = proc.prepare_forced_aligner_inputs(
            audio=[arr], transcript=[text], language=language,
        )
        if not word_lists or not word_lists[0]:
            return []
        inputs = inputs.to(self.device, self._dtype)
        with torch.no_grad():
            logits = aligner(**inputs).logits
        return proc.decode_forced_alignment(
            logits, inputs["input_ids"], word_lists, aligner.config.timestamp_token_id,
        )[0]

    def _segments_from_words(
        self, text: str, words: List[Dict[str, Any]], offset: float
    ) -> List[TranscriptSegment]:
        """Group aligned words into sentence-level segments carrying real speech times.

        The aligner's word list has punctuation and whitespace stripped, so the words are walked
        back onto ``text`` with a cursor: that recovers the original spelling *and* the punctuation
        between two words, which is where the sentence boundaries are. A segment's ``end_sec`` is
        its last word's end — when the speech stopped, not when the next one starts.
        """
        spans = []  # (char_start, char_end, start_time, end_time)
        cursor = 0
        for w in words:
            token = w.get("text") or ""
            i = text.find(token, cursor) if token else -1
            if i < 0:
                continue  # a token the recogniser's text does not contain: skip, keep the cursor
            spans.append((i, i + len(token), float(w["start_time"]), float(w["end_time"])))
            cursor = i + len(token)
        if not spans:
            return []

        segs: List[TranscriptSegment] = []
        buf: List[tuple] = []
        for k, sp in enumerate(spans):
            buf.append(sp)
            last = k + 1 == len(spans)
            nxt = spans[k + 1][0] if not last else len(text)
            gap = text[sp[1]:nxt]  # punctuation / spaces the word splitter dropped
            # The length cap only fires at an audible pause. Chinese "words" are single
            # characters, so an unconditional cap cuts inside one (张师傅烹饪咖 | 啡小课堂开始)
            # — the times stay right but the text is mangled. Waiting for a pause means a run of
            # genuinely continuous unpunctuated speech can exceed the cap, which is honest: a
            # sentence cannot be revealed before it ends.
            pause = (spans[k + 1][2] - sp[3]) if not last else 0.0
            over_len = buf[-1][3] - buf[0][2] >= self.segment_max_sec and pause >= _MIN_PAUSE_SEC
            if last or over_len or any(c in _SENT_END for c in gap):
                chunk = text[buf[0][0]:nxt].strip()
                if chunk:
                    segs.append(TranscriptSegment(offset + buf[0][2], offset + buf[-1][3], chunk))
                buf = []
        return segs

    # ---- public API --------------------------------------------------------------
    def transcribe_file(self, path: str, language: str = "") -> str:
        """Transcribe an audio-query attachment to text."""
        if not path or not os.path.exists(path):
            return ""
        if self.mock:
            return f"[asr-mock transcript of {os.path.basename(path)}]"

        lang = language_name(language) if self.force_language else None
        key = self._cache_key("file", self._source_id(path), {"language": lang})
        hit = self._cache_read(key)
        if hit is not None:
            self.stats["cache_hit"] += 1
            return hit.get("text", "")
        if self.backend == "cache":
            raise self._miss(f"the transcript of {os.path.basename(path)}")

        # A spoken query is one short utterance: transcribe it whole, no windowing.
        import numpy as np

        from ..data.media import decode_audio_file

        arr = decode_audio_file(path, sr=16000)
        if arr is None or not len(arr):
            return ""
        text = self._qwen3_transcribe(np.asarray(arr), lang)
        self.stats["computed"] += 1
        self._cache_write(key, {"kind": "file", "source": os.path.basename(path),
                                "language": lang, "text": text, **self._params()})
        return text

    def transcribe_timeline(
        self, media: MediaSource, start_sec: float, end_sec: float, language: str = ""
    ) -> List[TranscriptSegment]:
        """Transcribe the audio track over [start, end] into timestamped segments."""
        if not media.exists or end_sec <= start_sec:
            return []
        if self.mock:
            return [TranscriptSegment(start_sec, end_sec, "[asr-mock audio transcript]")]

        lang = language_name(language) if self.force_language else None
        window = {"start": round(start_sec, 2), "end": round(end_sec, 2), "language": lang}
        key = self._cache_key(
            "timeline", self._source_id(media.path), {**window, **self._timeline_params()}
        )
        hit = self._cache_read(key)
        if hit is not None:
            self.stats["cache_hit"] += 1
            return [
                TranscriptSegment(float(s["start"]), float(s["end"]), s["text"])
                for s in hit.get("segments", [])
            ]
        if self.backend == "cache":
            raise self._miss(
                f"the audio-track transcript of {os.path.basename(media.path)} "
                f"[{start_sec:.1f}s, {end_sec:.1f}s]"
            )

        segs = self._qwen3_timeline(media, start_sec, end_sec, lang)
        self.stats["computed"] += 1
        self._cache_write(key, {
            "kind": "timeline", "source": os.path.basename(media.path), **window,
            # ``stamps`` is audit trail, not key material: it records whether these times are real
            # word times or just window bounds, so a fallback cannot pass for an alignment.
            "stamps": "aligner" if self._use_aligner(lang) else "window",
            "segments": [{"start": s.start_sec, "end": s.end_sec, "text": s.text} for s in segs],
            **self._params(), **self._timeline_params(),
        })
        return segs

    def _use_aligner(self, language: Optional[str]) -> bool:
        """Whether this span gets real word timestamps.

        ``language=None`` (auto-detect / ``multi``) is allowed: the aligner's default word splitter
        is the CJK/whitespace one and needs no language. Only a language the aligner does not
        cover — or Japanese/Korean, whose splitters need extra packages — falls back.

        Accepts a code as well as a name. Not politeness: everything else here passes names, so a
        stray ``"zh"`` used to land in neither branch and silently downgrade the whole span to
        window stamps — a wrong-timestamps bug that looked like a working run.
        """
        if language is not None and language not in _ALIGNER_LANGS:
            language = language_name(language) or language
        return self.timestamps == "aligner" and (language is None or language in _ALIGNER_LANGS)

    def _qwen3_timeline(
        self, media: MediaSource, start_sec: float, end_sec: float, language: Optional[str]
    ) -> List[TranscriptSegment]:
        """Transcribe [start, end] window by window, then timestamp each window's text.

        Qwen3-ASR's own output carries no timestamps, so they come from one of two places:

        ``timestamps: aligner``
            Windows are the track's **speech regions** (:func:`_speech_spans`), and
            ``Qwen3-ForcedAligner-0.6B`` gives 80 ms word-level times inside each, grouped into
            sentences. ``window_sec`` is then only the cap on a single uninterrupted speech run.
        ``timestamps: window``
            A fixed ``window_sec`` grid, each window's text stamped with the window's own bounds.
            Since ``transcript_up_to`` reveals a segment only once it has *ended*, ``window_sec``
            becomes the transcript's lag — at ``temporal.tolerance_sec = 2.0`` even a 4 s window
            can hide an audio trigger. A fallback for languages the aligner does not cover.

        Both partitions are non-overlapping and derived from the span alone, so one cached
        transcript serves every probe on the grid.
        """
        import numpy as np

        # One ffmpeg call for the whole span, then slice: per-window `-ss` seeking would spawn
        # thousands of processes and cost more than the model forwards.
        track = media.audio_chunk(start_sec, end_sec - start_sec, sr=16000)
        if track is None or not len(track):
            return []
        track = np.asarray(track, dtype="float32")
        sr = 16000
        aligned = self._use_aligner(language)
        # Hard cap at 240 s: the forced aligner is documented for speech up to 5 minutes.
        step = max(0.5, min(self.window_sec, 240.0) if aligned else self.window_sec)
        windows = _speech_spans(track, sr, max_sec=step) if aligned else []
        if not windows:
            # Either the window-stamp mode, or the VAD found no sustained speech (4锻造__inter_02:
            # 4.5 s of hammer strikes). Do not let the VAD alone decide a span has nothing to say —
            # a false negative there deletes the whole transcript for that interaction, and the
            # recogniser is the better judge: on genuinely speechless audio it now returns empty
            # (auto-detect) instead of the filler that forced-language produced.
            edges = np.arange(0.0, end_sec - start_sec, step)
            windows = [(float(t), float(min(end_sec - start_sec, t + step))) for t in edges]

        segs: List[TranscriptSegment] = []
        context = ""
        for w0, w1 in windows:
            piece = track[int(w0 * sr): int(w1 * sr)]
            text = self._qwen3_transcribe(piece, language, context)
            if not text:
                continue
            words = self._qwen3_align(piece, text, language) if aligned else []
            # An alignment that runs past the audio it was given is not a rounding error, it means
            # the aligner lost the plot (typically on a transcript far longer than the window can
            # hold). Its times are then worthless everywhere in the window, not just at the tail, so
            # fall back to window bounds for the whole window rather than clamping the overflow —
            # clamping would keep the wrong times and hide the failure.
            if words and max(w["end_time"] for w in words) > (w1 - w0) + 1.0:
                self.stats["align_reject"] = self.stats.get("align_reject", 0) + 1
                words = []
            if words:
                segs.extend(self._segments_from_words(text, words, start_sec + w0))
            else:
                segs.append(TranscriptSegment(start_sec + w0, start_sec + w1, text))
            if self.context_chars > 0:
                context = (context + " " + text).strip()[-self.context_chars:]
        return segs


def format_transcript(segs: List[TranscriptSegment]) -> str:
    """Render segments as a compact timestamped block for text context."""
    if not segs:
        return ""
    lines = [f"[{s.start_sec:.1f}s] {s.text}" for s in segs if s.text]
    return "Audio track transcript:\n" + "\n".join(lines) if lines else ""
