"""Input-modality configuration — the ablation axis for the V vs V+A experiment.

A run is pinned to exactly one **cell**. Every cell is defined identically across model
families, which is what makes the numbers comparable:

    ``v``       query-attachment audio -> ASR text (for EVERY model, omni included);
                the video's audio track is DROPPED.
    ``v+asr``   query audio -> ASR text; the audio track -> a timestamped ASR transcript
                injected as text context (this is the historical ``asr_substitute`` behavior).
    ``va``      query audio fed as raw audio; the audio track fed as a raw waveform.
                Only models with ``"audio" in capabilities.modalities_in`` can run it.

Why ASR the *query* even in ``va``-capable models under ``v``: the spoken query carries the
instruction, so dropping it makes the sample unanswerable rather than audio-starved. Cell
``v`` is therefore a true "vision + text only" condition that is independent of model family,
and ``va`` is available only to audio-native models — which is exactly the claim under test.

Audio-dependent samples in cell ``v`` are still **run and scored** (tagged
``audio-dropped``), never skipped: the ablation is meaningless if the denominators differ
between cells.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class InputConfig:
    name: str                  # "v" | "v+asr" | "va"
    query_audio_native: bool   # feed the query attachment as raw audio (else ASR it to text)
    track_asr: bool            # inject an ASR transcript of the video's audio track as text
    track_native: bool         # feed the raw audio-track waveform

    @property
    def needs_audio_model(self) -> bool:
        """True when the cell can only be run by a model that consumes audio natively."""
        return self.query_audio_native or self.track_native

    @property
    def slug(self) -> str:
        """Filename-safe form of :attr:`name` (``v+asr`` -> ``v-asr``)."""
        return self.name.replace("+", "-")


V = InputConfig(name="v", query_audio_native=False, track_asr=False, track_native=False)
V_ASR = InputConfig(name="v+asr", query_audio_native=False, track_asr=True, track_native=False)
VA = InputConfig(name="va", query_audio_native=True, track_asr=False, track_native=True)

CELLS = {c.name: c for c in (V, V_ASR, VA)}

# Shell-friendly spellings plus the legacy ``modality.policy`` value this axis replaced.
_ALIASES = {
    "v": "v",
    "va": "va",
    "v+a": "va",
    "v+asr": "v+asr",
    "v-asr": "v+asr",
    "v_asr": "v+asr",
    "vasr": "v+asr",
    "asr_substitute": "v+asr",  # legacy configs/eval.yaml modality.policy
}

DEFAULT = "v+asr"  # matches the pre-ablation behavior, so existing runs reproduce


def resolve_input_config(spec: Union[str, InputConfig, None]) -> InputConfig:
    """Accept a cell name, a legacy policy string, or an :class:`InputConfig` (idempotent)."""
    if isinstance(spec, InputConfig):
        return spec
    key = (spec or DEFAULT).strip().lower()
    name = _ALIASES.get(key)
    if name is None:
        raise ValueError(
            f"unknown input config {spec!r}; choose from {sorted(CELLS)} "
            f"(aliases: {sorted(_ALIASES)})"
        )
    return CELLS[name]


def check_model_support(ic: InputConfig, modalities_in) -> None:
    """Raise when the model cannot supply what the cell demands.

    Called once at startup by the runner: failing loudly beats silently degrading ``va`` to
    ``v+asr`` and reporting it as a native-audio number.
    """
    if ic.needs_audio_model and "audio" not in set(modalities_in):
        raise ValueError(
            f"input cell {ic.name!r} needs native audio but the model only accepts "
            f"{sorted(modalities_in)}; run it with --input v or --input v+asr"
        )
