"""Choose native audio, cached scene ASR, or video input according to declared model capabilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from ..schema import Interaction
from .input_config import InputConfig, resolve_input_config

# tags
NATIVE = "native"                 # sample needs audio, cell feeds it natively (va)
ASR_SUBSTITUTED = "asr-substituted"  # sample needs audio, cell feeds an ASR transcript (v+asr)
AUDIO_DROPPED = "audio-dropped"   # sample needs audio, cell withholds it (v) — scored anyway
UNSUPPORTED = "unsupported"       # model cannot supply the cell at all (guarded at startup)
NA = "n/a"                        # sample does not involve audio


@dataclass
class ModalityDecision:
    tag: str            # native | asr-substituted | audio-dropped | unsupported | n/a
    use_asr: bool       # feed ASR-derived text context for the audio track
    supported: bool     # False -> skip the sample entirely (never true for audio-dropped)
    input_config: str = ""  # the run's cell name, echoed into the prediction record

    @property
    def audio_starved(self) -> bool:
        """Sample needs audio but this cell withholds it — the honest caveat for cell ``v``."""
        return self.tag == AUDIO_DROPPED


def decide_adaptation(
    it: Interaction,
    modalities_in,
    input_config: Union[str, InputConfig, None] = None,
) -> ModalityDecision:
    """``modalities_in`` is any container of modality strings (a model's Capabilities).

    ``input_config`` accepts a cell name, an :class:`InputConfig`, or the legacy
    ``"asr_substitute"`` policy string (which maps to cell ``v+asr``).
    """
    ic = resolve_input_config(input_config)
    has_audio = "audio" in set(modalities_in)
    if ic.needs_audio_model and not has_audio:
        # Guarded at startup by check_model_support; defensive so a mis-wired caller cannot
        # silently produce numbers labelled "native audio" from a vision-only model.
        return ModalityDecision(UNSUPPORTED, use_asr=False, supported=False, input_config=ic.name)
    if not it.needs_audio:
        return ModalityDecision(NA, use_asr=False, supported=True, input_config=ic.name)
    if ic.track_native:
        return ModalityDecision(NATIVE, use_asr=False, supported=True, input_config=ic.name)
    if ic.track_asr:
        return ModalityDecision(ASR_SUBSTITUTED, use_asr=True, supported=True, input_config=ic.name)
    return ModalityDecision(AUDIO_DROPPED, use_asr=False, supported=True, input_config=ic.name)
