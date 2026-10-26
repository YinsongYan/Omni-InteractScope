"""Audio initialization must survive initial/gapped visual-only duplex ticks."""
from types import SimpleNamespace

import numpy as np
import pytest

from omni_eval.models.minicpm_o import MiniCPMoModel


class Duplex:
    def __init__(self):
        self.audio_chunk_idx = 0
        self.calls = []
        self.generated_at = []
        self.fail = False

    def streaming_prefill(self, *, audio_waveform, **kwargs):
        self.calls.append((self.audio_chunk_idx, audio_waveform))
        self.audio_chunk_idx += 1  # pinned upstream also increments for vision
        return {"success": not self.fail, "reason": "audio not enough"}

    def streaming_generate(self, **kwargs):
        self.generated_at.append(self.audio_chunk_idx)
        return {"is_listen": True}

    def prepare(self, **kwargs):
        self.audio_chunk_idx = 0


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setenv("OMNI_HF_MOCK", "1")
    m = MiniCPMoModel()
    m._dup = Duplex()
    return m


def test_audio_encoder_starts_at_zero_after_visual_tick_and_gap(model):
    wav = np.ones(16000, dtype=np.float32)
    for chunk in (None, wav, wav, None, wav):
        model._step(object(), chunk)
    assert [i for i, a in model._dup.calls if a is not None] == [0, 1, 2]
    assert model._dup.generated_at == [1, 2, 3, 4, 5]
    assert model._audio_chunks == 3


def test_continuous_audio_retains_native_indices(model):
    wav = np.ones(16000, dtype=np.float32)
    for _ in range(3):
        model._step(object(), wav)
    assert [i for i, _ in model._dup.calls] == [0, 1, 2]
    assert all(a is wav for _, a in model._dup.calls)


def test_short_audio_preserves_samples_and_pads_only_missing_tail(model):
    wav = np.arange(15877, dtype=np.float32)
    original = wav.copy()
    model._step(object(), wav)
    fed = model._dup.calls[0][1]
    np.testing.assert_array_equal(fed[:len(wav)], original)
    assert len(fed) == 16000 and not np.any(fed[len(wav):])
    np.testing.assert_array_equal(wav, original)
    assert model.session_stats()["audio_padding_samples"] == 123


def test_prefill_failure_is_not_reported_as_model_silence(model):
    model._dup.fail = True
    with pytest.raises(RuntimeError, match="prefill failed"):
        model._step(object(), np.ones(16000))
    assert model._dup.generated_at == []
    assert model._audio_chunks == 0


def test_session_reset_clears_audio_clock(model, monkeypatch):
    model._step(object(), np.ones(15900))
    model.mock = False
    monkeypatch.setattr(model, "_ensure_duplex", lambda: model._dup)
    model.begin(SimpleNamespace(query_text=""))
    assert model._audio_chunks == model._audio_padded_samples == 0
    model._step(object(), None)
    model._step(object(), np.ones(16000))
    assert model._dup.calls[-1][0] == 0


def test_visual_only_units_keep_the_original_clock(model):
    for _ in range(3):
        model._step(object(), None)
    assert [i for i, _ in model._dup.calls] == [0, 1, 2]
    assert model._dup.generated_at == [1, 2, 3]
    assert model._audio_chunks == model._audio_padded_samples == 0
