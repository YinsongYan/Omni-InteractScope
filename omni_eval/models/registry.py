"""Lazy registry for the twelve evaluated model configurations and custom chat endpoints."""
from __future__ import annotations

import importlib
from typing import Callable, Dict, List

from .base import ProbeModel

_REGISTRY: Dict[str, str] = {'aura': 'omni_eval.models.aura_vllm:AURAModel', 'qwen3.5-omni-plus': 'omni_eval.models.qwen35_api:Qwen35OmniPlusModel', 'openai-compatible': 'omni_eval.models.openai_api:OpenAICompatModel', 'qwen3-vl-8b': 'omni_eval.models.qwen_vl:Qwen3VLModel', 'internvl3.5-8b': 'omni_eval.models.internvl:InternVL35Model', 'qwen2.5-omni-7b': 'omni_eval.models.qwen_omni:Qwen25OmniModel', 'qwen3-omni-30b-a3b': 'omni_eval.models.qwen_omni:Qwen3OmniModel', 'gemini-3.1-pro': 'omni_eval.models.gemini_api:Gemini31ProModel', 'gemini-3.5-flash': 'omni_eval.models.gemini_api:Gemini35FlashModel', 'mmduet2': 'omni_eval.models.mmduet2:MMDuet2Model', 'minicpm-o': 'omni_eval.models.minicpm_o:MiniCPMoModel', 'moss-vl-realtime': 'omni_eval.models.moss_vl_realtime:MossVLRealtimeModel', 'livecc-7b': 'omni_eval.models.livecc:LiveCCModel'}


def register(name: str, target: str) -> None:
    _REGISTRY[name] = target


def available() -> List[str]:
    return sorted(_REGISTRY)


def target_of(name: str) -> str:
    """``"module.path:ClassName"`` for one model, WITHOUT importing the adapter.

    Useful for inspecting adapters without importing their optional model dependencies.
    """
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {available()}")
    return _REGISTRY[name]


def build_model(name: str, **kwargs) -> ProbeModel:
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {available()}")
    module_path, _, cls_name = _REGISTRY[name].partition(":")
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    return cls(**kwargs)
