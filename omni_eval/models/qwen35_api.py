"""Qwen3.5-Omni-Plus: shared hosted media/prompt recipe, strict SSE transport."""
import os
from pathlib import Path
from .openai_api import OpenAICompatModel
from ..clients.qwen35_stream import Qwen35StreamClient


class Qwen35OmniPlusModel(OpenAICompatModel):
    _display_name = 'qwen3.5-omni-plus'
    _api_model = 'qwen3.5-omni-plus'

    def __init__(self, rpm=54, rate_file='', telemetry_path='', **kwargs):
        kwargs.setdefault('api_base', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        kwargs.setdefault('api_key_env', 'QWEN_OMNI_API_KEY')
        kwargs.setdefault('audio_part_style', 'input_audio')
        super().__init__(**kwargs)
        cache_dir = Path(self.cache_path).parent if self.cache_path else Path('.cache/omni-interactscope/api')
        self.rpm = rpm
        self.rate_file = rate_file or str(cache_dir / 'qwen35.rate')
        self.telemetry_path = telemetry_path
        if self.audio_part_style != 'input_audio':
            raise ValueError('Qwen3.5 requires the verified input_audio data URI recipe')

    def _ensure_model(self):
        if self.mock or self._client is not None:
            return
        with self._client_lock:
            if self._client is not None:
                return
            key = (os.environ.get(self.api_key_env) or '').strip()
            if not key:
                raise RuntimeError('Set your DashScope API key in ' + self.api_key_env)
            self._client = Qwen35StreamClient(self.api_model, base_url=self.api_base, api_key=key,
                cache_path=self.cache_path, temperature=self.temperature,
                max_output_tokens=self.max_output_tokens, json_mode=False,
                max_retries=self.max_retries, timeout=self.timeout,
                rpm=self.rpm, rate_file=self.rate_file, telemetry_path=self.telemetry_path)

    def _audio_part(self, mp3):
        self._counts['audio_parts'] += 1
        return {'type': 'input_audio', 'input_audio': {
            'data': self._data_uri('audio/mpeg', mp3), 'format': 'mp3'}}
