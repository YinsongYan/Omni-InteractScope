"""Official API configuration and client startup, without network requests."""
import io
import json
from pathlib import Path
import urllib.error

import pytest
import yaml

from omni_eval.cli import expand_env
from omni_eval.clients.gemini import GeminiClient, GeminiError
from omni_eval.clients.openai_compat import OpenAIClient, OpenAIError
from omni_eval.models.qwen35_api import Qwen35OmniPlusModel
from omni_eval.models.registry import build_model


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name,key_env,endpoint', [
    ('gemini-3.1-pro', 'GEMINI_API_KEY', 'https://generativelanguage.googleapis.com/v1beta/models'),
    ('gemini-3.5-flash', 'GEMINI_API_KEY', 'https://generativelanguage.googleapis.com/v1beta/models'),
    ('qwen3.5-omni-plus', 'QWEN_OMNI_API_KEY', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
])
def test_public_config_initializes_with_reader_key_and_official_endpoint(name, key_env, endpoint,
                                                                       monkeypatch, tmp_path):
    for variable in ('GEMINI_BASE_URL', 'QWEN_OMNI_BASE_URL', 'OPENAI_BASE_URL'):
        monkeypatch.setenv(variable, 'https://unused.example.invalid')
    monkeypatch.setenv(key_env, 'reader-key-placeholder')
    monkeypatch.setenv('OPENAI_API_KEY', 'judge-key-placeholder')
    config = yaml.safe_load((ROOT / f'configs/models/{name}.yaml').read_text())
    model = build_model(name, **expand_env(config['model_args']), cache_path=str(tmp_path / 'api/predictions.jsonl'))
    model._ensure_model()
    assert model._client.base_url == endpoint
    assert model._client.api_key == 'reader-key-placeholder'
    if name == 'qwen3.5-omni-plus':
        assert model._client.model == 'qwen3.5-omni-plus'
        assert model._client.pacer.path == tmp_path / 'api/qwen35.rate'
        assert model._client.pacer.path.parent.is_dir()


def test_qwen_requires_its_own_key_without_using_judge_key(monkeypatch, tmp_path):
    monkeypatch.delenv('QWEN_OMNI_API_KEY', raising=False)
    monkeypatch.setenv('OPENAI_API_KEY', 'judge-key-placeholder')
    model = Qwen35OmniPlusModel(cache_path=str(tmp_path / 'predictions.jsonl'))
    with pytest.raises(RuntimeError, match='QWEN_OMNI_API_KEY'):
        model._ensure_model()


def test_qwen_accepts_standard_dashscope_sse(monkeypatch, tmp_path):
    monkeypatch.setenv('QWEN_OMNI_API_KEY', 'reader-key-placeholder')
    model = Qwen35OmniPlusModel(cache_path=str(tmp_path / 'predictions.jsonl'))
    model._ensure_model()
    monkeypatch.setattr(model._client.pacer, 'wait', lambda: 0)
    events = [
        {'model': 'qwen3.5-omni-plus', 'choices': [{'index': 0, 'delta': {'content': 'ABSTAIN'}, 'finish_reason': None}]},
        {'model': 'qwen3.5-omni-plus', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]},
        {'model': 'qwen3.5-omni-plus', 'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}},
    ]
    response = io.BytesIO((''.join('data: ' + json.dumps(e) + '\n\n' for e in events)
                           + 'data: [DONE]\n\n').encode())
    response.status = 200
    response.headers = {'Content-Type': 'text/event-stream'}
    def request(req, **kwargs):
        assert req.full_url == 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
        assert req.get_header('Authorization') == 'Bearer reader-key-placeholder'
        return response
    monkeypatch.setattr('urllib.request.urlopen', request)
    assert model._client.generate('Observe the scene.') == 'ABSTAIN'


@pytest.mark.parametrize('client_type,error_type', [(GeminiClient, GeminiError), (OpenAIClient, OpenAIError)])
def test_provider_errors_do_not_echo_credentials_or_urls(client_type, error_type, monkeypatch):
    client = client_type('example-model', api_key='reader-key-placeholder', max_retries=1)
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError('https://unused.example.invalid', 401,
                                     'echoed-reader-key-placeholder', None, None)
    monkeypatch.setattr(client, '_post', fail)
    with pytest.raises(error_type) as exc:
        client.generate('Observe the scene.')
    assert 'HTTP 401' in str(exc.value)
    assert 'reader-key-placeholder' not in str(exc.value)
    assert 'example.invalid' not in str(exc.value)
