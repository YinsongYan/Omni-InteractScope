"""Strict Qwen SSE transport; process-shared pacing includes retries and restarts.

No raw requests, credentials, media payloads or gateway error envelopes are logged.
SSE transport is not native proactive streaming: each call is an independent probe.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

from .openai_compat import OpenAIClient, OpenAIError
from ..utils.io import append_jsonl

WIRE_VERSION = 'qwen35-sse-text-input-audio-data-uri-v1'


def parse_sse(lines, expected_model='qwen3.5-omni-plus'):
    chunks, models, usage, finish, done = [], set(), {}, None, False
    data = []

    def event(value):
        nonlocal usage, finish, done
        if value == '[DONE]':
            done = True
            return
        if done:
            raise OpenAIError('SSE data after completion')
        obj = json.loads(value)
        if not isinstance(obj, dict) or obj.get('error'):
            raise OpenAIError('provider error envelope')
        model = obj.get('model')
        if model:
            if model != expected_model:
                raise OpenAIError('provider model identity mismatch')
            models.add(model)
        if obj.get('usage'):
            usage = obj['usage']
        choices = obj.get('choices')
        if choices is None or not isinstance(choices, list) or len(choices) > 1:
            raise OpenAIError('invalid SSE choices')
        for ch in choices:
            if ch.get('index', 0) != 0:
                raise OpenAIError('unexpected choice index')
            delta = ch.get('delta') or {}
            content = delta.get('content')
            if content is not None:
                if not isinstance(content, str) or (finish and content):
                    raise OpenAIError('invalid SSE content')
                chunks.append(content)
            reason = ch.get('finish_reason')
            if reason:
                if reason != 'stop' or finish:
                    raise OpenAIError('incomplete or duplicate SSE finish')
                finish = reason

    for raw in lines:
        line = raw.decode('utf-8').rstrip('\r\n') if isinstance(raw, bytes) else raw.rstrip('\r\n')
        if line == '':
            if data:
                event('\n'.join(data)); data = []
        elif line.startswith('data:'):
            data.append(line[5:].lstrip(' '))
        elif line.startswith((':', 'event:', 'id:', 'retry:')):
            continue
        else:
            raise OpenAIError('non-SSE response')
    if data:
        event('\n'.join(data))
    text = ''.join(chunks)
    if not done or finish != 'stop' or models != {expected_model} or not text.strip():
        raise OpenAIError('incomplete, empty or unidentified SSE response')
    if not usage or not isinstance(usage.get('prompt_tokens'), int):
        raise OpenAIError('missing SSE token accounting')
    return text, dict(finish=finish, usage=usage, model_version=expected_model)


class SharedPacer:
    """One timestamp protected by flock: all processes share the same rate budget."""
    def __init__(self, path, rpm):
        if not 0 < rpm <= 60:
            raise ValueError('RPM must be in (0, 60]')
        self.path, self.interval = Path(path), 60.0 / rpm
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def wait(self):
        while True:
            with self.path.open('a+') as fh:
                fcntl.flock(fh, fcntl.LOCK_EX)
                fh.seek(0)
                last = float(fh.read().strip() or '0')
                now = time.time()
                wait = max(0, last + self.interval - now)
                if wait == 0:
                    fh.seek(0); fh.truncate(); fh.write(str(now)); fh.flush()
                    return now
            time.sleep(min(wait, 2.0))


class Qwen35StreamClient(OpenAIClient):
    def __init__(self, *args, rpm=54, rate_file='', telemetry_path='', **kwargs):
        super().__init__(*args, **kwargs)
        if self.json_mode or not rate_file:
            raise ValueError('text output and a shared rate file are required')
        self.pacer = SharedPacer(rate_file, rpm)
        self.telemetry_path = telemetry_path

    def _key(self, prompt, parts=None):
        key = super()._key(prompt, parts)
        return hashlib.sha256((self.base_url + '\x1f' + WIRE_VERSION + '\x1f' + key).encode()).hexdigest()

    def _post(self, prompt, parts=None):
        content = list(parts) + [{'type': 'text', 'text': prompt}] if parts else prompt
        body = dict(model=self.model, messages=[dict(role='user', content=content)],
                    temperature=self.temperature, max_tokens=self.max_output_tokens,
                    stream=True, modalities=['text'], stream_options=dict(include_usage=True))
        req = urllib.request.Request(self.base_url + '/chat/completions',
            data=json.dumps(body, ensure_ascii=False).encode(), method='POST',
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.api_key})
        started = self.pacer.wait()
        record = dict(started_at_unix=started, cache_key=self._key(prompt, parts),
                      wire_version=WIRE_VERSION, endpoint=self.base_url,
                      n_images=sum(p.get('type') == 'image_url' for p in parts or []),
                      n_audio=sum(p.get('type') == 'input_audio' for p in parts or []))
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                record['http_status'] = response.status
                if response.status != 200 or 'text/event-stream' not in response.headers.get('Content-Type', ''):
                    raise OpenAIError('unexpected HTTP response type')
                answer, meta = parse_sse(response)
            record.update(status='success', **meta)
            return answer, meta
        except Exception as exc:
            # Never stringify an upstream exception: gateways may echo request headers.
            record.update(status='error', error_type=type(exc).__name__)
            if isinstance(exc, urllib.error.HTTPError):
                record['http_status'] = exc.code
            raise OpenAIError('Qwen SSE request failed: ' + record['error_type']
                              + (' HTTP ' + str(record['http_status']) if record.get('http_status') else '')) from None
        finally:
            record['elapsed_sec'] = time.time() - started
            if self.telemetry_path:
                append_jsonl(self.telemetry_path, record)
