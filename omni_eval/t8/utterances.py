"""LiveCC's own continuation boundaries, credited when text becomes available."""
import re

CONTINUATION = re.compile(r'(?:\.\.\.|…)\s*$')


def scoring_emits(record):
    """Preserve raw inference; join LiveCC chunks at EOS or stream end for scoring.

    No GT-dependent merge, invented onset credit, or cross-utterance concatenation.
    An unfinished final fragment is retained and explicitly marked incomplete.
    """
    raw = sorted(record.get('emits', []), key=lambda e: e['time_sec'])
    if record.get('model') != 'livecc-7b':
        return raw
    output, pending = [], []

    def flush(complete):
        if not pending:
            return
        text = ' '.join(CONTINUATION.sub('', e['reply_text']).strip() for e in pending).strip()
        if text:
            output.append(dict(time_sec=pending[-1]['time_sec'], reply_text=text,
                response_type='text', onset_sec=pending[0]['time_sec'], n_chunks=len(pending),
                utterance_complete=complete, raw_emit_indices=[e['_index'] for e in pending]))
        pending.clear()

    for idx, emit in enumerate(raw):
        pending.append(dict(emit, _index=idx))
        if not CONTINUATION.search(emit['reply_text']):
            flush(True)
    flush(False)
    return output
