from types import SimpleNamespace
import numpy as np
import pytest
from omni_eval.schema import Interaction, Response
from omni_eval.models.base import Capabilities, Emit, ProbeResult
from omni_eval.data.media import Frame
from omni_eval.adaptation.modality import decide_adaptation
from omni_eval.protocols.base import PreparedInputs
from omni_eval.protocols.time_aligned import plan, run
from omni_eval.t8.protocol import clock_grid
from omni_eval.t8.scoring import score, ordered_matching
from omni_eval.evaluation import table2_record, validate_records
from omni_eval.dataset import from_release, local_path


def interaction():
    return Interaction('example', query_text='Notify me when the light turns green.', query_time=2400,
                       responses=[Response(1, response_type='text', notify_trigger_time=5000)])


def test_target_independent_clock_and_no_suffix_cutoff():
    it = interaction()
    assert plan(it, 6.3)[0] == [2.4, 3, 4, 5, 6, 6.3]
    it.responses[0].notify_trigger_time = 100000
    assert plan(it, 6.3)[0] == [2.4, 3, 4, 5, 6, 6.3]
    with pytest.raises(ValueError):
        plan(it, 6.3, cutoff=True)
    with pytest.raises(ValueError):
        plan(it, 6.3, arm='local')


class Media:
    path = 'anonymous.mp4'
    duration = 6.3
    pts = [0.0]
    def frame_at(self, t):
        return Frame(time_sec=t, image=np.zeros((4,4,3), dtype=np.uint8), synthetic=False)
    def sample_frames(self, start, end, **kwargs):
        return [self.frame_at(start), self.frame_at(end)]
    def audio_chunk(self, start, dur):
        self.intervals.append((start, start + dur))
        return np.zeros(int(dur * 16000), dtype=np.float32)
    def __init__(self):
        self.intervals = []


def test_native_warmup_is_causal_and_not_scored():
    class Native:
        capabilities = Capabilities(modalities_in={'text','image','audio'}, native_streaming=True)
        def name(self): return 'test-native'
        def begin(self, spec):
            assert not spec.interaction.responses and not spec.interaction.raw
            assert spec.query_text == interaction().query_text
        def observe(self, frame, t, audio): return Emit(t, reply_text='observed', raw='observed')
        def end(self): pass
    it = interaction(); media = Media()
    record = run(it, Native(), media, decide_adaptation(it, {'text','image','audio'}, 'va'), PreparedInputs(query_text=it.query_text))
    assert [r['time_sec'] for r in record['meta']['warmup']] == [0,1,2]
    assert [e['time_sec'] for e in record['emits']] == [2.4,3,4,5,6,6.3]
    assert media.intervals == [(0,1),(1,2),(2,2.4),(2.4,3),(3,4),(4,5),(5,6),(6,6.3)]


def test_prompted_probes_do_not_receive_responses_or_future_frames():
    class Prompted:
        capabilities = Capabilities()
        def name(self): return 'test-prompted'
        def probe(self, ctx):
            assert ctx.interaction.responses == []
            assert max(f.time_sec for f in ctx.frames) <= ctx.probe_time_sec
            return ProbeResult(False, raw='ABSTAIN')
    it = interaction()
    record = run(it, Prompted(), Media(), decide_adaptation(it, {'text','image'}, 'v'), PreparedInputs(query_text=it.query_text))
    assert record['emits'] == [] and record['meta']['n_probes'] == 6


def test_onset_conversion_is_idempotent_and_excludes_initialization():
    rec = dict(model='minicpm-o', protocol='time_aligned_native', meta={'query_time_sec':2.4},
               emits=[{'time_sec':3,'adapter_time_sec':2,'reply_text':'warmup'},
                      {'time_sec':6,'adapter_time_sec':4,'reply_text':'reply'}])
    new = table2_record(rec)
    assert len(new['emits']) == 1
    assert new['emits'][0]['time_sec'] == 4 and new['emits'][0]['completion_sec'] == 6
    assert table2_record(new) == new


def test_livecc_table2_groups_at_anchor_and_table3_uses_completion():
    rec = dict(model='livecc-7b', protocol='time_aligned_native', meta={'query_time_sec':0},
               emits=[dict(time_sec=t, reply_text=s) for t,s in [(1,'go ...'),(2,'left'),(3,'left'),(4,'stop')]])
    new = table2_record(rec)
    assert [e['time_sec'] for e in new['emits']] == [1,4]
    assert new['emits'][0]['reply_text'] == 'go left'
    from omni_eval.t8.utterances import scoring_emits
    assert scoring_emits(rec)[0]['time_sec'] == 2


def test_t8_ordered_matching_and_query_clipped_bounds():
    assert len(ordered_matching([[False,True],[True,False]])) == 1
    it = Interaction('t8', task_type='impairment_assistance', sub_task_type='C', query_time=5000,
                     responses=[Response(1, notify_trigger_time=6000, assistant_reply_text='first'),
                                Response(2, notify_trigger_time=7000, assistant_reply_text='second')])
    judge = SimpleNamespace(score=lambda q,g,p,rubric: SimpleNamespace(correct=g==p))
    rec = {'model':'custom','emits':[{'time_sec':4,'reply_text':'first'}, {'time_sec':6,'reply_text':'second'}, {'time_sec':7,'reply_text':'first'}]}
    result = score(it, rec, judge)
    assert result['hit_targets'] == 1 and not result['complete_interaction']


@pytest.mark.parametrize('native', [False, True])
def test_t8_collection_uses_current_labels_and_scoring_accepts_prior_labels(native):
    from omni_eval.t8.protocol import run as run_t8
    it = Interaction('t8', task_type='impairment_assistance', sub_task_type='B',
                     target_group='visually_impaired', qa_language='en',
                     query_text='What is ahead?', query_time=2400,
                     responses=[Response(1, notify_trigger_time=5000)])
    model = SimpleNamespace(
        capabilities=Capabilities(native_streaming=native), name=lambda: 'test-model',
        begin=lambda spec: None, observe=lambda frame, t, audio: None, end=lambda: None,
        probe=lambda ctx: ProbeResult(False, raw='ABSTAIN'),
    )
    rec = run_t8(it, model, Media(), decide_adaptation(it, {'text', 'image'}, 'v'), PreparedInputs())
    suffix = 'native' if native else 'probe'
    assert rec['protocol'] == 't8_' + suffix
    assert rec['protocol_version'] == 't8-causal-v1'
    rec['meta']['inference_complete'] = True
    rec['run_fingerprint'] = 'test'
    validate_records([it], [rec], 'table3')
    validate_records([it], [dict(rec, protocol='cat10_' + suffix)], 'table3')


def test_record_validation_rejects_failed_partial_and_duplicate_traces():
    it = interaction()
    rec = dict(interaction_id=it.interaction_id, model='custom',input_config='v',run_fingerprint='test',
               protocol='time_aligned_probe',supported=True,emits=[],meta={'duration_sec':6.3,'inference_complete':True})
    validate_records([it],[rec],'table2')
    with pytest.raises(ValueError): validate_records([it],[rec,rec],'table2')
    with pytest.raises(ValueError): validate_records([it],[],'table2')
    with pytest.raises(ValueError): validate_records([it],[dict(rec,meta={'duration_sec':6.3})],'table2')
    with pytest.raises(ValueError): validate_records([it],[dict(rec,status='terminal_failure')],'table2')


def test_release_preserves_ms_metadata_and_relative_paths(tmp_path):
    record={'schema_version':'omni-interactscope-1.1','interaction_id':'x','query_time_ms':2400,
            'video':'videos/clip.mp4','video_key':'original-id','language':'zh','video_duration_sec':6.3,
            'target_group':'hearing_impaired','qa_language':'en','responses':[{'response_id':1,
            'notify_trigger_time_ms':5000,'valid_response_window_ms':None,'long_memory_start_ms':None}]}
    it=from_release(record)
    assert (it.query_time_sec,it.responses[0].notify_trigger_sec,it.video_duration_sec,it.qa_language)==(2.4,5,6.3,'en')
    assert local_path(tmp_path,it.video)==tmp_path/'videos/clip.mp4'
    with pytest.raises(ValueError): local_path(tmp_path,'../outside')


def test_real_media_uses_pts_and_causal_audio(tmp_path):
    import shutil
    import subprocess
    pytest.importorskip('decord')
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe needed for real media test')
    path = tmp_path/'scene.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=blue:s=64x64:r=10:d=2',
                    '-f','lavfi','-i','sine=frequency=440:sample_rate=16000:duration=2',
                    '-c:v','mpeg4','-c:a','aac','-shortest',str(path)],check=True)
    from omni_eval.protocols.media import CausalMedia
    media=CausalMedia(str(path))
    try:
        frames=media.sample_frames(0,1.25,max_frames=32)
        assert all(not f.synthetic and f.time_sec<=1.25 for f in frames)
        a=media.audio_chunk(0,0.5);b=media.audio_chunk(0.5,0.5);whole=media.audio_chunk(0,1)
        np.testing.assert_array_equal(np.concatenate([a,b]),whole)
        assert len(whole)==16000
    finally:
        media.close()
