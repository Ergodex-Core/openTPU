"""Transport protocol tests. Mock text is NOT ASR/model/physical evidence."""
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import struct

import pytest

from opentpu.asr.phonon2_runner import (Artifact, Runner, RunnerError, BoardTransport,
    Permit, AGFI, CFG_HASH, MODEL, REVISION, PROFILE, PROGRAM_AT, Cancelled)
from tools.phonon2_prepare_runner import Writer

ROOT = Path(__file__).resolve().parents[1]


def identity():
    return dict(D=128, MCOLS=8, LANES=8, act_rows=16, core_khz=150000,
                calibrated=True, running=False, status=8,
                cfg=dict(S=1, TMEM_WORDS=65536, IMEM_WORDS=32768, ACT_BLOCKS=128,
                         DRAM_BYTES=4 << 30, SEQ_WC=True, hash=int(CFG_HASH, 16)),
                caps={k: True for k in ('mx', 'pair', 'gen', 'dstep', 'stream', 't2')},
                caps2=dict(portb=4, rowlanes=True, aline=64))


class Mock:
    backend = 'mock'

    def __init__(self):
        self.log, self.memory = [], bytearray(4096)
        self.owned, self.fail_at, self.runs = True, None, 0
        self.info, self.on_wait, self.on_write = identity(), lambda: None, lambda: None
        self.state = [1, 8192, 1, 1, 0, 1, 0, 0, 0] + [0]*7
        self.segments, self.lengths = b'', [0.0]

    def lease_valid(self, lease):
        return self.owned and lease == 'spare-test'

    def verify_host_sources(self, binding):
        self.log.append('verify_sources')

    def identity(self):
        self.log.append('identity')
        return self.info

    @contextmanager
    def session(self, check, deadline):
        yield

    def write(self, at, data):
        self.log.append(('write', at, data))
        self.memory[at:at+len(data)] = data
        self.on_write()

    def load_program(self, at, code, waits):
        assert at == PROGRAM_AT and len(waits) == len(code)//4
        self.log.append('load')

    def start(self):
        assert self.log[-1] == 'load'
        self.log.append('start')

    def wait(self, timeout):
        assert self.log[-1] == 'start' and timeout > 0
        self.runs += 1
        self.log.append('wait')
        self.on_wait()
        if self.runs == self.fail_at:
            raise TimeoutError('mock stuck program')
        if self.runs == 693:
            self.memory[2048:2112] = struct.pack('<16f', *self.state)
            self.memory[2176:2176+len(self.lengths)*4] = struct.pack('<'+'f'*len(self.lengths), *self.lengths)
            self.memory[2304:2304+len(self.segments)] = self.segments
        return dict(status=1, ecc_ded=False, cycles=0)

    def read(self, at, size):
        self.log.append(('read', at, size))
        return bytes(self.memory[at:at+size])

    def stop_and_drain(self, lease):
        self.log.append('stop' if self.lease_valid(lease) else 'stop_refused_lost_lease')
        return self.lease_valid(lease)


def fixture_artifact(tmp_path, *, mutate=None):
    w = Writer(tmp_path/'artifact')
    code = w.blob(bytes(32)); waits = w.blob(bytes(8)); const = w.blob(b'fixed')
    program = dict(kind='program', name='placeholder', words=8, code=code, waits=waits)
    actions = [dict(program, name='request_reset'),
               dict(kind='write', name='constants', at=0, bytes=5, blob=const),
               dict(kind='pcm', name='pcm')]
    actions += [dict(program, name='phase.'+str(i)) for i in range(691)]
    actions.append(dict(program, name='tdt.2'))
    m = dict(schema=1, model=MODEL, revision=REVISION, profile=PROFILE, agfi=AGFI,
             cfg_hash=CFG_HASH, samples=320, pcm_bytes=640, raw_padded_bytes=640,
             input_at=128, graph_bytes=4096, program_at=PROGRAM_AT, frames=1,
             valid=1, max_steps=9, output=dict(state=2048, text_lengths=2176, text=2304),
             actions=actions, blobs=w.blobs,
             binding=json.loads((ROOT/'docs/phonon2/runner_source_binding.json').read_text()))
    if mutate:
        mutate(m)
    raw = json.dumps(m).encode()
    (w.root/'manifest.json').write_bytes(raw)
    return w.root, hashlib.sha256(raw).hexdigest()


def permit(**kwargs):
    values = dict(approved=True, spare_card=True, lease_id='spare-test', agfi=AGFI,
                  image_query_evidence='MOCK: not a live image query', health_evidence='MOCK',
                  expires_monotonic=100)
    return Permit(**(values | kwargs))


def make(tmp_path):
    root, sha = fixture_artifact(tmp_path)
    device = Mock()
    return Runner(Artifact(root, sha), device, clock=lambda: 0), device


def test_full_transport_order_pcm_unaltered_and_mock_never_accepts_physical(tmp_path):
    runner, device = make(tmp_path)
    pcm = bytes(range(256))*2 + bytes(128)
    result = runner.run(pcm, permit=permit())
    assert device.runs == result['receipt']['dispatches'] == 693
    assert device.log.index(('write', 128, pcm)) < device.log.index('start', device.log.index(('write', 128, pcm)))
    assert [x for x in device.log if isinstance(x, tuple) and x[0]=='read'] == [('read',2048,64),('read',2176,4),('read',2304,32)]
    assert result['text'] == '' and not result['receipt']['hardware_used']
    assert not result['receipt']['physical_acceptance'] and not result['receipt']['ready']
    assert result['receipt']['program_wait_bytes'] == 693 * (128+8)


def test_device_text_bytes_are_coalesced_without_cleanup_or_ctc(tmp_path):
    runner, d = make(tmp_path)
    d.state = [1, 3, 1, 3, 3, 2, 2, 9, 1]+[0]*7
    d.lengths = [3, 6, 0]
    d.segments = b' a ' + bytes(29) + 'é é '.encode() + bytes(26) + bytes(32)
    result = runner.run(bytes(640), permit=permit())
    assert result['text'] == ' a é é '


@pytest.mark.parametrize('field,value', [('approved',False),('spare_card',False),('agfi','wrong'),('expires_monotonic',0)])
def test_preflight_refuses_before_device_access(tmp_path,field,value):
    runner,d = make(tmp_path)
    with pytest.raises(RunnerError): runner.run(bytes(640),permit=permit(**{field:value}))
    assert d.log == [] and not runner.dirty


@pytest.mark.parametrize('mutate', [lambda i:i['cfg'].update(hash=0), lambda i:i.update(running=True),
    lambda i:i.update(status=512), lambda i:i['caps'].update(gen=False), lambda i:i['caps2'].update(portb=1)])
def test_live_profile_and_health_fail_before_writes(tmp_path,mutate):
    runner,d = make(tmp_path); mutate(d.info)
    with pytest.raises(RunnerError): runner.run(bytes(640),permit=permit())
    assert d.log == ['verify_sources','identity'] and not runner.dirty


def test_manifest_and_blob_corruption_never_touch_transport(tmp_path):
    root,sha=fixture_artifact(tmp_path)
    with pytest.raises(RunnerError): Artifact(root,'0'*64)
    blob=next((root/'blobs').iterdir());blob.write_bytes(b'corrupt')
    with pytest.raises(RunnerError): Artifact(root,sha)


@pytest.mark.parametrize('mutate', [lambda m:m.update(graph_bytes=PROGRAM_AT),
    lambda m:m.update(input_at=4096), lambda m:m.update(frames=2),
    lambda m:m['actions'].pop(), lambda m:m['actions'][0].update(words=32776)])
def test_artifact_residency_and_sequencing_bounds(tmp_path,mutate):
    root,sha=fixture_artifact(tmp_path,mutate=mutate)
    with pytest.raises(RunnerError): Artifact(root,sha)


def test_timeout_drains_and_requires_new_owner_session(tmp_path):
    runner,d = make(tmp_path);d.fail_at=12
    with pytest.raises(RunnerError) as caught:runner.run(bytes(640),permit=permit())
    assert caught.value.code=='ASR_DEADLINE' and caught.value.receipt['quiescent']
    assert d.log[-1]=='stop' and not any(isinstance(x,tuple) and x[0]=='read' for x in d.log)
    with pytest.raises(RunnerError,match='recovery'):runner.run(bytes(640),permit=permit())


def test_cancel_before_dispatch_and_during_wait(tmp_path):
    runner,d=make(tmp_path)
    with pytest.raises(RunnerError) as e:runner.run(bytes(640),permit=permit(),cancelled=lambda:True)
    assert e.value.code=='ASR_CANCELLED' and not d.log
    cancelled=[False];d.on_wait=lambda:cancelled.__setitem__(0,True)
    with pytest.raises(RunnerError) as e:runner.run(bytes(640),permit=permit(),cancelled=lambda:cancelled[0])
    assert e.value.code=='ASR_CANCELLED' and d.log[-1]=='stop' and d.runs==1


def test_lost_lease_does_not_override_or_read_result(tmp_path):
    runner,d=make(tmp_path);d.on_wait=lambda:setattr(d,'owned',False)
    with pytest.raises(RunnerError) as e:runner.run(bytes(640),permit=permit())
    assert e.value.code=='ASR_LEASE_LOST' and e.value.receipt['quiescent'] is False
    assert d.log[-1]=='stop_refused_lost_lease'


@pytest.mark.parametrize('state_update,lengths,segments,code', [
    ({5:3},[0],b'','ASR_DECODE_INCOMPLETE'),({3:10},[0],b'','ASR_OUTPUT_INVALID'),
    ({0:0},[0],b'','ASR_OUTPUT_INVALID'),({7:1},[0],b'','ASR_OUTPUT_INVALID'),
    ({7:33},[33],b'','ASR_OUTPUT_INVALID'),({7:1,4:1,6:1},[1],b'\xff'+bytes(31),'ASR_OUTPUT_INVALID'),
    ({3:float('nan')},[0],b'','ASR_OUTPUT_INVALID')])
def test_invalid_or_incomplete_output_is_never_success(tmp_path,state_update,lengths,segments,code):
    runner,d=make(tmp_path)
    for k,v in state_update.items():d.state[k]=v
    d.lengths,d.segments=lengths,segments
    with pytest.raises(RunnerError) as e:runner.run(bytes(640),permit=permit())
    assert e.value.code==code and e.value.receipt['outcome']=='error'


def test_second_request_always_begins_with_reset_and_pcm_upload(tmp_path):
    runner,d=make(tmp_path)
    runner.run(bytes(640),permit=permit())
    d.runs=0;d.memory[:]=bytes([255])*4096;d.log=[]
    assert runner.run(bytes(640),permit=permit())['text']==''
    assert d.log[2:5]==['load','start','wait']


def test_board_adapter_is_inert_and_passes_explicit_wait_entries():
    class Board:
        def load_program(self,at,code,entries):self.loaded=(at,code.tobytes(),entries.tobytes())
    b=Board();t=BoardTransport(b,lambda _:True)
    assert not hasattr(b,'loaded')
    t.load_program(PROGRAM_AT,bytes(32),bytes(8))
    assert b.loaded==(PROGRAM_AT,bytes(32),bytes(8))


def test_cancel_poll_proxy_aborts_once_and_restores_original_transport():
    class Transport:
        def poll(self,*args):raise TimeoutError('still running')
    class Board:
        def __init__(self):self.t=Transport();self.stops=0
        def wait(self,timeout):
            try:return self.t.poll(0,1,1,timeout)
            except TimeoutError:
                self.stops+=1
                # Abort's stop/drain can poll again without reentering cancellation.
                assert self.t.poll==original.poll
                raise
    b=Board();original=b.t;t=BoardTransport(b,lambda _:True,clock=lambda:0)
    def cancel():raise Cancelled('cancel')
    with pytest.raises(Cancelled):
        with t.session(cancel,100):t.wait(100)
    assert b.t is original and b.stops==1
