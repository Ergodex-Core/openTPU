import json
import numpy as np
from opentpu import Config, isa as I
from opentpu.isasim import Slice
from opentpu.perfsim.io import load_bundle, save_bundle, state_hash


def test_export_only_bundle_roundtrip(tmp_path):
    cfg=Config(TMEM_WORDS=1024,DRAM_BYTES=65536)
    image=np.array([1,2,3,4,5,6,7,8],dtype=np.uint8)
    save_bundle(tmp_path,cfg,[I.halt()],image)
    path=tmp_path/'input.json'
    obj=json.loads(path.read_text())
    obj['config_export']=obj.pop('config')
    path.write_text(json.dumps(obj))
    loaded,program,data,_=load_bundle(tmp_path)
    assert loaded==cfg and program[0].op==I.HALT
    assert np.array_equal(data,image)


def test_numeric_state_hash_includes_quantization_metadata():
    state=Slice(Config(TMEM_WORDS=1024,DRAM_BYTES=65536),0,[I.halt()],np.zeros(4,np.uint8))
    before=state_hash(state)
    state.ascale[0,0]=3
    state.adoff[0,0,0]=2
    after=state_hash(state)
    assert before['ascale']!=after['ascale'] and before['adoff']!=after['adoff']
    assert before['act']==after['act']
