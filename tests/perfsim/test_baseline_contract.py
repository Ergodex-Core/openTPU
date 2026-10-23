"""Reject mismatched topology/launch paths rather than comparing unrelated machines."""
import json,subprocess,sys
import numpy as np
import pytest
from opentpu import Config,isa as I
from opentpu.perfsim import Architecture,Simulator,Unsupported
from opentpu.perfsim.baseline import rtl_defaults,rtl_parameters,verify_rtl_parameters


def test_baseline_ignores_ambient_board_knobs(monkeypatch):
 for k,v in {'OTPU_UARCH':'board','OTPU_MXU_E':'4','OTPU_TMEM_IMPL':'3','OTPU_TMEM_PMP':'2'}.items():monkeypatch.setenv(k,v)
 p=json.loads(subprocess.check_output([sys.executable,'-c','import json; from opentpu import Config; from opentpu.perfsim.baseline import rtl_parameters; print(json.dumps(rtl_parameters(Config(D=128,MCOLS=4))))']))
 assert (p['TMEM_IMPL'],p['TMEM_PMP'],p['MXU_E'],p['AXI'])==(0,0,0,0)
 assert (p['WIN'],p['RPB'],p['WPB'],p['FIFO_DEPTH'])==(16,32,1,1024)


@pytest.mark.parametrize('field,value',[('TMEM_IMPL',3),('TMEM_PMP',2),('MXU_E',4),('MXU_CL',8),('QUANT_XD',0),('ULANES',4),('COLL_GATHER',0),('AXI',2)])
def test_reject_wrong_rtl_machine(field,value):
 cfg=Config(D=128,MCOLS=4);p=rtl_parameters(cfg);p[field]=value
 with pytest.raises(ValueError,match='RTL build/config mismatch'):verify_rtl_parameters({'parameters':p},cfg)


def test_explicit_e4_contract_and_legacy_defaults():
 cfg=Config(D=128,MCOLS=4);arch=Architecture(mxu_epilogue_lanes=4);p=rtl_parameters(cfg,arch)
 assert p['MXU_E']==4;assert verify_rtl_parameters({'parameters':p},cfg,arch)==p
 defaults=rtl_defaults(cfg);legacy={k:v for k,v in rtl_parameters(cfg).items() if defaults[k]!=v}
 assert verify_rtl_parameters({'parameters':legacy},cfg)==rtl_parameters(cfg)


@pytest.mark.parametrize('impl',[1,2,3])
def test_reject_unimplemented_tmem(impl):
 with pytest.raises(Unsupported,match='TMEM_IMPL=0'):Architecture(tmem_impl=impl).validate(Config())


def test_program_resident_boundary_rejects_waitw_and_chain():
 for ins in [I.Instr(I.WAITW),I.Instr(I.HALT,flags=1)]:
  with pytest.raises(Unsupported):Simulator(Config(),[ins,I.halt()],np.zeros(4,np.uint8))


def test_reject_unrepresented_harness_queue_override():
 with pytest.raises(Unsupported,match='mxu_queue'):rtl_parameters(Config(),Architecture(mxu_queue=4))
