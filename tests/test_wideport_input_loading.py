"""Saved fixed three-MM input loading; all observed producer outputs are audits."""
from copy import deepcopy
import os
from pathlib import Path
import pytest
from tools.wideport_memory_feedback import read_trace
from tools.wideport_write_coupled import validate_write_coupled


@pytest.fixture
def evidence():
    value=os.environ.get("WP_INPUT_ORACLES")
    if not value:
        pytest.skip("requires preserved bounded input-loading fixtures")
    paths=list(Path(value).glob("page-reuse-*/two_credit/page-reuse-summary.json"))
    assert len(paths)==1,paths
    return paths[0].parent


def test_independently_generated_complete_input_loading(evidence):
    result=validate_write_coupled(evidence,generate_input_loading=True)
    loading=result["input_loading"]
    assert loading["independent_requests_data_and_completion"]
    assert loading["loaded_tmem_words"]==1200
    assert loading["boot"]["reads"]==10
    assert loading["instruction_rows"]==loading["sidecar_rows"]==5
    assert result["native_commands"]==1257 and result["native_responses"]==1252
    assert not result["calibrated"] and result["tok_s"] is None


@pytest.mark.parametrize("mutation",["load_request","load_address","load_data","load_done",
                                     "issued_word","boot_request","boot_address","boot_data",
                                     "wait_row","wait_data"])
def test_input_loading_oracle_corruption_rejected(evidence,mutation):
    _,source=read_trace(evidence/"tmfeedback.jsonl")
    events=deepcopy(source)
    if mutation=="load_request":
        e=next(e for e in events if e["memory_sources"]["dma_req"] and not e["memory_sources"]["dma_we"])
        e["memory_sources"]["dma_req"]=0
        signature="load.generated_request"
    elif mutation=="load_address":
        e=next(e for e in events if e["memory_sources"]["dma_req"] and not e["memory_sources"]["dma_we"])
        e["memory_sources"]["dma_addr"]+=32
        signature="load.generated_address"
    elif mutation=="load_data":
        e=next(e for e in events if e["input_loading"]["dma_writes"])
        e["input_loading"]["dma_writes"][0][1]^=1
        signature="load.dma_writes"
    elif mutation=="issued_word":
        e=next(e for e in events if e["input_loading"]["issued_dma_writes"])
        e["input_loading"]["issued_dma_writes"][0][0]+=8
        signature="load.issued_dma_writes"
    elif mutation=="load_done":
        e=next(e for e in events if e["input_loading"]["dma_busy"])
        e["done"]|=1
        signature="load.generated_done"
    elif mutation in ("boot_request","boot_address"):
        e=next(e for e in events if e["memory_sources"]["loader_req"])
        if mutation=="boot_request":
            e["memory_sources"]["loader_req"]=0
            signature="boot.generated_loader_req"
        else:
            e["memory_sources"]["loader_addr"]+=32
            signature="boot.generated_address"
    elif mutation=="boot_data":
        e=next(e for e in events if e["input_loading"]["loader_im_we"])
        e["input_loading"]["loader_im_data"]=f"{int(e['input_loading']['loader_im_data'],16)^1:0256x}"
        signature="boot.im_data"
    else:
        e=next(e for e in events if e["input_loading"]["loader_sc_we"])
        if mutation=="wait_row":
            e["input_loading"]["loader_sc_row"]+=1
            signature="boot.sc_row"
        else:
            e["input_loading"]["loader_sc_data"]=f"{int(e['input_loading']['loader_sc_data'],16)^1:064x}"
            signature="boot.sc_data"
    with pytest.raises(ValueError,match=signature):
        validate_write_coupled(evidence,generate_input_loading=True,slice_oracles=events)
