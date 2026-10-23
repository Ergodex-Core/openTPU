"""Actual C1 arithmetic and native/OCL integration under stalls and recovery."""
import importlib.util
from pathlib import Path
import shutil
import pytest
from opentpu import c1_hbm_rtl as H,c1_rtl as R

pytestmark=pytest.mark.skipif(not shutil.which('verilator'),reason='Verilator required')

@pytest.mark.parametrize('n,kb,mode',[(1,1,'quantized'),(17,3,'random'),(257,9,'quantized'),(512,1,'special'),(31,128,'negative')])
def test_exact_native_kernel(n,kb,mode,tmp_path):
    case=R.workload(n=n,kb=kb,k0=0,ab=0,mode=mode)
    result=H.run(case,tmp_path,latency=41,stall=31,delay_pc0=117)
    assert result['bit_exact']
    assert result['counters'][0x48]*64==n*kb*128

@pytest.mark.parametrize('fault,n',[(1,1),(2,33),(3,49)])
def test_native_fault_drains_and_recovers(fault,n,tmp_path):
    case=R.workload(n=n,kb=9,k0=0,ab=0)
    result=H.run(case,tmp_path,latency=80,stall=27,delay_pc0=151,fault=fault)
    assert result['bit_exact']


def test_alex_frozen_support_and_c1_sources():
    spec=importlib.util.spec_from_file_location('c1_build',H.ROOT/'boards/c1_f2_alex/build.py')
    b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    support=b.verify_support()
    assert support['archive_sha256']=='fdd38e2be3e87fe80d73b9af111911c2d26f8a699b767dfc37b62b8c5a1505dc'
    assert 'c1/otpu_c1_hbm_kernel.sv' in b.CORE_SOURCES
    assert all('c2' not in x for x in b.CORE_SOURCES)
    assert {'.vh','.svh','.tcl'}<=b.PRODUCT_SUFFIXES
