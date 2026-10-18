"""Review negative control: perturb only predicted issue latency; never change RTL or golden data."""
import sys
from dataclasses import replace
sys.path.insert(0,'/tmp/opentpu-final-committed-review')
from tools.perfsim import compare_native
Original=compare_native.Simulator
class FaultedSimulator(Original):
 def __init__(self,*args,**kwargs):
  kwargs['architecture']=replace(kwargs['architecture'],dispatch_latency=50)
  super().__init__(*args,**kwargs)
compare_native.Simulator=FaultedSimulator
compare_native.main()
