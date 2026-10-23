"""Opt-in protected host frontend for simulated W4A16 endpoints only.

This is not Board/XDMA, a production descriptor ABI, an ISA opcode or a hardware
capability register. Numeric context IDs are trusted simulation authority.
All snapshot/check/lease and visibility operations are synchronous/single-threaded.
"""
from collections import deque
from dataclasses import dataclass,fields

from ..mxu_job import SOFTWARE_CAPABILITIES,local_job_fields
from ..nvfp4_transport import (SimMemory,LocalJobTransport,decode_descriptor,uint,
    TransportError,DescriptorError,AccessDenied,Busy,StaleResponse,U64)


class CapabilityMismatch(TransportError):pass


@dataclass(frozen=True)
class Capabilities:
    protocol:str='opentpu.w4a16.sim-host'
    major:int=1
    minor:int=0
    framing:str='len32be-json-4096-v1'
    profiles:tuple=SOFTWARE_CAPABILITIES
    tokens:int=1
    outputs:int=1
    max_k:int=64
    in_flight:int=1
    address_bits:int=64
    fences:str='sync-input-release;acked-y-before-acquire-v1'
    cancellation:str='discard-previsibility;preserve-published-v1'


CAPABILITIES=Capabilities()


def check_capabilities(value):
    # Exact tuple, including types: a newer/unknown minor is not silently assumed
    # compatible. No fallback from legacy CAPS.W4/MX or CAPS2's old-map defaults.
    if type(value) is not Capabilities or any(
        type(getattr(value,f.name)) is not type(getattr(CAPABILITIES,f.name)) or
        getattr(value,f.name)!=getattr(CAPABILITIES,f.name) for f in fields(Capabilities)):
        raise CapabilityMismatch('unsupported complete simulated frontend capability tuple')


@dataclass(frozen=True)
class Session:
    namespace:object
    serial:int
    context:int
    generation:int


@dataclass(frozen=True)
class InputRelease:
    session:Session
    serial:int
    job:object
    snapshot:tuple


class SimEndpoint:
    """One trusted context, one accepted job, one completion credit.

    transport is exposed only to an explicit simulation device driver: it fetches
    packets, supplies a separately computed/RTL private result and acknowledges Y.
    The host frontend does not compute arithmetic or infer completion from HALTED.
    """
    def __init__(self,memory,*,context,generation):
        if type(memory) is not SimMemory:raise TransportError('SimMemory endpoint only')
        uint(context);uint(generation)
        self.context=context;self.transport=LocalJobTransport(memory,generation=generation)
        self._namespace=object();self._serial=0;self._capabilities=CAPABILITIES
        self._session=None;self._release=None;self._owner=None;self._handle=None
        self._last_epoch=None
        self.events=deque(maxlen=512)

    @property
    def capabilities(self):return self._capabilities

    def _next(self):
        if self._serial==U64:raise Busy('session/fence sequence exhausted')
        self._serial+=1;return self._serial

    def negotiate(self,*,context,generation,capabilities=CAPABILITIES):
        check_capabilities(capabilities);check_capabilities(self.capabilities)
        uint(context);uint(generation)
        if context!=self.context:raise AccessDenied('endpoint context mismatch')
        if generation!=self.transport.generation:raise StaleResponse('endpoint generation mismatch')
        if self.transport.state!='IDLE':raise Busy('old job/completion must drain before negotiation')
        s=Session(self._namespace,self._next(),context,generation)
        self._session=s;self._release=None
        self.events.append(('NEGOTIATED',s))
        return s

    def _check(self,session):
        if session is not self._session or session is None:
            raise StaleResponse('stale/foreign negotiated session')
        if session.generation!=self.transport.generation:raise StaleResponse('stale session generation')
        check_capabilities(self.capabilities)

    def _descriptor(self,session,frame):
        self._check(session)
        j=decode_descriptor(frame)
        try:local_job_fields(j,session.generation)
        except ValueError as e:raise DescriptorError(str(e)) from e
        if j.context!=session.context:raise AccessDenied('descriptor outside bound context')
        if self._last_epoch is not None and j.epoch<=self._last_epoch:
            raise StaleResponse('epoch must advance within a generation without wrap')
        return j

    def release_inputs(self,session,frame):
        j=self._descriptor(session,frame)
        if self.transport.state!='IDLE':raise Busy('job/completion credit occupied')
        snapshot=self.transport.memory.input_snapshot(j)
        token=InputRelease(session,self._next(),j,snapshot)
        self._release=token
        self.events.append(('INPUT_RELEASE',token))
        return token

    def submit(self,session,frame,release):
        j=self._descriptor(session,frame)
        if self.transport.state!='IDLE':raise Busy('job/completion credit occupied')
        if release is None or release is not self._release or release.session is not session:
            raise StaleResponse('stale/foreign/consumed input release')
        if j!=release.job:raise StaleResponse('descriptor changed after input release')
        if self.transport.memory.input_snapshot(j)!=release.snapshot:
            raise StaleResponse('mapping/input content changed after release; release again')
        # Single-threaded check-to-lease operation, not a concurrent TOCTOU proof.
        h=self.transport.submit(frame)
        self._release=None;self._owner=session;self._handle=h;self._last_epoch=j.epoch
        self.events.append(('ACCEPTED',h))
        return h

    def _completion_owner(self,session):
        if self._owner is not None:
            if session is not self._owner:raise StaleResponse('completion belongs to another session')
        else:self._check(session)

    def poll_completion(self,session):
        self._completion_owner(session)
        return self.transport.completion

    def acquire_completion(self,session):
        self._completion_owner(session)
        c=self.transport.take_completion(context=session.context)
        if c is not None:
            self.events.append(('COMPLETION_ACQUIRE',c))
            self._owner=None;self._handle=None
        return c

    def cancel(self,session,handle):
        self._completion_owner(session)
        if session is not self._owner or handle!=self._handle:raise StaleResponse('wrong accepted job')
        self.transport.cancel(handle)

    def reset(self,session,new_generation):
        self._check(session)
        # Validate/advance before invalidating tokens: invalid reset is a no-op.
        self.transport.reset(new_generation)
        self._session=None;self._release=None;self._last_epoch=None
        self.events.append(('RESET',new_generation))
        # Original owner remains permitted to acquire its terminal old record.


class ProtectedFrontend:
    """Explicit opt-in host API; legacy board transports fail closed without I/O."""
    def __init__(self,endpoint,*,context,generation,capabilities=CAPABILITIES):
        if type(endpoint) is not SimEndpoint:
            raise CapabilityMismatch('no negotiated W4A16 simulated endpoint; legacy boards unsupported')
        self._endpoint=endpoint
        self.session=endpoint.negotiate(context=context,generation=generation,capabilities=capabilities)
        self.capabilities=CAPABILITIES

    def release_inputs(self,frame):return self._endpoint.release_inputs(self.session,frame)
    def submit(self,frame,release):return self._endpoint.submit(self.session,frame,release)
    def poll_completion(self):return self._endpoint.poll_completion(self.session)
    def acquire_completion(self):return self._endpoint.acquire_completion(self.session)
    def cancel(self,handle):return self._endpoint.cancel(self.session,handle)
    def reset(self,new_generation):return self._endpoint.reset(self.session,new_generation)
