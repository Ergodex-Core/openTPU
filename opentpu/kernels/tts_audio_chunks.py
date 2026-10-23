"""Device codec-ID retention/replay; no host code or feature reconstruction."""
from .. import language as ol


@ol.jit
def retain_current(m):
    ol.store(m.current,ol.load(m.ids))


@ol.jit
def restore_current(m):
    ol.store(m.ids,ol.load(m.current))


@ol.jit
def record_code(m,slot: int):
    ol.store(m.codes[slot:slot+1,:],ol.load(m.ids)[None,:])


@ol.jit
def replay_code(m,slot: int):
    ol.store(m.ids,ol.load(m.codes[slot,:]))


@ol.jit
def restart_front(m):
    ol.store(m.history0,ol.zeros(m.history0.shape))
    ol.store(m.history1,ol.zeros(m.history1.shape))
