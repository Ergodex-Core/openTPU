"""Strict, serializable configuration; periods are integer picoseconds."""
from dataclasses import dataclass, asdict
from ..isasim import Config


class Unsupported(ValueError):
    pass


@dataclass(frozen=True)
class Architecture:
    core_period_ps: int = 5000
    window: int = 16
    mxu_queue: int = 2
    mxu_fifo_chunks: int = 1024
    mxu_result_rows: int = 32
    mxu_chain: int = 16
    mxu_epilogue_lanes: int = 0
    vpu_slots: int = 10
    vpu_composite_lanes: int = 0
    result_queue: int = 128
    tmem_read_ports_per_bank: int = 32
    tmem_write_ports_per_bank: int = 1
    quant_row_buffer_chunks: int = 1024
    issue_period: int = 1
    dispatch_latency: int = 2
    tmem_read_latency: int = 2
    max_cycles: int = 10_000_000

    def validate(self, cfg: Config):
        for k, v in asdict(self).items():
            if not isinstance(v, int) or v < (0 if k in ('mxu_epilogue_lanes', 'vpu_composite_lanes') else 1):
                raise ValueError(f'{k} must be a positive integer (lane overrides may be zero)')
        if cfg.S != 1:
            raise Unsupported('multi-slice collective/interconnect timing is not implemented; S must be 1')
        if cfg.SEQ_WC or cfg.STREAM or cfg.DSTEP:
            raise Unsupported('WAITC/STREAM/DSTEP configurations are not modelled')
        if cfg.D not in (32, 64, 128) or cfg.LANES not in (4, 8, 16):
            raise Unsupported('supported D: 32/64/128; LANES: 4/8/16')
        if cfg.MCOLS not in (2, 4, 8, 16) or cfg.act_rows < cfg.MCOLS:
            raise Unsupported('MCOLS must be 2/4/8/16; ACT_ROWS must cover MCOLS')
        if self.vpu_composite_lanes > cfg.LANES or self.mxu_epilogue_lanes > cfg.MCOLS:
            raise ValueError('lane override exceeds hardware width')
        if self.mxu_fifo_chunks < cfg.ACT_BLOCKS:
            raise Unsupported('MXU FIFO must hold a complete maximum weight row')


@dataclass(frozen=True)
class MemoryConfig:
    backend: str = 'simple'
    controller_period_ps: int = 1000
    memory_period_ps: int = 1000
    transaction_bytes: int = 16
    outstanding: int = 64
    ingress_per_controller_tick: int = 1
    simple_latency_ticks: int = 20
    simple_channels: int = 8
    simple_queue_per_channel: int = 32
    bridge: str = ''
    ramulator_config: str = ''
    rmw: bool = True
    record_trace: bool = False

    def validate(self):
        if self.backend not in ('simple', 'ramulator'):
            raise Unsupported(f'unknown memory backend {self.backend}')
        for k, v in asdict(self).items():
            if k in ('backend', 'bridge', 'ramulator_config', 'rmw', 'record_trace'):
                continue
            if not isinstance(v, int) or v <= 0:
                raise ValueError(f'{k} must be a positive integer')
        if self.transaction_bytes & (self.transaction_bytes - 1):
            raise ValueError('transaction_bytes must be a power of two')
        if self.controller_period_ps != self.memory_period_ps:
            raise Unsupported('this Ramulator controller ticks at DRAM tCK; independent ingress is modelled, divided controller clocks are unsupported')
        if self.backend == 'ramulator' and (not self.bridge or not self.ramulator_config):
            raise ValueError('Ramulator requires bridge and ramulator_config; there is no fallback')
