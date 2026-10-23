"""Tiny native FP32 graph executor; OpenTPU is an optional external dependency.

Every arithmetic/reduction operation runs on the card. The host performs tensor
layout/copies, token indexing and initialization only. Intermediates cross PCIe
per operation: this bounded correctness path is intentionally not optimized.
"""
from dataclasses import replace
from functools import partial
from math import prod
import os

import numpy as np
from opentpu import language as ol
from opentpu.runtime import Input, Output, compile_kernel

from .backend import _array, _inputs, _slice, _validate


@ol.jit
def _kernel(x, y, out, op):
    a = ol.load(x)
    if op in ("add", "sub", "mul", "div", "matmul"):
        b = ol.load(y)
    if op == "add": z = a + b
    elif op == "sub": z = a - b
    elif op == "mul": z = a * b
    elif op == "div": z = a / b
    elif op == "neg": z = -a
    elif op == "exp": z = ol.exp2(a * ol.LOG2E)
    elif op == "log": z = ol.log2(a) * ol.LN2
    elif op == "rsqrt": z = ol.rsqrt(a)
    elif op == "tanh":
        a = ol.minimum(ol.maximum(a, -40.0), 40.0)
        z = 2.0 * ol.recip(1.0 + ol.exp2(a * (-2.0 * ol.LOG2E))) - 1.0
    elif op == "sum": z = ol.sum(a, axis=-1)
    elif op == "max": z = ol.max(a, axis=-1)
    elif op == "matmul":
        z = ol.empty(out.shape, dense=True)
        for j in range(out.shape[1]):
            column = ol.empty((b.shape[0],))
            column.column().set(b[:, j:j + 1])
            z[:, j:j + 1].set((a @ column).column())
    else: raise ValueError(f"Unsupported numerical operation: {op}")
    ol.store(out, z)


class OpenTPUBackend:
    """Own one idle, already-loaded F2 device until close(); never loads images."""

    device = "f2"

    def __init__(self, image="burst200", device="f2:0"):
        if image != "burst200" or device != "f2:0":
            raise ValueError("Native backend requires the burst200 image on f2:0")
        from opentpu.host import regs as R
        from opentpu.host.board import Board, BoardBackend, device_config
        from opentpu.host.f2 import F2Transport
        from opentpu.host.runstate import DeviceLock
        from opentpu.target import arch_hash, config
        if float(os.environ.get("OTPU_LOCK_WAIT", "0") or 0) != 0:
            raise ValueError("Set OTPU_LOCK_WAIT=0; native proof runs must not wait for a device")
        self.transport = self.board = self.legacy = None
        self.kernel_count, self.device_cycles, self.op_counts = 0, 0, {}
        try:
            self.legacy = DeviceLock("f2:0", wait=0)
            self.transport = F2Transport(device)
            info = Board(self.transport, lock=False, calibrate=False).info()
            if info["running"] or info["cfg"]["hash"] != arch_hash(config("f2")):
                raise RuntimeError("F2 must be idle and match the OpenTPU F2 architecture")
            if info["core_khz"] != 200000 or info["build_id"] != 0x7e42a626 or not info["calibrated"]:
                raise RuntimeError("burst200 requires calibrated build 7e42a626 at 200 MHz")
            errors = R.ST_ERROR | R.ST_AXI_ERR | R.ST_WAIT_TO | R.ST_ECC_DED
            if self.transport.reg_read(R.R_STATUS) & errors:
                raise RuntimeError("F2 status reports an existing hardware error")
            if self.transport.reg_read(R.R_SCRATCH) != R.DRAM_INIT:
                raise RuntimeError("F2 memory must already be initialized by its owner")
            self.cfg = replace(device_config(info), DRAM_BYTES=1 << 20)
            self.board = BoardBackend(self.cfg, [np.zeros(1 << 19, np.uint8)],
                                      transport=self.transport, model="swarmfpga", own_transport=False)
            self.board.board.wait = partial(self.board.board.wait, timeout=2.0)
        except BaseException:
            self.close()
            raise

    def close(self):
        board, self.board = self.board, None
        transport, self.transport = self.transport, None
        legacy, self.legacy = self.legacy, None
        try:
            if board is not None:
                board.close()
        finally:
            try:
                if transport is not None:
                    transport.close()
            finally:
                if legacy is not None:
                    legacy.release()

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()

    def _launch(self, op, x, y=None, shape=None):
        y = np.zeros(1, np.float32) if y is None else y
        shape = tuple(shape or x.shape)
        if any(a.dtype != np.float32 for a in (x, y)):
            raise ValueError("Native arithmetic supports float32 tensors only")
        if x.size + y.size + prod(shape) > 8192 or not x.size or not prod(shape):
            raise ValueError("Native proof backend limits each kernel to 8192 tensor words")
        compiled, images = compile_kernel(_kernel, self.cfg, x=Input(x), y=Input(y),
                                          out=Output(shape), op=op)
        end = compiled.layout.params["out"][1] + 4 * prod(shape)
        if end > 1 << 19:
            raise ValueError("Native kernel exceeds reserved staging memory")
        self.board.write(0, 0, images[0][:end])
        stats = self.board.run(compiled.programs)
        self.kernel_count += 1
        self.device_cycles += stats["cycles"]
        self.op_counts[op] = self.op_counts.get(op, 0) + 1
        addr = compiled.layout.params["out"][1]
        return self.board.read(0, addr, 4 * prod(shape)).view(np.float32).reshape(shape).copy()

    def run(self, program, feeds):
        _validate(program)
        feeds, values = _inputs(program, feeds), {}
        for node in program["nodes"]:
            op, a, shape = node["op"], node["attrs"], tuple(node["shape"])
            args = [values[i] for i in node["inputs"]]
            x = args[0] if args else None
            if op == "input": value = feeds[a["name"]]
            elif op == "const": value = _array(a["value"], a.get("dtype", "float32"), shape)
            elif op == "reshape": value = x.reshape(shape)
            elif op == "transpose": value = x.transpose(a["axes"])
            elif op == "broadcast_to": value = np.broadcast_to(x, shape)
            elif op == "concat": value = np.concatenate(args, axis=a["axis"])
            elif op == "slice": value = x[_slice(a, x.ndim)]
            elif op == "slice_grad":
                value = np.zeros(shape, np.float32)
                value[_slice(a, value.ndim)] = x
            elif op in ("gather", "scatter_add"):
                indices, axis = args[1], a["axis"]
                bound = x.shape[axis] if op == "gather" else shape[axis]
                if indices.dtype != np.int64 or np.any(indices < 0) or np.any(indices >= bound):
                    raise ValueError("Invalid native gather/scatter indices")
                if op == "gather": value = np.take(x, indices, axis=axis)
                else:
                    rest = shape[:axis] + shape[axis + 1:]
                    rows = np.moveaxis(x.reshape(shape[:axis] + (indices.size,) + shape[axis + 1:]), axis, 0)
                    rows, ids = rows.reshape(indices.size, -1), indices.reshape(-1)
                    result = np.zeros((bound, prod(rest)), np.float32)
                    for index in np.unique(ids):
                        result[index] = self._launch("sum", rows[ids == index].T.copy(), shape=(prod(rest),))
                    value = np.moveaxis(result.reshape((bound,) + rest), 0, axis)
            elif op == "matmul":
                batch = shape[:-2]
                left = np.broadcast_to(x, batch + x.shape[-2:])
                right = np.broadcast_to(args[1], batch + args[1].shape[-2:])
                value = np.empty(shape, np.float32)
                for index in np.ndindex(batch):
                    value[index] = self._launch(op, left[index].copy(), right[index].copy(), shape[-2:])
            elif op in ("sum", "max"):
                axes = tuple(a["axis"])
                keep = tuple(i for i in range(x.ndim) if i not in axes)
                matrix = x.transpose(keep + axes).reshape(prod(x.shape[i] for i in keep), -1)
                value = self._launch(op, matrix.copy(), shape=(matrix.shape[0],)).reshape(shape) if axes else x.copy()
            else:
                vectors = [np.broadcast_to(arg, shape).reshape(-1).copy() for arg in args]
                value = self._launch(op, *vectors).reshape(shape)
            values[node["id"]] = _array(value, str(value.dtype), shape)
        return {name: values[i] for name, i in program["outputs"].items()}
