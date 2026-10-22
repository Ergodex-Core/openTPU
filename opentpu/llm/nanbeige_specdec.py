"""Greedy Nanbeige target verification with an explicit CPU DSpark reference draft.

Target matrix work uses the supplied Engine backend (ISA, RTL, or board). The
draft is NumPy on the host, never represented as an FPGA draft implementation.
Every returned token comes from target logits, including rejection corrections.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import operator
import time

import numpy as np

from ..compiler import CompileError
from .dspark import Context, verify_greedy


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer") from None


@dataclass
class Stats:
    target_backend: str
    draft_backend: str
    prompt_tokens: int = 0
    generated_tokens: int = 0
    decode_tokens: int = 0  # Does not include the first token selected after prefill.
    target_seconds: float = 0.0  # Includes compile, execution, and hidden-state readback.
    draft_seconds: float = 0.0  # Includes context projection and draft forward.
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    total_seconds: float = 0.0
    proposed_tokens: int = 0  # Drafts actually sent to target verification.
    computed_draft_tokens: int = 0
    computed_query_tokens: int = 0
    draft_query_tokens: int = 0
    max_verify_draft_tokens: int = 0
    accepted_tokens: int = 0
    target_calls: int = 0
    draft_calls: int = 0
    compile_fallbacks: int = 0
    target_prefill_cycles: int | None = 0
    target_decode_cycles: int | None = 0
    target_prefill_runs: int = 0
    target_decode_runs: int = 0
    target_run_stats: list[dict] = field(default_factory=list)
    iterations: list[dict] = field(default_factory=list)

    @property
    def acceptance_rate(self):
        return self.accepted_tokens / self.proposed_tokens if self.proposed_tokens else 0.0

    @property
    def decode_tps(self):
        return self.decode_tokens / self.decode_seconds if self.decode_seconds else 0.0


@dataclass(frozen=True)
class Result:
    tokens: tuple[int, ...]
    stats: Stats
    stop_reason: str


class Decoder:
    """Single-stream, greedy-only speculative decoder.

    ``draft_tokens=3`` asks the target for four rows (anchor plus three drafts).
    ``draft_query_tokens`` defaults to the checkpoint's block_size (seven for
    released Nanbeige DSpark). The whole noncausal draft block is evaluated and
    only its first ``draft_tokens`` predictions are verified. Shortening the
    draft block changes even its prefix predictions; it is a distinct experiment.
    If those rows exceed TMEM/IMEM, compilation falls back to fewer drafts.
    Exceptions from device execution are deliberately not swallowed.
    """

    def __init__(self, target, draft, *, draft_tokens=3, prefill_chunk=4,
                 draft_query_tokens=None):
        if draft_tokens < 0 or prefill_chunk < 1:
            raise ValueError("draft_tokens must be nonnegative and prefill_chunk positive")
        if target.batch != 1 or target.stateful:
            raise ValueError("Nanbeige DSpark requires one non-stateful target stream")
        if target.spec.hidden != draft.spec.target_hidden:
            raise ValueError("DSpark target hidden width does not match the target")
        if target.spec.vocab != draft.target_vocab:
            raise ValueError("DSpark target vocabulary does not match the target")
        missing = set(draft.spec.capture_layers) - set(target.image.capture_slots)
        if missing:
            raise ValueError(f"Enable target capture_layers {sorted(missing)} before creating the decoder")
        self.target, self.draft = target, draft
        self.draft_tokens = min(int(draft_tokens), max(0, target.image.rows - 1))
        self.draft_query_tokens = (draft.spec.block_size if draft_query_tokens is None
                                   else draft_query_tokens)
        if (type(self.draft_query_tokens) is not int or self.draft_query_tokens < 1
                or self.draft_query_tokens < self.draft_tokens):
            raise ValueError("draft_query_tokens must be positive and cover the requested verification prefix")
        self.prefill_chunk = min(int(prefill_chunk), target.image.rows)
        self.context: Context | None = None
        self.stats: Stats | None = None

    def _commit_context(self, hidden, positions):
        before = time.perf_counter()
        more = self.draft.project_context(hidden, positions)
        self.context = more if self.context is None else self.context.append(more)
        self.stats.draft_seconds += time.perf_counter() - before

    def _run_target(self, tokens, *, all_logits=True):
        """Compile/run a block, shrinking it only on a compilation failure."""
        target, stats = self.target, self.stats
        start = target.pos
        n = len(tokens)
        stats_start = len(target.stats)
        before = time.perf_counter()
        while True:
            try:
                if n == 1:
                    logits = target.step(int(tokens[0]))[None, :]
                else:
                    rows = [(0, start + i) for i in range(n)]
                    lr = list(range(n)) if all_logits else [n - 1]
                    logits = target.run_rows(rows, list(tokens[:n]), lr)
                    # run_rows writes positional KV but deliberately leaves pos unchanged.
                    target.pos = start + n
                hidden = target.read_hidden(self.draft.spec.capture_layers, rows=n)
                break
            except CompileError:
                if n == 1:
                    raise
                stats.compile_fallbacks += 1
                n -= 1
        stats.target_seconds += time.perf_counter() - before
        stats.target_calls += 1
        raw = target.stats[stats_start:]
        stats.target_run_stats.extend(dict(run) if run is not None else {} for run in raw)
        values = [run.get("cycles") if run is not None else None for run in raw]
        cycles = sum(int(c) for c in values) if values and all(
            isinstance(c, (int, np.integer)) for c in values) else None
        name = f"target_{self._phase}_cycles"
        old = getattr(stats, name)
        setattr(stats, name, old + cycles if old is not None and cycles is not None else None)
        setattr(stats, f"target_{self._phase}_runs",
                getattr(stats, f"target_{self._phase}_runs") + len(raw))
        return n, logits, hidden

    def generate(self, prompt, *, max_new_tokens=64, eos_tokens=None) -> Result:
        prompt = tuple(_integer(t, "Prompt token ID") for t in prompt)
        if not prompt or any(t < 0 or t >= self.target.spec.vocab for t in prompt):
            raise ValueError("A nonempty prompt of target-vocabulary token IDs is required")
        max_new_tokens = _integer(max_new_tokens, "max_new_tokens")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        if len(prompt) > self.target.cap:
            raise ValueError("Prompt exceeds target KV capacity")
        eos = tuple(self.target.spec.eos if eos_tokens is None else eos_tokens)
        stats = Stats(type(self.target.backend).__name__, self.draft.backend,
                      prompt_tokens=len(prompt), draft_query_tokens=self.draft_query_tokens,
                      max_verify_draft_tokens=self.draft_tokens)
        self.stats, self.context = stats, None
        self._phase = "prefill"
        verify_rows = self.draft_tokens + 1
        started = time.perf_counter()
        self.target.reset()
        if max_new_tokens == 0:
            stats.total_seconds = time.perf_counter() - started
            return Result((), stats, "length")

        # Use explicit chunks so every committed prompt row's auxiliary state is read.
        at = 0
        while at < len(prompt):
            n, logits, hidden = self._run_target(
                prompt[at:at + self.prefill_chunk], all_logits=False)
            self._commit_context(hidden, np.arange(at, at + n))
            at += n
        stats.prefill_seconds = time.perf_counter() - started
        if not np.all(np.isfinite(logits[-1])):
            raise ValueError("Target prefill returned nonfinite logits")
        anchor = int(np.argmax(logits[-1]))
        output = [anchor]
        decode_start = time.perf_counter()
        self._phase = "decode"
        reason = "length"
        while len(output) < max_new_tokens:
            if anchor in eos:
                reason = "eos"
                break
            start = self.target.pos
            if start >= self.target.cap:
                reason = "capacity"
                break
            remaining = max_new_tokens - len(output)
            # Leave room for a correction/bonus; it becomes the next pending anchor.
            count = min(verify_rows - 1, remaining - 1, self.target.cap - start - 1)
            draft_before = time.perf_counter()
            if count:
                proposal = self.draft.draft(anchor, start, self.context, count,
                                           query_tokens=self.draft_query_tokens)
                proposed = [int(t) for t in proposal.tokens]
                if len(proposed) != count:
                    raise ValueError("DSpark returned the wrong proposal length")
                stats.draft_calls += 1
                stats.computed_draft_tokens += count
                stats.computed_query_tokens += self.draft_query_tokens
            else:
                proposed = []
            stats.draft_seconds += time.perf_counter() - draft_before
            n, logits, hidden = self._run_target([anchor, *proposed])
            if n < len(proposed) + 1:
                # A later context can only consume more attention storage. Avoid
                # drafting/compiling a block already known not to fit this run.
                verify_rows = min(verify_rows, n)
            proposed = proposed[:n - 1]
            verified = verify_greedy(np.asarray(proposed, dtype=np.int64), logits,
                                     max_tokens=remaining, eos_tokens=eos)
            # Inputs committed: anchor + accepted proposal tokens, not correction/bonus.
            committed = verified.accepted + 1
            end = start + committed
            if self.target.rollback(end) != end:
                raise RuntimeError("Target could not retain the exact speculative prefix")
            self._commit_context(hidden[:committed], np.arange(start, end))
            output.extend(verified.tokens)
            stats.proposed_tokens += len(proposed)
            stats.accepted_tokens += verified.accepted
            stats.iterations.append(dict(position=start, verify_rows=n,
                                         draft_queries=self.draft_query_tokens if count else 0,
                                         proposed=len(proposed), accepted=verified.accepted,
                                         emitted=len(verified.tokens), committed_position=end))
            anchor = output[-1]
            if anchor in eos:
                reason = "eos"
                break
        if anchor in eos:
            reason = "eos"
        stats.decode_seconds = time.perf_counter() - decode_start
        stats.total_seconds = time.perf_counter() - started
        stats.generated_tokens = len(output)
        stats.decode_tokens = len(output) - 1
        return Result(tuple(output), stats, reason)
