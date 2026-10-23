"""Timing instrumentation for a future trusted runtime bridge, without runtime/card access.

Call from the frozen evaluator's actual caller boundary. Do not call from candidate
generation. Time comes from perf_counter, not a candidate-supplied duration or TPS.
"""
import time
from .common import require


class TokenRecorder:
    def __init__(self, expected_count, clock=time.perf_counter):
        require(type(expected_count) is int and expected_count >= 2, "at least two output tokens required")
        self.clock=clock
        self.count=expected_count
        self.tokens=[]
        self.times=[]
        self.events={"request_start_s":clock()}
        self.prefill_tokens=0

    def prefill_start(self):
        require("prefill_start_s" not in self.events, "prefill started twice")
        self.events["prefill_start_s"]=self.clock()

    def prefill_done(self, tokens):
        require("prefill_start_s" in self.events and "prefill_done_s" not in self.events,
                "invalid prefill completion")
        require(type(tokens) is int and tokens>0,"actual timed prefill token count required")
        self.prefill_tokens=tokens
        self.events["prefill_done_s"]=self.clock()

    def deliver_accepted(self, token_id, deliver):
        """deliver performs the real host serialization/delivery; timestamp after it returns."""
        require("prefill_done_s" in self.events,"prefill boundary missing")
        require(type(token_id) is int and token_id>=0,"invalid accepted token ID")
        require(len(self.tokens)<self.count,"too many output tokens")
        deliver(token_id)
        self.tokens.append(token_id)
        self.times.append(self.clock())

    def done(self):
        require(len(self.tokens)==self.count,"incomplete accepted output")
        require("request_done_s" not in self.events,"request completed twice")
        self.events["request_done_s"]=self.clock()
        return {"events":{**self.events,"accepted_s":list(self.times)},
                "accepted_token_ids":list(self.tokens),"timed_prefill_tokens":self.prefill_tokens}
