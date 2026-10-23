"""Synthetic contract/integration tests. NO hardware, model or measured FPGA throughput."""
import copy
import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evolve.adapters import verify_window
from evolve.common import GateError, Journal, Queued, Uncertain, digest, immutable_json, read_json
from evolve.evaluate import decision, dominates, pareto, validate_trace, workload_key
from evolve.fixture_worker import execute
from evolve.loop import ROOT, initialize, load_run, reconcile, rollback, run_loop, status
from evolve.propose import BASELINE, candidate_id, propose, validate
from evolve.record import TokenRecorder


def config():
    return read_json(ROOT/"configs/fixture.json")


def trace(c=None):
    c = c or config()
    w = c["workloads"][0]
    return execute({"workload": w, "seed": c["seed"], "heldout": c["heldout"], "settings": BASELINE,
                    "pair": 0, "workload_sha256": workload_key(w)})


class Gates(unittest.TestCase):
    def test_recorder_includes_actual_delivery(self):
        values=iter([1.,1.01,1.1,1.2,1.3,1.31])
        r=TokenRecorder(2,clock=lambda:next(values))
        r.prefill_start();r.prefill_done(4)
        delivered=[]
        r.deliver_accepted(5,delivered.append);r.deliver_accepted(7,delivered.append)
        t=r.done()
        self.assertEqual(delivered,[5,7])
        self.assertEqual(t["events"]["accepted_s"],[1.2,1.3])
        self.assertEqual(t["events"]["request_done_s"],1.31)

    def test_recorder_rejects_early_stop(self):
        r=TokenRecorder(2)
        with self.assertRaises(GateError):r.done()

    def test_fixture_numbers_are_never_physical(self):
        c=config()
        with self.assertRaisesRegex(GateError,"wrong evidence stage"):
            validate_trace(trace(c),c["workloads"][0],c["seed"],c["heldout"],fixture=False)

    def test_metrics_count_accepted_tokens_and_intervals(self):
        c=config(); w=c["workloads"][0]; t=trace(c)
        m=validate_trace(t,w,c["seed"],c["heldout"],fixture=True)
        self.assertAlmostEqual(m["wall_decode_tps"], 40.04)
        self.assertAlmostEqual(m["device_decode_tps"], 60.)
        self.assertLess(m["request_output_tps"],m["wall_decode_tps"])

    def test_wrong_tokens_counts_precision_context_backend_stage(self):
        c=config(); w=c["workloads"][0]
        mutations=[lambda t:t.update(accepted_token_ids=t["accepted_token_ids"][:-1]),
                   lambda t:t["accepted_token_ids"].__setitem__(1,999),
                   lambda t:t["identity"].update(weights="fp4"),
                   lambda t:t.update(loop_count=2),lambda t:t.update(backend="isa"),
                   lambda t:t.update(initial_context_tokens=1),lambda t:t.update(cpu_fallback=True),
                   lambda t:t.update(complete=False),lambda t:t.update(heldout_pass=False),
                   lambda t:t["heldout_results"][0].update(accepted_token_ids=[7]),
                   lambda t:t.update(state="cold_process"),
                   lambda t:t["events"].update(prefill_done_s=float("nan")),
                   lambda t:t["events"]["accepted_s"].__setitem__(2,0),
                   lambda t:t["device"].update(decoded_tokens=999)]
        for change in mutations:
            with self.subTest(change=change):
                t=copy.deepcopy(trace(c)); change(t)
                with self.assertRaises(GateError): validate_trace(t,w,c["seed"],c["heldout"],fixture=True)

    def test_generator_cannot_change_evaluator_or_actions(self):
        for key in ("command","path","precision","output_tokens","rtl","fixture"):
            with self.assertRaises(GateError): validate({**BASELINE,key:"bad"})
        with self.assertRaises(GateError): validate({**BASELINE,"quantum":True})
        seen=[candidate_id(BASELINE)]
        for n in range(11):
            c=propose(BASELINE,seen,n)
            self.assertNotIn(c["id"],seen);seen.append(c["id"])
        self.assertIsNone(propose(BASELINE,seen,12))

    def test_speculation_exact_greedy_and_rollback_required(self):
        c=config(); w=c["workloads"][0]; w["speculation"]=True
        c["seed"]["speculation_qualified"]=True
        c["heldout"]["rollback_cases"]=[{"id":"reject-after-one","committed_state_sha256":"0"*64,
                                          "continuation_token_ids":[1,2,3]}]
        c["seed"]["correctness_suite_sha256"]=digest(c["heldout"])
        t=trace(c)
        t["speculation"]={"verification":"exact_greedy","rollback_pass":True,
            "rollback_results":copy.deepcopy(c["heldout"]["rollback_cases"]),
            "reference_token_ids":t["accepted_token_ids"],"accepted_output_tokens":w["output_tokens"],
            "draft_s":.01,"verify_s":.01,"accept_s":.001,"rollback_s":0,
            "proposed_draft_tokens":16,"accepted_draft_tokens":4}
        validate_trace(t,w,c["seed"],c["heldout"],fixture=True)
        for bad in ({"rollback_pass":False},{"rollback_results":[]},{"verification":"sampled_unverified"},
                    {"accepted_output_tokens":20},{"accepted_draft_tokens":17},{"accept_s":999}):
            q=copy.deepcopy(t);q["speculation"].update(bad)
            with self.assertRaises(GateError): validate_trace(q,w,c["seed"],c["heldout"],fixture=True)


class Decisions(unittest.TestCase):
    def rows(self, gain=1.15,ttft=.95,prefill=1.05):
        return [{"case_id":"warm","a":{"wall_decode_tps":40.,"ttft_s":.5,"prefill_tps":80.},
                 "b":{"wall_decode_tps":40*gain,"ttft_s":.5*ttft,"prefill_tps":80*prefill}}
                for _ in range(8)]

    def test_improvement_promotes(self):
        self.assertTrue(decision(self.rows(),config()["policy"],1)["promote"])

    def test_latency_and_prefill_regressions_reject_despite_faster_decode(self):
        for kw in ({"ttft":1.2},{"prefill":.7},{"gain":1.001}):
            self.assertFalse(decision(self.rows(**kw),config()["policy"],1)["promote"])

    def test_noise_and_incomplete_pairs_reject(self):
        rows=self.rows(); rows[0]["a"]["wall_decode_tps"]=20
        self.assertFalse(decision(rows,config()["policy"],1)["archive_eligible"])
        with self.assertRaises(GateError):decision(rows[:3],config()["policy"],1)

    def test_pareto_retains_tradeoff(self):
        a=decision(self.rows(),config()["policy"],1)
        b=decision(self.rows(gain=1.4,ttft=1.2),config()["policy"],1)
        self.assertFalse(dominates(a,b)); self.assertFalse(dominates(b,a))
        self.assertEqual(len(pareto([{"id":"a","result":a},{"id":"b","result":b}])),2)


class Ownership(unittest.TestCase):
    def test_pending_portal_always_queues(self):
        with self.assertRaises(Queued): verify_window({"schema":1,"access":"authorized","portal_pending":True}, {})

    def test_historical_restore_is_not_current_ownership(self):
        with self.assertRaises(Queued): verify_window({"schema":1,"access":"unknown"}, {})

    def test_expired_window_rejected(self):
        now=dt.datetime.now(dt.timezone.utc)
        w={"schema":1,"access":"authorized","portal_pending":False,"other_owner_active":False,
           "instance_name":"opentpu-f2","instance_id":"i-0c78de2253f66519d",
           "iam_profile":"opentpu-f2-ssm","device_lock":"/tmp/otpu/f2-0000:34:00.0.lock",
           "exclusive":True,"loaded_image":"agfi-test","verified_at":(now-dt.timedelta(minutes=3)).isoformat(),
           "expires_at":(now+dt.timedelta(minutes=2)).isoformat(),"owner":"test","receipt_sha256":"0"*64}
        with self.assertRaisesRegex(GateError,"stale"):verify_window(w,{"agfi":"agfi-test"},now)


class Integration(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name); self.run=self.root/"run"
        self.cfg=self.root/"config.json"; self.cfg.write_text(json.dumps(config()))
        initialize(self.cfg,self.run)

    def tearDown(self): self.tmp.cleanup()

    def test_dry_run_resume_no_duplicate_and_rollback(self):
        first=run_loop(self.run,1)
        self.assertEqual(first["completed_candidates"],1)
        self.assertEqual(first["trials"],32)
        second=run_loop(self.run,2)
        self.assertEqual(second["completed_candidates"],2)
        self.assertEqual(second["trials"],64)
        self.assertEqual(run_loop(self.run,2)["trials"],64)
        rolled=rollback(self.run,"test explicit rollback")
        self.assertEqual(rolled["kind"],"TEST_FIXTURE")
        self.assertEqual(rolled["last_event"],"rollback")

    def test_evidence_tamper_blocks_resume(self):
        run_loop(self.run,1)
        p=next((self.run/"evidence").glob("*.json"));p.chmod(0o644);p.write_text("{}")
        with self.assertRaisesRegex(GateError,"evidence changed"):load_run(self.run)

    def test_frozen_suite_and_evaluator_tampering_block(self):
        m=self.run/"manifest.json";x=read_json(m);x["config"]["heldout"]["cases"][0]["expected_token_ids"][0]=9
        m.chmod(0o644);m.write_text(json.dumps(x))
        with self.assertRaises(GateError):load_run(self.run)

    def test_crash_blocks_implicit_retry_and_owner_reconciliation_rejects(self):
        from evolve.propose import propose
        c=propose(BASELINE,[candidate_id(BASELINE)],0)
        j=Journal(self.run/"journal");j.append("candidate",c)
        key=c["id"]+"-warm-00-a"
        j.append("trial_started",{"key":key,"candidate_id":c["id"]})
        with self.assertRaises(Uncertain):run_loop(self.run,1)
        receipt=self.root/"resolved.json"
        receipt.write_text(json.dumps({"trial_id":key,"outcome":"aborted_idle_runtime_restored",
                                       "owner":"TEST_FIXTURE","source_receipt_sha256":"0"*64}))
        reconcile(self.run,key,receipt)
        self.assertEqual(run_loop(self.run,1)["completed_candidates"],1)
        self.assertEqual(status(self.run)["trials"],0)

    def test_live_config_queues_without_spawning(self):
        c=config();c["mode"]="live";c["ownership_window"]={"schema":1,"access":"unknown"}
        path=self.root/"live.json";path.write_text(json.dumps(c));r=self.root/"live"
        initialize(path,r)
        with patch("evolve.adapters.subprocess.Popen",side_effect=AssertionError("must not execute")):
            s=run_loop(r)
        self.assertEqual(s["last_event"],"queued")
        self.assertFalse(s["last_data"]["hardware_attempted"])

    def test_immutable_journal_no_overwrite(self):
        with self.assertRaises(FileExistsError):immutable_json(self.run/"manifest.json",{})
        p=self.run/"journal/000000.json";p.chmod(0o644);p.write_text("{}")
        with self.assertRaises((GateError,KeyError)):load_run(self.run)


if __name__=="__main__":unittest.main()
