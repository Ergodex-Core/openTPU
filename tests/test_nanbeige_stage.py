"""The preparation tool cannot execute model or production-control requests."""
import io
import json
from pathlib import Path
import runpy
import sys

import pytest

from tools import nanbeige_stage as cli


def test_hold_retains_models_and_only_serves_status_or_discard(capsys):
    receipt = dict(status="prepared_host_ram", hardware_executed=False,
                   model_forward_executed=False, outage_budget_qualified=False)
    commands = [{"action": "upload"}, {"action": "validate"}, {"action": "maintenance"},
                {"action": "status", "id": "a"}, {"action": "status", "extra": True},
                {"action": "discard", "id": "b"}, {"action": "status", "id": "after"}]
    cli.hold((object(), object()), receipt, io.StringIO("\n".join(map(json.dumps, commands))))
    out = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [x["event"] for x in out] == ["error", "error", "error", "status", "error", "discarded"]
    assert all(x["hardware_executed"] is False for x in out)
    assert out[3]["id"] == "a" and not out[3]["outage_budget_qualified"]


def test_cli_has_no_hardware_execution_or_maintenance_command(tmp_path, monkeypatch, capsys):
    def fake(*args, **kw):
        return (object(), object()), dict(status="prepared_host_ram", hardware_executed=False,
            model_forward_executed=False, old_300_second_harness_allowed=False,
            outage_budget_qualified=False)
    monkeypatch.setattr(cli, "prepare_pair", fake)
    target = tmp_path / "receipt.json"
    argv = ["--nanbeige-model-dir", "n", "--gemma-model-dir", "g",
            "--nanbeige-oracle", "nr", "--gemma-oracle", "gr", "--json", str(target)]
    assert cli.main(argv) == 0
    assert json.loads(target.read_text())["hardware_executed"] is False
    assert json.loads(capsys.readouterr().out)["event"] == "ready"
    with pytest.raises(FileExistsError):
        cli.main(argv)
    with pytest.raises(SystemExit):
        cli.main(argv + ["--infer"])


def test_incompatible_profile_fails_before_checkpoint_or_tokenizer_work(monkeypatch):
    monkeypatch.setattr(cli.imgcache, "checkpoint", lambda p: pytest.fail("checkpoint read"))
    with pytest.raises(ValueError, match="wide-port"):
        cli.prepare_pair("absent", "absent", "absent", "absent",
                         profile="tools/data/nanbeige/burst200.toml")


def test_wrong_checkpoint_oracle_fails_before_packing(tmp_path, monkeypatch):
    oracle = tmp_path / "oracle.json"
    oracle.write_text(json.dumps(dict(status="complete", model={"repo_id": "wrong"})))
    monkeypatch.setattr(cli, "prepare_nanbeige", lambda *a, **k: pytest.fail("packed model"))
    monkeypatch.setattr(cli.imgcache, "checkpoint", lambda p: None)
    with pytest.raises(ValueError, match="pinned 4.2 3B"):
        cli.prepare_pair(tmp_path, tmp_path, oracle, oracle)


def test_legacy_300_second_harness_exits_before_any_action(monkeypatch):
    script = Path(__file__).resolve().parents[1] / "docs/data/nanbeige_wideport/physical-gate.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--window", "never-authorized"])
    with pytest.raises(SystemExit, match="disallowed"):
        runpy.run_path(str(script), run_name="__main__")
