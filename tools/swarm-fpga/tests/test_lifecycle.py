"""Ownership regressions for CLI entry points and the supported device selector."""
import builtins
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from swarmfpga.__main__ import main


@pytest.mark.parametrize("command", ["train", "generate"])
@pytest.mark.parametrize("fail", [False, True])
def test_cli_releases_owned_model_on_success_and_error(command, fail):
    model = MagicMock()
    model.__enter__.return_value = model
    # Model.__exit__ delegates cleanup to its owned backend; emulate that boundary.
    def exit_model(*args):
        model.close()
        return False
    model.__exit__.side_effect = exit_model
    model.backend.device = "reference"
    model.train.return_value = [{"loss": 1.0, "step": 1}]
    model.generate.return_value = np.array([[0, 1, 2]], dtype=np.int64)
    failure = RuntimeError("test execution failure")
    if fail:
        getattr(model, command).side_effect = failure
    args = (["train", "config.yaml", "rows.npy", "state.npz"] if command == "train"
            else ["generate", "state.npz", "--tokens", "0,1", "--max-new-tokens", "1"])
    with patch("sys.argv", ["swarmfpga", *args]), \
            patch("swarmfpga.__main__.SwarmModel", return_value=model) as model_class, \
            patch("swarmfpga.__main__.load_config", return_value=model.config), \
            patch("swarmfpga.__main__.np.load", return_value=np.array([[0, 1, 2]])), \
            patch.object(Path, "read_bytes", return_value=b"dataset fixture"):
        model_class.load.return_value = model
        if fail:
            with pytest.raises(RuntimeError, match="test execution failure"):
                main()
        else:
            main()
    model.__enter__.assert_called_once_with()
    model.__exit__.assert_called_once()
    model.close.assert_called_once_with()
    assert model.__exit__.call_args.args[0] is (RuntimeError if fail else None)


def test_native_rejects_other_slots_before_loading_board_transport():
    pytest.importorskip("opentpu", reason="optional OpenTPU dependency is not installed")
    from swarmfpga.opentpu import OpenTPUBackend
    original_import = builtins.__import__

    def reject_board_import(name, *args, **kwargs):
        if name.startswith("opentpu.host"):
            raise AssertionError("invalid device must fail before board transport imports")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=reject_board_import):
        with pytest.raises(ValueError):
            OpenTPUBackend(device="f2:1")
