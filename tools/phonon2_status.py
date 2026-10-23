#!/usr/bin/env python3
"""Print Phonon-2's fail-closed model status for Inferlab integration."""
import argparse
import json

from opentpu.asr.phonon2 import inferlab_model_status, model_status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inferlab", action="store_true", help="emit the existing Inferlab model-card fields")
    args = parser.parse_args()
    print(json.dumps(inferlab_model_status() if args.inferlab else model_status(), indent=2))
