"""python -m swarmfpga train|generate|inspect --help"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
from . import SwarmModel, load_config
from .model import build


def main():
    parser = argparse.ArgumentParser(description="Minimal explicit training and inference")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("config")
    train.add_argument("dataset", help="integer .npy rows of sequence_length + 1 tokens")
    train.add_argument("checkpoint")
    train.add_argument("--resume")
    train.add_argument("--steps", type=int)
    generate = sub.add_parser("generate")
    generate.add_argument("checkpoint")
    generate.add_argument("--tokens", required=True, help="comma-separated token IDs")
    generate.add_argument("--max-new-tokens", type=int, default=8)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("config")
    args = parser.parse_args()
    if args.command == "train":
        config = load_config(args.config)
        if args.steps is not None and args.steps < 0:
            parser.error("steps must be a nonnegative integer")
        data = np.load(args.dataset, allow_pickle=False)
        with (SwarmModel.load(args.resume, expected_config=config) if args.resume else SwarmModel(config)) as model:
            for metrics in model.train(data, args.steps):
                print(json.dumps(metrics))
            model.save(args.checkpoint)
        print(json.dumps({"checkpoint": args.checkpoint, "numpy": np.__version__, "python": platform.python_version(),
                          "dataset_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()}))
    elif args.command == "generate":
        with SwarmModel.load(args.checkpoint) as model:
            print(json.dumps({"device": model.backend.device, "tokens": model.generate(
                [[int(x) for x in args.tokens.split(",")]], args.max_new_tokens).tolist()}))
    else:
        config = load_config(args.config)
        program = build(config, config.batch_size, config.sequence_length, training=True)
        print(json.dumps({"config": config.to_dict(), "nodes": len(program["nodes"]),
                          "required_operations": sorted({n["op"] for n in program["nodes"]})}, indent=2))


if __name__ == "__main__":
    main()
