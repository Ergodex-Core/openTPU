# Dependency notices

Ramulator 2.0a: https://github.com/CMU-SAFARI/ramulator2
Pinned commit: be93be78055d922aa1d4d33e15bcc8f2b0c61a9d (v2.0a peeled commit).
Copyright (c) 2023 SAFARI Research Group at ETH Zurich and Carnegie Mellon University.
MIT license, retained in `LICENSE.ramulator`.

The external build also uses yaml-cpp (MIT), spdlog (MIT), and argparse (MIT), pinned in
`../dependencies.lock.json`. Their complete upstream source and license files remain in
`SOURCE/ext/` when built. Distribute those licenses with any redistributed linked binaries.
No third-party binary is committed in this repository.

`prepare.py` performs three narrowly scoped changes: write-command callback, rank/pseudochannel
refresh selection, dependent-template syntax for Clang. Run output retains the prepared clone's
diff hash. This is a modified library, not a claim that upstream includes these changes.

SCALE-Sim is research reference only, MIT licensed, revision recorded in the lock file. No
SCALE-Sim code is copied. The OpenTPU implementation and bridge use Apache-2.0 as the repository.
