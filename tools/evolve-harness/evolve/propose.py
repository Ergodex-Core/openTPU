"""Pure declarative generator: no filesystem, subprocess, credentials, or network access.

It can only produce settings from an exhaustive bounded allowlist. Evaluators, RTL,
fixtures, checkpoint/precision/context/length, and commands are never candidate fields.
"""
from itertools import product
from .common import digest, require

SPACE = {"program_cache": [True, False], "prompt_runs": [True, False], "quantum": [16, 32, 64]}
BASELINE = {"program_cache": True, "prompt_runs": True, "quantum": 32}


def validate(settings):
    require(type(settings) is dict and set(settings) == set(SPACE), "candidate has forbidden fields")
    for k, v in settings.items():
        require(any(type(v) is type(x) and v == x for x in SPACE[k]), "candidate value outside allowlist: " + k)
    return settings


def candidate_id(settings):
    validate(settings)
    return digest(settings)[:16]


def propose(parent, seen, generation):
    """Nearest unseen neighbour of archive parent, with deterministic generation tie-break."""
    validate(parent)
    choices = [dict(zip(SPACE, v)) for v in product(*SPACE.values())]
    choices = [x for x in choices if candidate_id(x) not in set(seen)]
    if not choices:
        return None
    choices.sort(key=lambda c: (sum(c[k] != parent[k] for k in SPACE), digest([generation, c])))
    c = choices[0]
    return {"id": candidate_id(c), "settings": c, "parent_id": candidate_id(parent)}
