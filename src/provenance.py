"""Mechanism (not a wish) for 'null and real use the same estimator'.

M2 records the hash of the estimator code it ran; M5 recomputes it and refuses
to compare against thresholds produced by different code.
"""
from __future__ import annotations

import hashlib
import os

ESTIMATOR_FILES = ["stats_engine.py", "discovery.py", "features.py", "targets.py",
                   "panel.py", "config.py"]


def estimator_fingerprint() -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    out = {}
    for f in ESTIMATOR_FILES:
        p = os.path.join(here, f)
        with open(p, "rb") as fh:
            out[f] = hashlib.sha256(fh.read()).hexdigest()[:16]
    joint = hashlib.sha256("|".join(f"{k}:{v}" for k, v in sorted(out.items())).encode())
    out["_joint"] = joint.hexdigest()[:16]
    return out
