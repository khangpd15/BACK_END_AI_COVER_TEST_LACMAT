"""Descriptive timing evidence and optional isolated pretrained image branch."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np


def collect_research_evidence(trajectories, decoded):
    events = []
    for event_id, entry in decoded.items():
        event = entry["event"]
        samples = trajectories[event["cycle"]]["samples"]
        times = sorted({s["timestampMs"] for s in samples if isinstance(s.get("timestampMs"), (float, int))})
        deltas = np.diff(times)
        positive = deltas[deltas > 0]
        events.append({
            "eventId": event_id, "frameId": event["frameId"],
            "timestampMs": event["timestampMs"],
            "onsetIntervalMs": event["onsetIntervalMs"],
            "timingUncertaintyMs": event["timingUncertaintyMs"],
            "observedMedianFps": float(1000 / np.median(positive)) if len(positive) else None,
            "sourceFramesAfterUncover500Ms": sum(
                event["timestampMs"] <= t <= event["timestampMs"] + 500 for t in times),
        })
    result = {"status": "RESEARCH_ONLY", "clinicalDiagnosis": None, "timing": events,
              "imageBranch": {"status": "NOT_CONFIGURED"}}
    root, python = os.getenv("HIRSCHBERG_RESEARCH_ROOT"), os.getenv("HIRSCHBERG_RESEARCH_PYTHON")
    if root and python and decoded:
        payload = {"events": [{"eventId": event_id,
                              **{side: pixels.tolist() for side, pixels in entry["images"].items()}}
                             for event_id, entry in decoded.items()]}
        try:
            worker = Path(__file__).with_name("research_worker.py")
            completed = subprocess.run([python, "-B", str(worker), root],
                                       input=json.dumps(payload), text=True, capture_output=True,
                                       timeout=45, check=True, cwd=root)
            result["imageBranch"] = json.loads(completed.stdout.strip().splitlines()[-1])
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            result["imageBranch"] = {"status": "UNAVAILABLE", "reason": "RESEARCH_BRANCH_FAILED"}
    return result
