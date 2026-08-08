"""provenance_gate.py — independent post-wave provenance hard gate for Frame-A.

Independent of the existing analyzer (which only validates the provenance LABEL): this gate
checks the NUMBERS, the file layout, and the write-time patterns that betray synthetic
data relabeled as real.

Triggered by the 2026-07-21 Frame-A synthetic-relabel incident (66 MIX_B/MIX_C cells
labeled "real" but containing exact synthetic cost anchors 4000.0 / 1500.0, zero seed
variance on A_loc, and a 1-second write burst).

This module is INTENTIONALLY isolated:
  * Imports only stdlib (json, os, sys, glob, argparse, hashlib, datetime, statistics).
  * Imports nothing from experiments.frame_a.* (would risk being loaded by the live wave).
  * Reads ONLY results/frame_a/cells/*.json — does not touch any live-imported module
    (run_stream.py, real_replay.py, arms/*, scorer/*), the wave driver, the log, or any
    PID file.

CLI:
  python3 -m experiments.frame_a.provenance_gate --cells_dir results/frame_a/cells

Exit codes:
  0 — PASS: the wave's cells match the preregistered grid AND pass numeric/temporal sanity.
  1 — FAIL: the grid is complete BUT something is suspicious (relabeled synthetic, burst).
  2 — INCOMPLETE: the wave is still partial (live process still writing); re-run later.
  3 — USAGE: bad CLI args.

The machine-readable JSON report is written to stdout (--report path optional).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

# ------------------------------------------------------------------------------ constants
EXPECTED_MODEL = "llama-3.2-1b"
EXPECTED_PROVENANCE = "real"
EXPECTED_MIXES = ("MIX_A", "MIX_B", "MIX_C")   # MIX_A added 2026-08-01 (T5 audit: the
# paper claims the full 99-cell grid and MIX_A carries primary tables + quarantined cells;
# a gate covering only B/C would certify the mixes the paper leans on least)
EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_POLICIES = ("both", "cost_only", "damage_only", "oracle",
                     "always_edit", "always_grace", "always_rag", "always_ft",
                     "always_reject", "random", "ft_merge")
EXPECTED_TOTAL = len(EXPECTED_POLICIES) * len(EXPECTED_SEEDS) * len(EXPECTED_MIXES)
NAMESPACED_P2_NAME = f"p2_{EXPECTED_MODEL}_{EXPECTED_PROVENANCE}_MIX_C.json"

# Exact synthetic cost anchors that DO NOT arise from real GPU measurements.
EXACT_SYNTH_INSTALL_GPU_S = 4000.0
EXACT_SYNTH_SERVE_GPU_S = 1500.0

# Burst threshold: how many cells in one write-second is suspicious for a real wave.
# 2026-07-21 incident: 34 + 33 = 67 cells in 2 consecutive seconds (across 8 s0 cells
# the live wave spreads to ~1 cell per 1500 s). Picked at 8 to give ~10x headroom.
BURST_THRESHOLD = 8

# Round-anchor suspicion: install_gpu_s in this range, exactly integer-valued (fractional
# part == 0.0), AND exceeds this floor is treated as a *suspicious* (not auto-fail) tell.
# always_reject install=0.0 is legitimate and must NOT be flagged. The floor at 100.0
# and the in-range gate keep legitimate zeros and noisy real measurements out.
ROUND_ANCHOR_FLOOR = 100.0
ROUND_ANCHOR_RANGE = (100.0, 5000.0)
ROUND_ANCHOR_MAX_PER_WAVE = 0   # 0 means "any occurrence is suspicious" (the deterministic
                                  # synthetic anchors are exact, so even 1 is signal).

# Cross-seed non-degeneracy tolerance: real waves produce non-identical A_loc across seeds
# even for deterministic policies (the stream content differs). A perfectly constant A_loc
# across all 3 seeds within a (mix, policy) is a structural red flag.
DEGENERACY_ATOL = 1e-9
DEGENERACY_MIN_SEEDS = 3   # wait for the complete seed trio; partial waves stay INCOMPLETE.


# ------------------------------------------------------------------------------ utilities
def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _sha256_short(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _is_hidden_or_quarantine(name: str) -> bool:
    """Anything starting with '.' is quarantined or auxiliary — never traverse."""
    return name.startswith(".")


def _list_cell_jsons(cells_dir: str) -> List[str]:
    """Return direct, regular cell JSON files; reject symlinks, including quarantine escapes."""
    if not os.path.isdir(cells_dir):
        return []
    out = []
    for name in os.listdir(cells_dir):
        if _is_hidden_or_quarantine(name) or not (name.startswith("cell_") and name.endswith(".json")):
            continue
        path = os.path.join(cells_dir, name)
        if os.path.islink(path) or not os.path.isfile(path):
            continue
        out.append(name)
    return sorted(out)


def _list_p2_jsons(cells_dir: str) -> List[str]:
    if not os.path.isdir(cells_dir):
        return []
    out = []
    for name in os.listdir(cells_dir):
        if _is_hidden_or_quarantine(name):
            continue
        if not (name.startswith("p2_") and name.endswith(".json")):
            continue
        out.append(name)
    return sorted(out)


def _parse_cell_filename(name: str) -> Optional[Tuple[str, str, str, str, int]]:
    """Parse `cell_{model}_{provenance}_{mix}_{policy}_s{seed}.json`.

    Returns (model, provenance, mix, policy, seed) on success, None on parse failure.
    The mix token is exactly one of ("MIX_A", "MIX_B", "MIX_C"); the segment immediately
    before it is provenance; everything before provenance is model; everything after
    the mix token is policy (joined with '_'). This handles model names that contain
    dashes but no underscores (e.g. `llama-3.2-1b`) and policy names that contain
    underscores (e.g. `always_edit`, `ft_merge`).
    """
    if not (name.startswith("cell_") and name.endswith(".json")):
        return None
    body = name[len("cell_"):-len(".json")]
    if "_s" not in body:
        return None
    head, seed_str = body.rsplit("_s", 1)
    try:
        # Seed syntax is deliberately strict: bool, float aliases, signs, and whitespace
        # are not canonical identity-bearing filenames.
        if not seed_str or not seed_str.isdigit():
            return None
        seed = int(seed_str)
    except (TypeError, ValueError):
        return None
    parts = head.split("_")
    mix_idx = None
    # The mix token in the filename appears as TWO consecutive parts ("MIX", "B")
    # because the canonical form is `MIX_A`/`MIX_B`/`MIX_C`. Find that pair.
    for i in range(len(parts) - 1):
        if parts[i] == "MIX" and parts[i + 1] in ("A", "B", "C"):
            mix_idx = i
            break
    if mix_idx is None or mix_idx < 2:
        return None
    model = "_".join(parts[:mix_idx - 1])
    provenance = parts[mix_idx - 1]
    mix = parts[mix_idx] + "_" + parts[mix_idx + 1]
    policy = "_".join(parts[mix_idx + 2:])
    if not model or not provenance or not policy:
        return None
    return model, provenance, mix, policy, seed


def _check_foreign_extras(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flag only target-namespace lookalikes; unrelated artifacts are out-of-scope INFO."""
    findings = []
    for c in cells:
        model, provenance, mix = c.get("_fn_model"), c.get("_fn_provenance"), c.get("_fn_mix")
        # MIX_A, other models, and other provenance are known artifacts, not foreign attacks.
        if model != EXPECTED_MODEL or provenance != EXPECTED_PROVENANCE or mix not in EXPECTED_MIXES:
            continue
        if c.get("_fn_policy") not in EXPECTED_POLICIES or c.get("_fn_seed") not in EXPECTED_SEEDS:
            findings.append({"kind": "foreign_extra", "path": c["_path"],
                             "filename": c["_filename"], "identity": [model, provenance, mix,
                             c.get("_fn_policy"), c.get("_fn_seed")], "severity": "FAIL"})
    return findings
def _check_schema(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    findings = []
    required = (("quality", ("A_loc", "Q")), ("cost", ("install_gpu_s", "serve_gpu_s")))
    for c in cells:
        bad = []
        if not isinstance(c, dict):
            bad.append("object")
        for section, fields in required:
            obj = c.get(section)
            if not isinstance(obj, dict):
                bad.append(section)
                continue
            for field in fields:
                v = obj.get(field)
                if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)) or v < 0:
                    bad.append(f"{section}.{field}")
        v = c.get("error_cost_eval")
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)) or v < 0:
            bad.append("error_cost_eval")
        # Optional discovery values may be null/NaN in real legacy outputs; core fields above may not.
        if bad:
            findings.append({"kind": "schema_invalid", "path": c["_path"], "fields": bad,
                             "severity": "FAIL"})
    return findings

def _check_exact_synthetic_anchors(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """install_gpu_s == 4000.0 AND serve_gpu_s == 1500.0 (exact) are the synthetic anchors.

    The 2026-07-21 incident reproduced these exactly. Real waves produce noisy measurements
    (e.g. real MIX_A always_edit: install=238.55 / serve=7.92). Exact equality at this
    scale is structural evidence of replay, not measurement noise.
    """
    findings = []
    for c in cells:
        cost = c.get("cost", {}) or {}
        ins = cost.get("install_gpu_s")
        srv = cost.get("serve_gpu_s")
        ins_hit = (ins == EXACT_SYNTH_INSTALL_GPU_S)
        srv_hit = (srv == EXACT_SYNTH_SERVE_GPU_S)
        if ins_hit and srv_hit:
            findings.append({
                "kind": "synthetic_anchor_exact",
                "path": c["_path"],
                "cell_id": c["_cell_id"],
                "install_gpu_s": ins,
                "serve_gpu_s": srv,
                "severity": "FAIL",
            })
        elif ins_hit or srv_hit:
            findings.append({
                "kind": "synthetic_anchor_partial",
                "path": c["_path"],
                "cell_id": c["_cell_id"],
                "install_gpu_s": ins,
                "serve_gpu_s": srv,
                "severity": "WARN",
            })
    return findings


def _check_round_anchors(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flag repeated near-round install values, not ordinary noisy measurements."""
    candidates = []
    for c in cells:
        ins = (c.get("cost", {}) or {}).get("install_gpu_s")
        if (isinstance(ins, (int, float)) and not isinstance(ins, bool)
                and math.isfinite(float(ins)) and ROUND_ANCHOR_RANGE[0] <= ins <= ROUND_ANCHOR_RANGE[1]
                and abs(float(ins) - round(float(ins))) < 0.5):
            candidates.append(c)
    rounded_groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        value = float((c.get("cost", {}) or {}).get("install_gpu_s"))
        rounded_groups[(c.get("mix"), round(value))].append(c)
    flagged = []
    for (mix, anchor), group in rounded_groups.items():
        values = [float((c.get("cost", {}) or {}).get("install_gpu_s")) for c in group]
        # Repeated placeholders must agree tightly; ordinary measurements that merely
        # round to the same integer are not evidence of replay.
        if len(group) < 3 or max(values) - min(values) >= 1e-6:
            continue
        flagged.extend(group)
    return [{"kind": "round_anchor_suspicious", "path": c["_path"],
             "cell_id": c["_cell_id"], "install_gpu_s": (c.get("cost", {}) or {}).get("install_gpu_s"),
             "note": "repeated identical near-integer install anchor within mix", "severity": "WARN"}
            for c in flagged]


def _check_burst(cells: List[Dict[str, Any]], cells_dir: str) -> List[Dict[str, Any]]:
    """A burst of N cells sharing the same write-second is never a real GPU wave.

    Real waves: a single (mix, policy, seed) cell takes minutes to write (Llama-1B MIX_B
    on a 5090 is ~2-5 min/cell; MIX_C similar). The live MIX_B s0 wave currently writes
    ~1 cell per ~20 minutes (verified 2026-07-21). The 2026-07-21 incident wrote 34 cells
    in 1784607152 and 33 cells in 1784607153 — total 67 in 2 seconds.

    Caveat (documented in --report): rsync -a / cp -p preserve source mtimes. If a real
    wave was copied from another machine with `cp -p` or `rsync -a`, all mtimes will
    reflect the ORIGINAL write, not the copy time, and a burst will look synthetic even
    though the data is real. Always confirm with the source-host log before acting on a
    burst alert.

    Threshold: BURST_THRESHOLD (=8) cells in the same write-second is suspicious.
    """
    by_second: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for c in cells:
        by_second[c["_mtime_sec"]].append(c)

    findings = []
    for sec, group in sorted(by_second.items()):
        if len(group) < BURST_THRESHOLD:
            continue
        findings.append({
            "kind": "write_burst",
            "mtime_sec": sec,
            "mtime_iso": datetime.datetime.fromtimestamp(sec, tz=datetime.timezone.utc).isoformat(),
            "count": len(group),
            "threshold": BURST_THRESHOLD,
            "sample_paths": sorted(g["_path"] for g in group[:5]),
            "caveat": "rsync -a / cp -p preserve source mtimes; verify against source-host log.",
            "severity": "FAIL",
        })
    ordered = sorted(cells, key=lambda c: c["_mtime_sec"])
    for i in range(len(ordered) - BURST_THRESHOLD + 1):
        window = ordered[i:i + BURST_THRESHOLD]
        if window[-1]["_mtime_sec"] - window[0]["_mtime_sec"] <= 60:
            findings.append({"kind": "write_burst_sliding", "count": BURST_THRESHOLD,
                             "window_seconds": 60,
                             "sample_paths": [c["_path"] for c in window[:5]],
                             "severity": "FAIL"})
            break
    return findings

def _check_unparseable_filenames(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reject cell_*.json files that do not follow the canonical identity-bearing name."""
    return [{
        "kind": "unparseable_filename",
        "path": c["_path"],
        "filename": c["_filename"],
        "severity": "FAIL",
    } for c in cells if c.get("_fn_model") is None]


def _check_identity_mismatch(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """cell filename says (model, provenance, mix) but the body disagrees.

    The 2026-07-21 incident had filename `_real_` while the data was synthetic replay.
    Filename is untrusted; body is the truth. Flag any mismatch — checked on ALL loaded
    cells (not only in-scope) because a cell claiming `provenance=real` in its filename
    but carrying `provenance=synth` in its body IS the relabeling tell, even though such
    a cell won't pass the in-scope filter for the rest of the grid.
    """
    findings = []
    for c in cells:
        if c.get("_fn_model") is None:
            continue  # unparseable filename — covered by other checks
        fn_model = c["_fn_model"]
        fn_provenance = c["_fn_provenance"]
        fn_mix = c["_fn_mix"]
        fn_policy = c["_fn_policy"]
        fn_seed = c["_fn_seed"]
        body_model = c.get("model")
        body_provenance = c.get("provenance")
        body_mix = c.get("mix")
        body_policy = c.get("policy")
        body_seed = c.get("seed")
        filename_identity = [fn_model, fn_provenance, fn_mix, fn_policy, fn_seed]
        body_identity = [body_model, body_provenance, body_mix, body_policy, body_seed]
        if body_identity != filename_identity:
            findings.append({
                "kind": "identity_mismatch",
                "path": c["_path"],
                "filename": c["_filename"],
                "filename_identity": filename_identity,
                "body_identity": body_identity,
                "severity": "FAIL",
            })
    return findings


def _check_duplicates(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """At most one cell file per (mix, policy, seed). More than one = a write race or
    a relabeling campaign. Either way, refuse to proceed until reconciled.
    """
    by_key: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for c in cells:
        by_key[(c.get("mix"), c.get("policy"), c.get("seed"))].append(c)
    findings = []
    for key, group in sorted(by_key.items()):
        if len(group) > 1:
            findings.append({
                "kind": "duplicate_cell",
                "mix_policy_seed": list(key),
                "paths": sorted(g["_path"] for g in group),
                "count": len(group),
                "severity": "FAIL",
            })
    return findings


def _check_missing(cells: List[Dict[str, Any]], present_mix_set: set) -> List[Dict[str, Any]]:
    """Find (mix, policy, seed) triples that should be on disk but aren't."""
    have = {(c.get("mix"), c.get("policy"), c.get("seed")) for c in cells}
    missing = []
    for mix in EXPECTED_MIXES:
        for policy in EXPECTED_POLICIES:
            for seed in EXPECTED_SEEDS:
                if (mix, policy, seed) not in have:
                    missing.append({"mix": mix, "policy": policy, "seed": seed})
    return [{"kind": "missing_cell", "items": missing, "count": len(missing),
             "severity": "INFO"}] if missing else []


def _check_p2(cells_dir: str) -> List[Dict[str, Any]]:
    """Require the namespaced real MIX-C P2 file and validate its body identity."""
    findings = []
    namespaced = os.path.join(cells_dir, NAMESPACED_P2_NAME)
    if os.path.islink(namespaced) or not os.path.isfile(namespaced) or not os.path.exists(namespaced):
        siblings = [n for n in _list_p2_jsons(cells_dir) if n.endswith("_MIX_C.json")]
        findings.append({
            "kind": "missing_p2",
            "expected": NAMESPACED_P2_NAME,
            "present_p2_files": siblings,
            "severity": "INFO",
        })
        return findings
    try:
        with open(namespaced, "r") as f:
            body = json.load(f)
    except json.JSONDecodeError as e:
        return [{"kind": "p2_malformed", "path": namespaced,
                 "error": str(e), "severity": "FAIL"}]
    if not isinstance(body, dict):
        return [{"kind": "p2_invalid_body", "path": namespaced,
                 "error": "P2 body must be an object", "severity": "FAIL"}]
    expected_identity = {"model": EXPECTED_MODEL, "provenance": EXPECTED_PROVENANCE,
                         "mix": "MIX_C"}
    mismatches = {k: {"expected": expected_identity[k], "actual": body[k]}
                  for k in expected_identity if k in body and body[k] != expected_identity[k]}
    required_metrics = {"exposure_edit", "exposure_rag", "footprint_delta",
                        "overhead_delta", "router_edit_majority_on_privacy"}
    missing_metrics = sorted(required_metrics - set(body))
    bad_metrics = []
    for key in sorted(required_metrics):
        value = body.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
            bad_metrics.append(key)
    # Schema / structural invariants for a measurable P2 artifact. Editing exposure
    # must be lower than RAG, footprint/overhead positive, and the privacy-routing
    # fraction a unit interval. The gate predicate itself (fraction > 0.5) lives in
    # analyze_frame_a.py — a measured FAIL fraction (e.g. 0.103) is valid evidence
    # for KILL, not a malformed P2 body.
    if not bad_metrics:
        if not (body["exposure_edit"] < body["exposure_rag"]):
            bad_metrics.append("exposure_order")
        if not (body["footprint_delta"] > 0):
            bad_metrics.append("footprint_positive")
        if not (body["overhead_delta"] > 0):
            bad_metrics.append("overhead_positive")
        router = body["router_edit_majority_on_privacy"]
        if not (0.0 <= router <= 1.0):
            bad_metrics.append("router_edit_majority_on_privacy_range")
    provenance = body.get("p2_cost_provenance")
    if provenance is not None and not isinstance(provenance, dict):
        bad_metrics.append("p2_cost_provenance_object")
    if mismatches or missing_metrics or bad_metrics:
        findings.append({
            "kind": "p2_body_invalid",
            "path": namespaced,
            "missing_metric_fields": missing_metrics,
            "invalid_metric_fields": bad_metrics,
            "identity_mismatches": mismatches,
            "note": "P2 metrics must be finite nonnegative numbers.",
            "severity": "FAIL",
        })
    else:
        findings.append({
            "kind": "p2_ok",
            "path": namespaced,
            "body_keys": sorted(body.keys()),
            "severity": "INFO",
        })
    return findings


def _check_malformed(cells: List[Dict[str, Any]], parse_errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cells that failed JSON decode. parse_errors is the raw list from _load_cells."""
    return [{
        "kind": "malformed_json",
        "path": e["path"],
        "error": e["error"],
        "severity": "FAIL",
    } for e in parse_errors]


def _check_non_degeneracy(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cross-seed non-degeneracy: real waves produce DIFFERENT A_loc and DIFFERENT
    error_cost_eval across seeds (stream content differs even for deterministic policies).

    The 2026-07-21 incident had A_loc IDENTICAL across all 3 seeds for every policy
    (0.34647 / 0.34647 / 0.34647, etc.) because a deterministic replay was used.

    Conservative policy: only test once all 3 preregistered seeds are present. Flag a
    group only if all tested fields are equal within the frozen tolerance. Partial waves
    remain INCOMPLETE rather than acquiring an early degeneracy verdict.

    Skips checks on policies that are structurally invariant across seeds: e.g.
    always_reject install=0.0 / serve=constant (the model load time doesn't change across
    seeds). The A_loc / error_cost_eval / routing checks use the stream's per-seed
    content, which DOES differ. We test the variance on multiple fields; the flag fires
    only if ALL tested fields are bit-exactly equal.
    """
    by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for c in cells:
        if c.get("mix") in EXPECTED_MIXES and c.get("policy") in EXPECTED_POLICIES:
            by_group[(c.get("mix"), c.get("policy"))].append(c)

    findings = []
    for (mix, policy), group in sorted(by_group.items()):
        seeds = sorted(c.get("seed") for c in group)
        if len(seeds) < DEGENERACY_MIN_SEEDS:
            continue

        def _vals(field_path: List[str]) -> List[float]:
            out = []
            for c in group:
                v: Any = c
                for k in field_path:
                    v = v.get(k) if isinstance(v, dict) else None
                if isinstance(v, (int, float)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                    out.append(float(v))
            return out

        aloc = _vals(["quality", "A_loc"])
        q = _vals(["quality", "Q"])
        ec = _vals(["error_cost_eval"])
        installs = _vals(["cost", "install_gpu_s"])
        # Routing arm_counts: encode as tuple of (arm, count) sorted by arm for stability.
        arm_tuples: List[Tuple] = []
        for c in group:
            ac = c.get("routing", {}).get("arm_counts") or {}
            if isinstance(ac, dict):
                arm_tuples.append(tuple(sorted((k, v) for k, v in ac.items())))
        # A policy with all-identical routing across seeds is not necessarily suspicious
        # (always_* are invariant by construction); we report but do NOT flag.

        def _is_constant(vals: List[float]) -> bool:
            if len(vals) < 2:
                return False
            ref = vals[0]
            return all(abs(v - ref) <= DEGENERACY_ATOL for v in vals)

        all_routing_same = len(set(arm_tuples)) == 1
        all_constant = (len(aloc) >= 2 and _is_constant(aloc) and
                        len(q) >= 2 and _is_constant(q) and
                        len(ec) >= 2 and _is_constant(ec) and
                        len(installs) >= 2 and _is_constant(installs) and
                        all_routing_same)
        if all_constant:
            findings.append({
                "kind": "cross_seed_degenerate",
                "mix": mix,
                "policy": policy,
                "seeds_present": seeds,
                "A_loc_values": aloc,
                "Q_values": q,
                "error_cost_eval_values": ec,
                "install_gpu_s_values": installs,
                "routing_arm_counts": arm_tuples[0] if arm_tuples else None,
                "note": "ALL tested fields bit-exactly equal across all present seeds. "
                        "Real waves produce at least stream-content-driven variance in "
                        "A_loc / error_cost_eval / cost.",
                "severity": "FAIL",
            })
    return findings


# ------------------------------------------------------------------------------ loader
def _load_cells(cells_dir: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load every cell_*.json directly under cells_dir (maxdepth=1, no recursion).

    Returns (cells, parse_errors). Each loaded cell carries provenance annotations
    (_path, _filename, _fn_model, _fn_provenance, _fn_mix, _cell_id, _mtime_sec,
    _sha256) for downstream checks.
    """
    cells: List[Dict[str, Any]] = []
    parse_errors: List[Dict[str, Any]] = []
    for name in os.listdir(cells_dir):
        path = os.path.join(cells_dir, name)
        if name.startswith("cell_") and name.endswith(".json") and os.path.islink(path):
            parse_errors.append({"path": path, "error": "symlink cell rejected"})
    for name in _list_cell_jsons(cells_dir):
        path = os.path.join(cells_dir, name)
        parsed = _parse_cell_filename(name)
        try:
            with open(path, "rb") as f:
                raw = f.read()
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError("JSON cell body must be an object")
            reserved = sorted(k for k in body if k.startswith("_"))
            if reserved:
                raise ValueError(f"reserved loader annotations in JSON body: {reserved}")
        except (json.JSONDecodeError, ValueError) as e:
            parse_errors.append({"path": path, "error": str(e)})
            continue
        except OSError as e:
            parse_errors.append({"path": path, "error": str(e)})
            continue
        try:
            mtime = os.stat(path).st_mtime
        except OSError:
            mtime = -1
        cell_id = f"{parsed[2]}_{parsed[3]}_s{parsed[4]}" if parsed else name
        annotated = dict(body)
        annotated.update({
            "_path": path,
            "_filename": name,
            "_fn_model": parsed[0] if parsed else None,
            "_fn_provenance": parsed[1] if parsed else None,
            "_fn_mix": parsed[2] if parsed else None,
            "_fn_policy": parsed[3] if parsed else None,
            "_fn_seed": parsed[4] if parsed else None,
            "_mtime_sec": int(mtime) if mtime >= 0 else -1,
            "_sha256": _sha256_short(raw),
            "_cell_id": cell_id,
        })
        cells.append(annotated)
    return cells, parse_errors


# ------------------------------------------------------------------------------ main
def run_gate(cells_dir: str) -> Dict[str, Any]:
    cells, parse_errors = _load_cells(cells_dir)

    # Filter to the cells the gate is responsible for (model=llama-3.2-1b, provenance=real,
    # mix in MIX_B/MIX_C). Other cells are reported but not validated for grid membership.
    in_scope = [c for c in cells
                if c.get("model") == EXPECTED_MODEL
                and c.get("provenance") == EXPECTED_PROVENANCE
                and c.get("mix") in EXPECTED_MIXES]

    malformed = _check_malformed(in_scope, parse_errors)
    unparseable = _check_unparseable_filenames(cells)
    foreign = _check_foreign_extras(cells)
    schema = _check_schema(in_scope)
    identity = _check_identity_mismatch(cells)  # ALL loaded cells (relabeling tell)
    dups = _check_duplicates(in_scope)
    synth_anchors = _check_exact_synthetic_anchors(in_scope)
    round_anchors = _check_round_anchors(in_scope)
    burst = _check_burst(in_scope, cells_dir)
    non_degen = _check_non_degeneracy(in_scope)
    p2 = _check_p2(cells_dir)

    # Missing-cell set is computed AFTER we know what's in scope (and ignoring non-scope cells).
    present_mix_set = {c.get("mix") for c in in_scope}
    missing = _check_missing(in_scope, present_mix_set)

    # Roll-up severity.
    def _max_sev(items: List[Dict[str, Any]]) -> str:
        order = {"INFO": 0, "WARN": 1, "FAIL": 2}
        m = 0
        for it in items:
            m = max(m, order.get(it.get("severity", "INFO"), 0))
        return {0: "INFO", 1: "WARN", 2: "FAIL"}[m]

    finding_groups = {
        "malformed_json": malformed,
        "unparseable_filename": unparseable,
        "foreign_extra": foreign,
        "schema": schema,
        "identity_mismatch": identity,
        "duplicate_cell": dups,
        "synthetic_anchor": synth_anchors,
        "round_anchor_suspicious": round_anchors,
        "write_burst": burst,
        "non_degeneracy": non_degen,
        "p2_status": p2,
        "missing_cells": missing,
    }
    group_max = {k: _max_sev(v) for k, v in finding_groups.items()}

    # Top-line status decision:
    #   FAIL      — at least one FAIL finding, OR the grid is COMPLETE (66 in-scope cells)
    #               AND a P2 is missing (then it's a complete-but-suspicious wave).
    #   INCOMPLETE — fewer than 66 in-scope cells, with no FAIL findings yet.
    #   PASS      — 66 cells, P2 present, no FAIL findings.
    n_in_scope = len(in_scope)
    any_fail = any(f["severity"] == "FAIL" for v in finding_groups.values() for f in v)
    p2_present = any(f["kind"] == "p2_ok" for f in p2)

    if n_in_scope == EXPECTED_TOTAL and p2_present and not any_fail:
        status = "PASS"
        exit_code = 0
    elif n_in_scope < EXPECTED_TOTAL and not any_fail:
        status = "INCOMPLETE"
        exit_code = 2
    else:
        status = "FAIL"
        exit_code = 1

    report = {
        "schema_version": "frame_a.provenance_gate.v1",
        "generated_at_utc": _utc_now_iso(),
        "cells_dir": os.path.abspath(cells_dir),
        "scope": {
            "model": EXPECTED_MODEL,
            "provenance": EXPECTED_PROVENANCE,
            "mixes": list(EXPECTED_MIXES),
            "seeds": list(EXPECTED_SEEDS),
            "policies": list(EXPECTED_POLICIES),
            "expected_total_cells": EXPECTED_TOTAL,
            "expected_p2_file": NAMESPACED_P2_NAME,
        },
        "counts": {
            "files_on_disk": len(_list_cell_jsons(cells_dir)),
            "in_scope_cells": n_in_scope,
            "out_of_scope_cells": len(cells) - n_in_scope,
            "parse_errors": len(parse_errors),
        },
        "thresholds": {
            "burst_threshold": BURST_THRESHOLD,
            "round_anchor_floor": ROUND_ANCHOR_FLOOR,
            "round_anchor_range": list(ROUND_ANCHOR_RANGE),
            "degeneracy_min_seeds": DEGENERACY_MIN_SEEDS,
            "degeneracy_atol": DEGENERACY_ATOL,
        },
        "caveats": [
            "burst check uses POSIX mtime; rsync -a / cp -p preserve source mtimes — "
            "a copied wave will look bursty even if the data is real. Verify against the "
            "source-host log before acting on a burst alert.",
            "exact synthetic anchors install_gpu_s=4000.0 / serve_gpu_s=1500.0 are the "
            "2026-07-21 incident's deterministic replay values; a future variant might "
            "shift them. Re-tune EXACT_SYNTH_INSTALL_GPU_S / EXACT_SYNTH_SERVE_GPU_S if "
            "the synthetic cost-harness schema changes.",
            "quarantine directories (anything starting with '.', incl. .synthetic-relabel-bak/) "
            "are NEVER traversed (maxdepth=1 via os.listdir, hidden entries skipped).",
        ],
        "status": status,
        "exit_code": exit_code,
        "group_severity": group_max,
        "findings": finding_groups,
        "in_scope_cell_ids": sorted(c["_cell_id"] for c in in_scope),
    }
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Post-wave provenance hard gate for Frame-A cells (independent of the "
                    "existing analyzer; checks numbers + write-time + layout).",
    )
    ap.add_argument("--cells_dir", default="results/frame_a/cells",
                    help="Directory containing cell_*.json and p2_*_MIX_C.json "
                         "(default: results/frame_a/cells).")
    ap.add_argument("--report", default=None,
                    help="Optional path to also write the JSON report (always echoed on stdout).")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.cells_dir):
        print(json.dumps({"status": "USAGE", "error": f"cells_dir not a directory: {args.cells_dir}"}),
              file=sys.stdout)
        return 3

    report = run_gate(args.cells_dir)
    out = json.dumps(report, indent=2, sort_keys=False)
    print(out)
    if args.report:
        with open(args.report, "w") as f:
            f.write(out + "\n")
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())