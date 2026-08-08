"""selftest_provenance_gate.py — executable selftest for provenance_gate.py.

Runs in <2s on a CPU. Builds two fixtures in a tmpdir:
  1) synthetic_incident/  — replicates the 2026-07-21 Frame-A burst (67 cells in 2
     seconds, exact install=4000.0 / serve=1500.0 anchors, zero seed variance on A_loc,
     missing P2). Expects FAIL.
  2) legitimate_varying/ — real-looking cells: spread mtimes, exact anchors absent,
     install noisy (e.g. 238.55 / 198.07), per-seed A_loc variance, P2 present.
     Expects PASS once all 66 cells + P2 are written; partial variant expects
     INCOMPLETE.

Each fixture builds the full grid (11 policies × 3 seeds × 2 mixes = 66) so the grid-
membership check is not the trigger — only the NUMERIC / TEMPORAL checks are.

Exit code: 0 all-pass, 1 any-fail.

Imports only stdlib + the gate under test. Does NOT import any live-frame_a module.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import provenance_gate as pg  # noqa: E402


# ------------------------------------------------------------------------------ fixtures
def _base_cell(model: str, mix: str, policy: str, seed: int, install: float, serve: float,
               a_loc: float, q: float, error_cost: float) -> Dict:
    """A minimal but schema-valid cell, structured like a real Frame-A cell."""
    return {
        "mix": mix,
        "policy": policy,
        "seed": seed,
        "model": model,
        "provenance": "real",
        "quality": {"A_upd": 1.0, "A_loc": a_loc, "A_cum": 1.0, "A_rip": 1.0, "Q": q},
        "cost": {"install_gpu_s": install, "serve_gpu_s": serve,
                 "total_gpu_s": install + serve,
                 "serve_overhead_total": 0.0, "store_bytes_peak": 0.0,
                 "exposure_surface_mean": 0.0},
        "error_cost_eval": error_cost,
        "discovery": {"n_damaging_gt": 60, "recall_at_decide": 0.0,
                      "chance": 0.0993, "lift": 0.0,
                      "predictor_ceiling": 0.4407, "recall_over_ceiling": 0.0},
        "routing": {"edit_on_privacy": 0, "privacy_total": 0,
                    "arm_counts": {"edit": 500}},
        "stream_hash": f"hash_{policy}_{seed}",
    }


def build_synthetic_incident(root: str, subdir: str = "synthetic_incident") -> str:
    """Replicates the 2026-07-21 burst: 67 cells in 2 seconds, exact synthetic anchors,
    zero per-policy seed variance on A_loc, NO namespaced P2. -> gate must FAIL."""
    d = os.path.join(root, subdir)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)
    fixed_time = int(time.time()) - 86400  # yesterday, like the 2026-07-21 incident
    paths: List[str] = []
    for mix in pg.EXPECTED_MIXES:
        for policy in pg.EXPECTED_POLICIES:
            for seed in pg.EXPECTED_SEEDS:
                # ZERO seed variance on A_loc, Q, install, error_cost — synthetic replay tell.
                cell = _base_cell(pg.EXPECTED_MODEL, mix, policy, seed,
                                  install=pg.EXACT_SYNTH_INSTALL_GPU_S,
                                  serve=pg.EXACT_SYNTH_SERVE_GPU_S,
                                  a_loc=0.34647042753955315, q=0.8039411282618659,
                                  error_cost=33793.73824635465)
                path = os.path.join(d, pg._build_filename("real", mix, policy, seed))
                # Two write-burst seconds (34 in sec, 33 in sec+1), mirroring the incident.
                burst_sec = fixed_time + (0 if len(paths) < 34 else 1)
                os.utime(path, (burst_sec, burst_sec)) if os.path.exists(path) else None
                with open(path, "w") as f:
                    json.dump(cell, f)
                os.utime(path, (burst_sec, burst_sec))
                paths.append(path)
    # Namespaced P2 deliberately absent.
    return d


def build_legitimate_varying(root: str, *, omit_p2: bool = False,
                             drop_fraction: float = 0.0,
                             subdir: str = "legitimate_varying") -> str:
    """Real-looking cells: spread mtimes, exact anchors absent, per-seed variance."""
    d = os.path.join(root, subdir)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)
    base_time = int(time.time()) - 7200  # 2h ago; cells written ~5 min apart
    cell_idx = 0
    for mix in pg.EXPECTED_MIXES:
        for policy in pg.EXPECTED_POLICIES:
            for seed in pg.EXPECTED_SEEDS:
                cell_idx += 1
                # Per-policy noise so A_loc / install / error_cost vary across seeds.
                # install varies ~ +-25 around a policy-mean; serve ~ constant-ish.
                policy_offset = (hash(policy) % 1000) / 1000.0
                install = 200.0 + policy_offset * 100.0 + seed * 13.7
                serve = 6.0 + seed * 0.21
                a_loc = 0.72 + policy_offset * 0.05 + seed * 0.011
                q = 0.83 + policy_offset * 0.02 + seed * 0.003
                error_cost = 5000.0 + seed * 100.0 + policy_offset * 1000.0
                cell = _base_cell(pg.EXPECTED_MODEL, mix, policy, seed,
                                  install=install, serve=serve,
                                  a_loc=a_loc, q=q, error_cost=error_cost)
                path = os.path.join(d, pg._build_filename("real", mix, policy, seed))
                # Vary mtimes: one per ~5 minutes, no two in the same second.
                t = base_time + cell_idx * 300
                if drop_fraction > 0 and (cell_idx / 66.0) <= drop_fraction:
                    continue  # simulate a still-running wave
                with open(path, "w") as f:
                    json.dump(cell, f)
                os.utime(path, (t, t))
    if not omit_p2:
        p2 = {
            "exposure_edit": 0.0, "exposure_rag": 1.0,
            "footprint_delta": 128000.0, "overhead_delta": 0.6,
            "router_edit_majority_on_privacy": 0.80,
        }
        p2_path = os.path.join(d, pg.EXPECTED_MODEL and pg.NAMESPACED_P2_NAME)
        with open(p2_path, "w") as f:
            json.dump(p2, f)
    return d


# Helper — synthesize the canonical cell filename.
def _build_filename_helper(provenance: str, mix: str, policy: str, seed: int) -> str:
    return f"cell_{pg.EXPECTED_MODEL}_{provenance}_{mix}_{policy}_s{seed}.json"


# Inject the helper into the gate module under test (so the fixtures match real filenames).
pg._build_filename = _build_filename_helper  # type: ignore[attr-defined]


# ------------------------------------------------------------------------------ tests
def _check(label: str, got: str, want: str) -> bool:
    ok = got == want
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}: got={got!r} want={want!r}")
    return ok


def _check_contains(label: str, report: Dict, group: str, kind: str) -> bool:
    items = report["findings"].get(group, [])
    hit = any(it.get("kind") == kind for it in items)
    mark = "PASS" if hit else "FAIL"
    print(f"  [{mark}] {label}: finding group={group!r} kind={kind!r} present={hit}")
    return hit


def _check_not_contains(label: str, report: Dict, group: str, kind: str) -> bool:
    items = report["findings"].get(group, [])
    hit = any(it.get("kind") == kind for it in items)
    mark = "PASS" if not hit else "FAIL"
    print(f"  [{mark}] {label}: finding group={group!r} kind={kind!r} absent={not hit}")
    return not hit


def test_synthetic_incident_fails(root: str) -> bool:
    print("\n[test_synthetic_incident_fails] expecting FAIL with multiple tells.")
    d = build_synthetic_incident(root)
    rep = pg.run_gate(d)
    ok = True
    ok &= _check("status", rep["status"], "FAIL")
    ok &= _check_contains("synthetic_anchor_exact", rep, "synthetic_anchor", "synthetic_anchor_exact")
    ok &= _check_contains("write_burst", rep, "write_burst", "write_burst")
    ok &= _check_contains("non_degeneracy", rep, "non_degeneracy", "cross_seed_degenerate")
    ok &= _check_contains("missing_p2", rep, "p2_status", "missing_p2")
    return ok


def test_legitimate_varying_passes(root: str) -> bool:
    print("\n[test_legitimate_varying_passes] expecting PASS with full grid + P2.")
    d = build_legitimate_varying(root)
    rep = pg.run_gate(d)
    ok = True
    ok &= _check("status", rep["status"], "PASS")
    ok &= _check("in_scope_cells", rep["counts"]["in_scope_cells"], pg.EXPECTED_TOTAL)
    ok &= _check_not_contains("no synthetic anchor", rep, "synthetic_anchor", "synthetic_anchor_exact")
    ok &= _check_not_contains("no burst", rep, "write_burst", "write_burst")
    ok &= _check_not_contains("no degeneracy", rep, "non_degeneracy", "cross_seed_degenerate")
    return ok


def test_partial_wave_incomplete(root: str) -> bool:
    print("\n[test_partial_wave_incomplete] expecting INCOMPLETE (50% of grid, no fails).")
    d = build_legitimate_varying(root, omit_p2=True, drop_fraction=0.5)
    rep = pg.run_gate(d)
    ok = True
    ok &= _check("status", rep["status"], "INCOMPLETE")
    ok &= _check("in_scope_cells<66", rep["counts"]["in_scope_cells"] < pg.EXPECTED_TOTAL, True)
    ok &= _check("exit_code=2", rep["exit_code"], 2)
    return ok


def test_quarantine_never_traversed(root: str) -> bool:
    print("\n[test_quarantine_never_traversed] cells inside .synthetic-relabel-bak/ ignored.")
    d = os.path.join(root, "quarantine_check")
    os.makedirs(d, exist_ok=True)
    # Put the legitimate varying fixture in cells/, plus a poisoned quarantine.
    cells_dir = build_legitimate_varying(d)
    quarantine = os.path.join(cells_dir, ".synthetic-relabel-bak")
    os.makedirs(quarantine, exist_ok=True)
    poisoned = _base_cell(pg.EXPECTED_MODEL, "MIX_B", "always_edit", 0,
                          install=pg.EXACT_SYNTH_INSTALL_GPU_S,
                          serve=pg.EXACT_SYNTH_SERVE_GPU_S,
                          a_loc=0.346, q=0.8, error_cost=33793.7)
    with open(os.path.join(quarantine, "cell_llama-3.2-1b_real_MIX_B_always_edit_s99.json"), "w") as f:
        json.dump(poisoned, f)
    rep = pg.run_gate(cells_dir)
    ok = True
    ok &= _check("status", rep["status"], "PASS")
    # The poisoned s99 file must NOT appear in in_scope_cell_ids.
    ok &= _check("quarantine file excluded", "MIX_B_always_edit_s99" in rep["in_scope_cell_ids"], False)
    return ok


def test_identity_mismatch_detected(root: str) -> bool:
    print("\n[test_identity_mismatch_detected] filename says real, body says synth.")
    d = os.path.join(root, "identity_mismatch")
    os.makedirs(d, exist_ok=True)
    base_time = int(time.time()) - 3600
    cell_idx = 0
    for mix in pg.EXPECTED_MIXES:
        for policy in pg.EXPECTED_POLICIES:
            for seed in pg.EXPECTED_SEEDS:
                cell_idx += 1
                cell = _base_cell(pg.EXPECTED_MODEL, mix, policy, seed,
                                  install=200.0 + seed * 13.7,
                                  serve=6.0 + seed * 0.21,
                                  a_loc=0.72 + seed * 0.011,
                                  q=0.83 + seed * 0.003,
                                  error_cost=5000.0 + seed * 100.0)
                cell["provenance"] = "synth"  # body disagrees with filename
                path = os.path.join(d, _build_filename_helper("real", mix, policy, seed))
                with open(path, "w") as f:
                    json.dump(cell, f)
                os.utime(path, (base_time + cell_idx * 300, base_time + cell_idx * 300))
    rep = pg.run_gate(d)
    ok = True
    ok &= _check("status", rep["status"], "FAIL")
    ok &= _check_contains("identity_mismatch finding", rep, "identity_mismatch", "identity_mismatch")
    return ok


def test_malformed_json_detected(root: str) -> bool:
    print("\n[test_malformed_json_detected] one truncated JSON must trip malformed.")
    d = os.path.join(root, "malformed")
    os.makedirs(d, exist_ok=True)
    # Reuse the legitimate fixture then overwrite one cell with truncated JSON.
    cells_dir = build_legitimate_varying(d)
    bad_path = os.path.join(cells_dir, _build_filename_helper("real", "MIX_B", "always_edit", 0))
    with open(bad_path, "w") as f:
        f.write("{this is not valid json")
    rep = pg.run_gate(cells_dir)
    ok = True
    ok &= _check("status", rep["status"], "FAIL")
    ok &= _check_contains("malformed_json", rep, "malformed_json", "malformed_json")
    return ok


def test_p2_body_invalid_detected(root: str) -> bool:
    print("\n[test_p2_body_invalid_detected] malformed structural P2 must FAIL.")
    d = build_legitimate_varying(root, subdir="bad_p2")
    p = os.path.join(d, pg.NAMESPACED_P2_NAME)
    with open(p, "w") as f:
        json.dump({"model": "wrong-model", "exposure_edit": 0.0}, f)
    rep = pg.run_gate(d)
    ok = True
    ok &= _check("status", rep["status"], "FAIL")
    ok &= _check_contains("p2 body invalid", rep, "p2_status", "p2_body_invalid")
    return ok


def test_unparseable_filename_detected(root: str) -> bool:
    print("\n[test_unparseable_filename_detected] malformed cell filename must FAIL.")
    d = build_legitimate_varying(root, subdir="bad_filename")
    src = os.path.join(d, _build_filename_helper("real", "MIX_B", "always_edit", 0))
    with open(src) as f:
        body = json.load(f)
    with open(os.path.join(d, "cell_unparseable.json"), "w") as f:
        json.dump(body, f)
    rep = pg.run_gate(d)
    ok = True
    ok &= _check("status", rep["status"], "FAIL")
    ok &= _check_contains("unparseable filename", rep, "unparseable_filename",
                          "unparseable_filename")
    return ok


def test_policy_seed_mismatch_detected(root: str) -> bool:
    print("\n[test_policy_seed_mismatch_detected] renamed policy/seed must FAIL.")
    d = build_legitimate_varying(root, subdir="policy_seed_mismatch")
    p = os.path.join(d, _build_filename_helper("real", "MIX_B", "always_edit", 0))
    with open(p) as f:
        body = json.load(f)
    body["policy"] = "always_grace"
    body["seed"] = 2
    with open(p, "w") as f:
        json.dump(body, f)
    rep = pg.run_gate(d)
    ok = True
    ok &= _check("status", rep["status"], "FAIL")
    ok &= _check_contains("policy/seed identity mismatch", rep, "identity_mismatch",
                          "identity_mismatch")
    return ok


def test_cli_exit_codes(root: str) -> bool:
    print("\n[test_cli_exit_codes] main() returns expected exit codes.")
    import io
    from contextlib import redirect_stdout
    ok = True
    for label, build, want in [
        ("PASS", lambda: build_legitimate_varying(root, subdir="cli_pass"), 0),
        ("FAIL", lambda: build_synthetic_incident(root, subdir="cli_fail"), 1),
        ("INCOMPLETE", lambda: build_legitimate_varying(root, omit_p2=True, drop_fraction=0.5,
                                                       subdir="cli_incomplete"), 2),
    ]:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = pg.main(["--cells_dir", build()])
        rep = json.loads(buf.getvalue())
        ok &= _check(f"cli {label} rc", rc, want)
        ok &= _check(f"cli {label} status", rep["status"],
                     {0: "PASS", 1: "FAIL", 2: "INCOMPLETE"}[want])
    return ok


def test_known_mix_a_coexists(root: str) -> bool:
    print("\n[test_known_mix_a_coexists] known MIX_A artifacts remain out of scope.")
    d = build_legitimate_varying(root, subdir="known_mix_a")
    cell = _base_cell(pg.EXPECTED_MODEL, "MIX_A", "always_edit", 0,
                      install=238.55, serve=7.92, a_loc=0.71, q=0.82, error_cost=5100.0)
    path = os.path.join(d, _build_filename_helper("real", "MIX_A", "always_edit", 0))
    with open(path, "w") as f:
        json.dump(cell, f)
    rep = pg.run_gate(d)
    ok = _check("status", rep["status"], "PASS")
    ok &= _check_not_contains("MIX_A not foreign", rep, "foreign_extra", "foreign_extra")
    ok &= _check("no noisy round warnings", len(rep["findings"]["round_anchor_suspicious"]), 0)
    return ok


def test_adversarial_cases(root: str) -> bool:
    print("\n[test_adversarial_cases] hostile bypass regressions.")
    ok = True
    # Anchor jitter: repeated identical near-round placeholders fail; varied noise passes.
    d = build_legitimate_varying(root, subdir="anchor_jitter")
    for policy in ("always_edit", "always_ft", "always_grace"):
        p = os.path.join(d, _build_filename_helper("real", "MIX_B", policy, 0))
        body = json.load(open(p)); body["cost"]["install_gpu_s"] = 1000.2; json.dump(body, open(p, "w"))
    rep = pg.run_gate(d); ok &= _check("anchor jitter", len(rep["findings"]["round_anchor_suspicious"]) > 0, True)
    # JSON scalar and truncated JSON are corruption FAILs.
    for label, payload in (("scalar", "null"), ("truncated", "{")):
        d = build_legitimate_varying(root, subdir=label)
        p = os.path.join(d, _build_filename_helper("real", "MIX_B", "always_edit", 0))
        open(p, "w").write(payload); rep = pg.run_gate(d)
        ok &= _check(label, rep["status"], "FAIL")
    # Negative/string/NaN core metrics fail.
    for value in (-1, "bad", float("nan")):
        d = build_legitimate_varying(root, subdir="bad_install")
        p = os.path.join(d, _build_filename_helper("real", "MIX_B", "always_edit", 0))
        body = json.load(open(p)); body["cost"]["install_gpu_s"] = value; json.dump(body, open(p, "w"), allow_nan=True)
        rep = pg.run_gate(d); ok &= _check("bad install", rep["status"], "FAIL")
    # Foreign target namespace, metadata overwrite, and bool/float seed aliases fail.
    d = build_legitimate_varying(root, subdir="foreign")
    src = os.path.join(d, _build_filename_helper("real", "MIX_B", "always_edit", 0))
    os.rename(src, os.path.join(d, "cell_llama-3.2-1b_real_MIX_B_unknown_s0.json")); rep = pg.run_gate(d)
    ok &= _check("foreign target", rep["status"], "FAIL")
    d = build_legitimate_varying(root, subdir="metadata")
    p = os.path.join(d, _build_filename_helper("real", "MIX_B", "always_edit", 0)); body = json.load(open(p)); body["_fn_seed"] = 99; json.dump(body, open(p, "w")); rep = pg.run_gate(d)
    ok &= _check("metadata overwrite rejected", rep["status"], "FAIL")
    # Slow-drip burst: eight writes spread over one minute still fail.
    d = build_legitimate_varying(root, subdir="slow_drip")
    base = int(time.time()) - 1000
    for i, name in enumerate(pg._list_cell_jsons(d)[:8]):
        os.utime(os.path.join(d, name), (base + i * 8, base + i * 8))
    ok &= _check("slow drip", pg.run_gate(d)["status"], "FAIL")
    # P2 in a subdirectory does not satisfy the namespaced root requirement.
    d = build_legitimate_varying(root, omit_p2=True, subdir="p2_subdir")
    os.makedirs(os.path.join(d, "nested")); json.dump({"exposure_edit": 0, "exposure_rag": 1,
        "footprint_delta": 1, "overhead_delta": 1, "router_edit_majority_on_privacy": .8},
        open(os.path.join(d, "nested", pg.NAMESPACED_P2_NAME), "w"))
    ok &= _check("P2 subdir missing", pg.run_gate(d)["status"], "FAIL")
    # Missing and semantically invalid P2 metrics fail.
    for label, mutate in (
        ("missing", lambda p: p.pop("overhead_delta")),
        ("negative", lambda p: p.update(footprint_delta=-1)),
        ("ordering", lambda p: p.update(exposure_edit=2, exposure_rag=1)),
        ("router", lambda p: p.update(router_edit_majority_on_privacy=1.5)),
        ("string", lambda p: p.update(overhead_delta="bad")),
        ("nan", lambda p: p.update(overhead_delta=float("nan"))),
    ):
        d = build_legitimate_varying(root, subdir="p2_bad_" + label)
        path = os.path.join(d, pg.NAMESPACED_P2_NAME); p2 = json.load(open(path)); mutate(p2)
        json.dump(p2, open(path, "w"), allow_nan=True)
        ok &= _check("P2 " + label, pg.run_gate(d)["status"], "FAIL")
    # Bool/float seed aliases are unparseable target lookalikes.
    for alias in ("True", "1.0"):
        d = build_legitimate_varying(root, subdir="seed_alias")
        src = os.path.join(d, _build_filename_helper("real", "MIX_B", "always_edit", 0))
        os.rename(src, os.path.join(d, f"cell_{pg.EXPECTED_MODEL}_real_MIX_B_always_edit_s{alias}.json"))
        ok &= _check("seed alias", pg.run_gate(d)["status"], "FAIL")
    # Symlinked target cell and P2 cannot bypass direct regular-file loading.
    d = build_legitimate_varying(root, subdir="symlink")
    target = os.path.join(d, _build_filename_helper("real", "MIX_B", "always_edit", 0))
    backup = target + ".target"; os.rename(target, backup); os.symlink(backup, target)
    ok &= _check("cell symlink", pg.run_gate(d)["status"], "FAIL")
    d = build_legitimate_varying(root, subdir="p2_symlink")
    p2path = os.path.join(d, pg.NAMESPACED_P2_NAME); p2backup = p2path + ".target"
    os.rename(p2path, p2backup); os.symlink(p2backup, p2path)
    ok &= _check("P2 symlink", pg.run_gate(d)["status"], "FAIL")
    return ok


# ------------------------------------------------------------------------------ entry
def main() -> int:
    print("selftest_provenance_gate.py — Frame-A provenance hard gate")
    root = tempfile.mkdtemp(prefix="frame_a_gate_")
    try:
        results = [
            test_synthetic_incident_fails(root),
            test_legitimate_varying_passes(root),
            test_partial_wave_incomplete(root),
            test_quarantine_never_traversed(root),
            test_identity_mismatch_detected(root),
            test_malformed_json_detected(root),
            test_p2_body_invalid_detected(root),
            test_unparseable_filename_detected(root),
            test_policy_seed_mismatch_detected(root),
            test_known_mix_a_coexists(root),
            test_adversarial_cases(root),
            test_cli_exit_codes(root),
        ]
    finally:
        shutil.rmtree(root, ignore_errors=True)
    n_pass = sum(1 for r in results if r)
    n_fail = sum(1 for r in results if not r)
    print(f"\n=== {n_pass} passed, {n_fail} failed (of {len(results)}) ===")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())