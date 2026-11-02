"""
Run the three-pillar whole-CNS structural specificity, parallel virtual Split-GAL4 DN screen,
and within-connectome fru/dsx sex-switch experiments on MaleCNS v1.0 and Female BANC v888.

Usage:
    python experiments/run_structural_specificity_and_screen.py --part all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import mannwhitneyu, spearmanr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.connectome_engine import ConnectomeDataset, _HAS_TRITON, resolve_female_connectome_dir  # noqa: E402
from src.structural_specificity import (  # noqa: E402
    CSRSnapshot,
    compute_synapse_threshold_audit,
    evaluate_sex_switch_double_dissociation_female,
    evaluate_sex_switch_double_dissociation_male,
    evaluate_three_circuits_open_loop,
    extract_dn_screen_catalog,
    extract_sex_switch_mask,
    extract_three_circuit_anatomy,
    rewire_all_isomorphic,
    rewire_degree_preserving,
    rewire_fru_dsx_subgraph,
    rewire_isomorphic_matched,
    rewire_kc_mbon_compartments,
    rewire_partial_matched,
    rewire_pn_kc_random_claws,
    rewire_within_instance,
    rewire_within_region,
    rewire_within_type,
    rewire_within_type_side,
    run_parallel_dn_perturbation_screen,
)

DEFAULT_OUT = REPO / "results" / "structural_specificity_screen_results.json"
REWIRE_SEEDS = (11, 22, 33, 44, 55)
REAL_EVAL_SEEDS = (500, 501, 502, 503, 504)
DN_SCREEN_SEEDS = (700, 701, 702, 703, 704)


def _assert_gpu_csr_synced(ds: ConnectomeDataset) -> None:
    """Verify CUDA CSR arrays match CPU W_base_csr byte-for-byte after any in-place rewiring."""
    if not getattr(ds, "_cuda_ready", False):
        return
    W = ds.W_base_csr
    assert np.array_equal(ds.csr_indptr_gpu.cpu().numpy(), W.indptr.astype(np.int32)), "Stale GPU indptr"
    assert np.array_equal(ds.csr_indices_gpu.cpu().numpy(), W.indices.astype(np.int32)), "Stale GPU indices"
    assert np.array_equal(ds.csr_data_gpu.cpu().numpy(), W.data.astype(np.float32)), "Stale GPU data"


def _mean_sd(vals: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(vals, dtype=np.float64)
    return {
        "mean": round(float(a.mean()), 4),
        "sd": round(float(a.std(ddof=1)) if len(a) > 1 else 0.0, 4),
        "n": int(len(a)),
    }


def _mwu_p(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 1.0
    if np.allclose(a, b):
        return 1.0
    _, p = mannwhitneyu(a, b, alternative="two-sided")
    return round(float(p), 6)


# ---------------------------------------------------------------------------
# Pillar 1: Three-Tier Whole-CNS Wiring Specificity Hierarchy
# ---------------------------------------------------------------------------
def run_pillar_1_hierarchy(ds_m: ConnectomeDataset, rewire_seeds: Sequence[int]) -> Dict[str, object]:
    print("\n=== [Pillar 1] Three-Circuit Whole-CNS Wiring Specificity Hierarchy ===", flush=True)
    snap = CSRSnapshot.capture(ds_m)
    anat = extract_three_circuit_anatomy(ds_m)

    conditions: List[Tuple[str, Optional[Callable[[ConnectomeDataset, int], Dict[str, object]]]]] = [
        ("real", None),
        ("rewired_pn_kc_random_claws", rewire_pn_kc_random_claws),
        ("rewired_kc_mbon_compartments", rewire_kc_mbon_compartments),
        ("rewired_instance", rewire_within_instance),
        ("rewired_type_side", rewire_within_type_side),
        ("rewired_partial_48", lambda d, s: rewire_partial_matched(d, s, 0.562)),
        ("rewired_type", rewire_within_type),
        ("rewired_region", rewire_within_region),
        ("rewired_full", rewire_degree_preserving),
    ]

    # Calibrate partial_48 so its realized fraction_edges_rewired matches rewired_type_side (~48.0%)
    snap.restore(ds_m)
    ts_rep = rewire_within_type_side(ds_m, rewire_seeds[0])
    target_frac = float(ts_rep["fraction_edges_rewired"])
    snap.restore(ds_m)
    conditions = [
        (
            (name, (lambda d, s, tf=target_frac: rewire_partial_matched(d, s, tf)))
            if name == "rewired_partial_48"
            else (name, fn)
        )
        for name, fn in conditions
    ]

    validation_rows: List[Dict[str, object]] = []
    trial_rows: List[Dict[str, object]] = []

    try:
        for cond_name, rewire_fn in conditions:
            t_c = time.perf_counter()
            if rewire_fn is None:
                snap.restore(ds_m)
                _assert_gpu_csr_synced(ds_m)
                for es in REAL_EVAL_SEEDS:
                    metrics = evaluate_three_circuits_open_loop(
                        ds_m, anat, n_trials=5, stim_ms=60, bg_noise_hz=20.0, seed=es
                    )
                    trial_rows.append({
                        "condition": cond_name,
                        "rewire_seed": None,
                        "eval_seed": int(es),
                        "fraction_edges_rewired": 0.0,
                        **metrics,
                    })
            else:
                for rs in rewire_seeds:
                    snap.restore(ds_m)
                    rep = rewire_fn(ds_m, int(rs))
                    _assert_gpu_csr_synced(ds_m)
                    rep["condition"] = cond_name
                    validation_rows.append(rep)
                    metrics = evaluate_three_circuits_open_loop(
                        ds_m, anat, n_trials=5, stim_ms=60, bg_noise_hz=20.0, seed=500 + int(rs)
                    )
                    trial_rows.append({
                        "condition": cond_name,
                        "rewire_seed": int(rs),
                        "eval_seed": 500 + int(rs),
                        "fraction_edges_rewired": rep["fraction_edges_rewired"],
                        **metrics,
                    })

            sub = [r for r in trial_rows if r["condition"] == cond_name]
            fr = float(np.mean([r["fraction_edges_rewired"] for r in sub]))
            print(
                f"  {cond_name:28s} (rewired={100*fr:5.1f}%, {time.perf_counter()-t_c:5.1f}s) | "
                f"Steer={np.mean([r['steer_goal_sep_hz'] for r in sub]):6.2f} Hz | "
                f"CX acc={100*np.mean([r['cx_decode_acc'] for r in sub]):5.1f}% "
                f"(Δcos={np.mean([r['cx_coding_contrast'] for r in sub]):+.3f}, "
                f"hz={np.mean([r['cx_mean_hz'] for r in sub]):4.2f}) | "
                f"MB KC acc={100*np.mean([r['mb_kc_decode_acc'] for r in sub]):5.1f}% "
                f"(MBON acc={100*np.mean([r['mb_mbon_decode_acc'] for r in sub]):5.1f}%)",
                flush=True,
            )
    finally:
        snap.restore(ds_m)

    metric_keys = (
        "fraction_edges_rewired",
        "steer_goal_sep_hz",
        "steer_leg_mn_sep_hz",
        "cx_decode_acc",
        "cx_coding_contrast",
        "cx_antipodal_mod",
        "cx_mean_hz",
        "cx_epg_on_hz",
        "cx_epg_off_hz",
        "cx_epg_persistence_ratio",
        "mb_kc_decode_acc",
        "mb_kc_coding_contrast",
        "mb_kc_sparsity",
        "mb_mbon_decode_acc",
        "mb_mbon_coding_contrast",
    )
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cond_name, _ in conditions:
        sub = [r for r in trial_rows if r["condition"] == cond_name]
        summary[cond_name] = {mk: _mean_sd([r[mk] for r in sub]) for mk in metric_keys}

    def _vals(c: str, mk: str) -> List[float]:
        return [float(r[mk]) for r in trial_rows if r["condition"] == c]

    statistical_tests = {
        "steer_type_side_vs_partial48_mwu_p": _mwu_p(_vals("rewired_type_side", "steer_goal_sep_hz"), _vals("rewired_partial_48", "steer_goal_sep_hz")),
        "steer_type_side_vs_type_mwu_p": _mwu_p(_vals("rewired_type_side", "steer_goal_sep_hz"), _vals("rewired_type", "steer_goal_sep_hz")),
        "cx_acc_instance_vs_type_side_mwu_p": _mwu_p(_vals("rewired_instance", "cx_decode_acc"), _vals("rewired_type_side", "cx_decode_acc")),
        "cx_hz_real_vs_instance_mwu_p": _mwu_p(_vals("real", "cx_mean_hz"), _vals("rewired_instance", "cx_mean_hz")),
        "cx_hz_instance_vs_type_side_mwu_p": _mwu_p(_vals("rewired_instance", "cx_mean_hz"), _vals("rewired_type_side", "cx_mean_hz")),
        "mb_kc_acc_pn_kc_random_vs_region_mwu_p": _mwu_p(_vals("rewired_pn_kc_random_claws", "mb_kc_decode_acc"), _vals("rewired_region", "mb_kc_decode_acc")),
        "mb_mbon_acc_real_vs_kc_mbon_rewired_mwu_p": _mwu_p(_vals("real", "mb_mbon_decode_acc"), _vals("rewired_kc_mbon_compartments", "mb_mbon_decode_acc")),
    }

    return {
        "n_epg_wedges": len(anat.epg_wedges_8),
        "n_cx_instance_pools": len(anat.cx_readout_pools),
        "odor_glomeruli": anat.odor_glom_names,
        "n_kenyon_cells": int(len(anat.kc_indices)),
        "n_mbon_types": len(anat.mbon_readout_pools),
        "statistical_tests": statistical_tests,
        "validation": validation_rows,
        "trials": trial_rows,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Pillar 2: Parallel Virtual Split-GAL4 Screen of All 486 DN Cell Types
# ---------------------------------------------------------------------------
def run_pillar_2_dn_screen(ds_m: ConnectomeDataset) -> Dict[str, object]:
    print("\n=== [Pillar 2] Parallel Virtual Split-GAL4 Screen (All 486 DN Cell Types) ===", flush=True)
    snap = CSRSnapshot.capture(ds_m)
    cat = extract_dn_screen_catalog(ds_m)
    print(f"  Cataloged {len(cat.dn_types)} Descending Neuron cell types -> K = {1 + len(cat.dn_types)} parallel slots", flush=True)

    try:
        t0 = time.perf_counter()
        _assert_gpu_csr_synced(ds_m)
        real_screen = run_parallel_dn_perturbation_screen(ds_m, cat, seeds=DN_SCREEN_SEEDS, stim_ms=80, bg_noise_hz=20.0)
        print(f"  Real connectome screen ({len(DN_SCREEN_SEEDS)} CRN seeds) finished in {time.perf_counter()-t0:.1f}s", flush=True)

        # Control 1: Type + Side Rewired Null Screen
        snap.restore(ds_m)
        ts_rep = rewire_within_type_side(ds_m, seed=11)
        _assert_gpu_csr_synced(ds_m)
        t1 = time.perf_counter()
        ts_screen = run_parallel_dn_perturbation_screen(ds_m, cat, seeds=DN_SCREEN_SEEDS, stim_ms=80, bg_noise_hz=20.0)
        print(f"  Type+side null screen finished in {time.perf_counter()-t1:.1f}s", flush=True)

        # Control 2: Full Configuration-Model Rewired Null Screen
        snap.restore(ds_m)
        full_rep = rewire_degree_preserving(ds_m, seed=11)
        _assert_gpu_csr_synced(ds_m)
        t2 = time.perf_counter()
        full_screen = run_parallel_dn_perturbation_screen(ds_m, cat, seeds=DN_SCREEN_SEEDS, stim_ms=80, bg_noise_hz=20.0)
        print(f"  Full-shuffle null screen finished in {time.perf_counter()-t2:.1f}s", flush=True)
    finally:
        snap.restore(ds_m)

    # Merge null-screen deltas into real_screen table for direct hub-confound comparison
    ts_by_type = {r["dn_type"]: r for r in ts_screen["dn_table"]}
    full_by_type = {r["dn_type"]: r for r in full_screen["dn_table"]}
    merged_rows = []
    for r in real_screen["dn_table"]:
        tp = r["dn_type"]
        rt = ts_by_type[tp]
        rf = full_by_type[tp]
        merged_rows.append({
            **r,
            "type_side_null_delta_vnc_leg_sep_hz": rt["delta_vnc_leg_sep_hz"],
            "type_side_null_delta_vnc_wing_mn_hz": rt["delta_vnc_wing_mn_hz"],
            "full_null_delta_vnc_leg_sep_hz": rf["delta_vnc_leg_sep_hz"],
            "full_null_delta_vnc_wing_mn_hz": rf["delta_vnc_wing_mn_hz"],
        })

    top_steering_leg = sorted(merged_rows, key=lambda x: x["delta_vnc_leg_sep_hz"], reverse=True)[:15]
    top_steering_readout = sorted(merged_rows, key=lambda x: x["delta_turn_readout_hz"], reverse=True)[:15]
    top_courtship_wing = sorted(merged_rows, key=lambda x: x["delta_vnc_wing_mn_hz"], reverse=True)[:15]

    syn_arr = np.array([r["vnc_out_syn"] for r in merged_rows], dtype=float)
    leg_arr = np.array([r["delta_vnc_leg_sep_hz"] for r in merged_rows], dtype=float)
    wing_arr = np.array([r["delta_vnc_wing_mn_hz"] for r in merged_rows], dtype=float)
    sp_leg, p_sp_leg = spearmanr(syn_arr, leg_arr)
    sp_wing, p_sp_wing = spearmanr(syn_arr, wing_arr)

    print("  Top 5 DN Steering Bottlenecks (VNC Leg MN Asymmetry):",
          [(r["dn_type"], r["delta_vnc_leg_sep_hz"]) for r in top_steering_leg[:5]], flush=True)
    print("  Top 5 DN Courtship Song Bottlenecks (VNC Wing MNs):",
          [(r["dn_type"], r["delta_vnc_wing_mn_hz"]) for r in top_courtship_wing[:5]], flush=True)

    return {
        "n_dn_types_screened": real_screen["n_dn_types_screened"],
        "parallel_k_slots": real_screen["parallel_k_slots"],
        "crn_unperturbed_max_diff_hz": real_screen["crn_unperturbed_max_diff_hz"],
        "wildtype_baselines": {
            "real_vnc_leg_sep_hz": real_screen["wt_vnc_leg_sep_hz_mean"],
            "real_turn_sep_hz": real_screen["wt_turn_sep_hz_mean"],
            "real_vnc_wing_mn_hz": real_screen["wt_vnc_wing_mn_hz_mean"],
            "type_side_vnc_leg_sep_hz": ts_screen["wt_vnc_leg_sep_hz_mean"],
            "type_side_vnc_wing_mn_hz": ts_screen["wt_vnc_wing_mn_hz_mean"],
            "full_null_vnc_leg_sep_hz": full_screen["wt_vnc_leg_sep_hz_mean"],
            "full_null_vnc_wing_mn_hz": full_screen["wt_vnc_wing_mn_hz_mean"],
        },
        "hub_confound_audit": {
            "pearson_r_vnc_syn_vs_leg_impact": real_screen["pearson_r_vnc_syn_vs_leg_impact"],
            "pearson_r_vnc_syn_vs_wing_impact": real_screen["pearson_r_vnc_syn_vs_wing_impact"],
            "spearman_rho_vnc_syn_vs_leg_impact": round(float(sp_leg), 4),
            "spearman_p_leg": round(float(p_sp_leg), 6),
            "spearman_rho_vnc_syn_vs_wing_impact": round(float(sp_wing), 4),
            "spearman_p_wing": round(float(p_sp_wing), 6),
        },
        "null_validation": {"type_side": ts_rep, "full": full_rep},
        "top_15_steering_vnc_leg_bottlenecks": top_steering_leg,
        "top_15_steering_turn_readout_bottlenecks": top_steering_readout,
        "top_15_courtship_vnc_wing_bottlenecks": top_courtship_wing,
        "all_dn_types": merged_rows,
    }


# ---------------------------------------------------------------------------
# Pillar 3: Cross-Dataset Threshold Audit & Symmetric fru/dsx Sex-Switch
# ---------------------------------------------------------------------------
def _run_sex_switch_suite_one_sex(
    ds: ConnectomeDataset,
    sex_label: str,
    eval_fn: Callable[[ConnectomeDataset], Dict[str, float]],
    rewire_seeds: Sequence[int],
) -> Dict[str, object]:
    snap = CSRSnapshot.capture(ds)
    sex_mask = extract_sex_switch_mask(ds)
    rows = np.repeat(np.arange(ds.N, dtype=np.int32), np.diff(ds.W_base_csr.indptr))
    cols = ds.W_base_csr.indices
    n_both = int(np.sum(sex_mask[rows] & sex_mask[cols]))
    n_touch = int(np.sum(sex_mask[rows] | sex_mask[cols]))

    cond_specs: List[Tuple[str, Optional[Callable[[ConnectomeDataset, int], Dict[str, object]]]]] = [
        ("real", None),
        ("rewired_fru_both", lambda d, s: rewire_fru_dsx_subgraph(d, s, mode="both")),
        ("rewired_iso_matched_both", lambda d, s, nb=n_both: rewire_isomorphic_matched(d, s, nb)),
        ("rewired_fru_touch", lambda d, s: rewire_fru_dsx_subgraph(d, s, mode="touch")),
        ("rewired_iso_matched_touch", lambda d, s, nt=n_touch: rewire_isomorphic_matched(d, s, nt)),
        ("rewired_all_isomorphic", rewire_all_isomorphic),
        ("rewired_full", rewire_degree_preserving),
    ]

    trials: List[Dict[str, object]] = []
    validations: List[Dict[str, object]] = []
    try:
        for cname, fn in cond_specs:
            t_c = time.perf_counter()
            if fn is None:
                snap.restore(ds)
                _assert_gpu_csr_synced(ds)
                for b_seed in (600, 610, 620, 630, 640):
                    m = eval_fn(ds, engine_seeds=tuple(b_seed + i for i in range(5)))
                    trials.append({"condition": cname, "rewire_seed": None, "fraction_edges_rewired": 0.0, **m})
            else:
                for rs in rewire_seeds:
                    snap.restore(ds)
                    rep = fn(ds, int(rs))
                    _assert_gpu_csr_synced(ds)
                    rep["condition"] = cname
                    validations.append(rep)
                    m = eval_fn(ds, engine_seeds=tuple(600 + 10 * int(rs) + i for i in range(5)))
                    trials.append({
                        "condition": cname,
                        "rewire_seed": int(rs),
                        "fraction_edges_rewired": rep["fraction_edges_rewired"],
                        **m,
                    })
            sub = [r for r in trials if r["condition"] == cname]
            fr = 100.0 * float(np.mean([r["fraction_edges_rewired"] for r in sub]))
            print(f"  [{sex_label}] {cname:26s} (rewired={fr:5.2f}%, {time.perf_counter()-t_c:4.1f}s): {sub[0]}", flush=True)
    finally:
        snap.restore(ds)

    metric_names = [k for k in trials[0].keys() if k not in ("condition", "rewire_seed")]
    summary = {}
    for cname, _ in cond_specs:
        sub = [r for r in trials if r["condition"] == cname]
        summary[cname] = {mk: _mean_sd([r[mk] for r in sub]) for mk in metric_names}

    def _svals(c: str, mk: str) -> List[float]:
        return [float(r[mk]) for r in trials if r["condition"] == c]

    stat_tests: Dict[str, float] = {
        "nav_fru_touch_vs_iso_matched_touch_mwu_p": _mwu_p(
            _svals("rewired_fru_touch", "nav_turn_sep_hz"),
            _svals("rewired_iso_matched_touch", "nav_turn_sep_hz"),
        ),
        "nav_fru_touch_vs_all_isomorphic_mwu_p": _mwu_p(
            _svals("rewired_fru_touch", "nav_turn_sep_hz"),
            _svals("rewired_all_isomorphic", "nav_turn_sep_hz"),
        ),
    }
    if "courtship_pip10_aroused_hz" in metric_names:
        stat_tests["pip10_fru_touch_vs_iso_matched_touch_mwu_p"] = _mwu_p(
            _svals("rewired_fru_touch", "courtship_pip10_aroused_hz"),
            _svals("rewired_iso_matched_touch", "courtship_pip10_aroused_hz"),
        )
        stat_tests["wing_mn_fru_touch_vs_iso_matched_touch_mwu_p"] = _mwu_p(
            _svals("rewired_fru_touch", "courtship_wing_mn_hz"),
            _svals("rewired_iso_matched_touch", "courtship_wing_mn_hz"),
        )
        stat_tests["unaroused_leak_real_vs_fru_both_mwu_p"] = _mwu_p(
            _svals("real", "courtship_pip10_unaroused_leak_hz"),
            _svals("rewired_fru_both", "courtship_pip10_unaroused_leak_hz"),
        )
    if "receptivity_vpodn_hz" in metric_names:
        stat_tests["vpodn_fru_touch_vs_iso_matched_touch_mwu_p"] = _mwu_p(
            _svals("rewired_fru_touch", "receptivity_vpodn_hz"),
            _svals("rewired_iso_matched_touch", "receptivity_vpodn_hz"),
        )

    return {
        "sex": sex_label,
        "n_neurons_total": int(ds.N),
        "n_sex_switch_neurons": int(sex_mask.sum()),
        "pct_sex_switch_neurons": round(100.0 * float(sex_mask.sum()) / float(ds.N), 3),
        "edges_both_endpoints": n_both,
        "edges_touch_endpoint": n_touch,
        "statistical_tests": stat_tests,
        "validation": validations,
        "trials": trials,
        "summary": summary,
    }


def run_pillar_3_sex_switch_and_threshold(
    ds_m: ConnectomeDataset, ds_f: ConnectomeDataset, rewire_seeds: Sequence[int]
) -> Dict[str, object]:
    print("\n=== [Pillar 3A] Cross-Dataset Synapse Threshold & FIB-SEM vs TEM Recall Audit ===", flush=True)
    audit = compute_synapse_threshold_audit(ds_m, ds_f)
    for row in audit["threshold_curve"]:
        print(
            f"  |w| >= {row['min_synapses']:2d}: Male={row['male_edges']:10,d} edges | "
            f"Female={row['female_edges']:9,d} edges | Ratio={row['male_to_female_edge_ratio']:5.2f}x",
            flush=True,
        )
    print("  Functional gate check:", audit["functional_gate_comparison"], flush=True)

    print("\n=== [Pillar 3B] Symmetric Within-Connectome fru/dsx Double Dissociation ===", flush=True)
    male_res = _run_sex_switch_suite_one_sex(
        ds_m, "male", evaluate_sex_switch_double_dissociation_male, rewire_seeds
    )
    female_res = _run_sex_switch_suite_one_sex(
        ds_f, "female", evaluate_sex_switch_double_dissociation_female, rewire_seeds
    )

    return {
        "threshold_and_recall_audit": audit,
        "male_within_connectome_dissociation": male_res,
        "female_within_connectome_dissociation": female_res,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Option A whole-CNS structural specificity & screen experiments.")
    ap.add_argument("--part", choices=("all", "hierarchy", "screen", "sex_switch"), default="all")
    ap.add_argument("--rewire-seeds", type=int, default=len(REWIRE_SEEDS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    t_start = time.perf_counter()
    r_seeds = REWIRE_SEEDS[: int(args.rewire_seeds)]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = (
        json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    )
    payload["hardware_provenance"] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "triton_available": bool(_HAS_TRITON),
        "pytorch_version": torch.__version__,
        "n_rewire_seeds": len(r_seeds),
        "rewire_seeds": list(r_seeds),
    }
    print("Hardware provenance:", payload["hardware_provenance"], flush=True)

    ds_m = ConnectomeDataset()
    ds_f = ConnectomeDataset(str(resolve_female_connectome_dir())) if args.part in ("all", "sex_switch") else None

    if args.part in ("all", "hierarchy"):
        payload["pillar_1_wiring_hierarchy"] = run_pillar_1_hierarchy(ds_m, r_seeds)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.part in ("all", "screen"):
        payload["pillar_2_virtual_split_gal4_dn_screen"] = run_pillar_2_dn_screen(ds_m)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.part in ("all", "sex_switch"):
        assert ds_f is not None
        payload["pillar_3_sex_switch_and_threshold_audit"] = run_pillar_3_sex_switch_and_threshold(
            ds_m, ds_f, r_seeds
        )
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    payload["elapsed_wall_time_s"] = round(time.perf_counter() - t_start, 2)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved complete Option A results -> {out_path} ({payload['elapsed_wall_time_s']}s)", flush=True)


if __name__ == "__main__":
    main()
