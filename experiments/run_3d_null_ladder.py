"""
Closed-Loop 3D NeuroMechFly v2 Null-Model Ladder (MaleCNS v1.0 & Female BANC v888)
==================================================================================
Evaluates single-connectome (K = 1) closed-loop 3D goal-directed walking and steering
in NeuroMechFly v2 (FlyGym 2.1.1 / MuJoCo 3.16, 0.48 s episodes) across eight
degree-, weight- and sign-preserving conditions in MaleCNS v1.0 and Female BANC v888:
  - real, rewired_type_side, partial_matched_type_side, rewired_type,
    partial_matched_type, rewired_region, rewired (full shuffle), zero_weights

Usage:
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python experiments/run_3d_null_ladder.py --workers 16
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.connectome_engine import (  # noqa: E402
    _HAS_TRITON,
    ConnectomeDataset,
    WholeCNSEngine,
    resolve_compiled_connectome_dir,
    resolve_female_connectome_dir,
)
from src.structural_specificity import (  # noqa: E402
    rewire_degree_preserving,
    rewire_partial_matched,
    rewire_within_region,
    rewire_within_type,
    rewire_within_type_side,
)
from src.vnc_neuromechfly_3d import EmbodiedNeuroMechFly3DArena  # noqa: E402

EPISODE = dict(target_xy=(5.6, 2.7), duration_s=0.48, frame_ms=8)
ENGINE_SEEDS = {"female": (7101, 7102, 7103), "male": (7201, 7202, 7203)}
REWIRE_SEEDS = (11, 22, 33)
STRUCTURED_NULLS = ("rewired", "rewired_region", "rewired_type", "rewired_type_side")
MATCHED_TO = {"partial_matched_type_side": "rewired_type_side", "partial_matched_type": "rewired_type"}
LADDER_CONDITIONS = ("real", *STRUCTURED_NULLS, *MATCHED_TO, "zero_weights")
TURN_SATURATION_HZ = 0.44 * 38.0
DEFAULT_OUT = REPO / "results" / "rewiring_3d_results.json"

NULL_OPERATORS = {
    "rewired": rewire_degree_preserving,
    "rewired_region": rewire_within_region,
    "rewired_type": rewire_within_type,
    "rewired_type_side": rewire_within_type_side,
}


def compiled_dir(sex: str) -> str:
    return str(resolve_compiled_connectome_dir() if sex == "male" else resolve_female_connectome_dir())


def assert_gpu_matches_cpu(ds: ConnectomeDataset) -> None:
    if not getattr(ds, "_cuda_ready", False):
        return
    W = ds.W_base_csr
    assert np.array_equal(ds.csr_indptr_gpu.cpu().numpy(), W.indptr.astype(np.int32)), "GPU indptr stale"
    assert np.array_equal(ds.csr_indices_gpu.cpu().numpy(), W.indices.astype(np.int32)), "GPU indices stale"
    assert np.array_equal(ds.csr_data_gpu.cpu().numpy(), W.data.astype(np.float32)), "GPU weights stale"


def sync_gpu_copy(ds: ConnectomeDataset) -> None:
    if not getattr(ds, "_cuda_ready", False):
        return
    ds._cuda_ready = False
    ds.W_T_gpu = None
    ds.ensure_cuda_tensors(need_coo_fallback=not _HAS_TRITON)
    assert_gpu_matches_cpu(ds)


def zero_synaptic_weights(ds: ConnectomeDataset) -> None:
    ds.W_base_csr.data[:] = 0.0
    if getattr(ds, "_cuda_ready", False):
        ds.csr_data_gpu.zero_()
        if ds.W_T_gpu is not None:
            ds.W_T_gpu._values().zero_()
        assert_gpu_matches_cpu(ds)


def fresh_dataset(
    rewire_seed: Optional[int], null: str = "rewired", cdir: Optional[str] = None
) -> Tuple[ConnectomeDataset, Optional[Dict]]:
    ds = ConnectomeDataset(cdir)
    report = NULL_OPERATORS[null](ds, rewire_seed) if rewire_seed is not None else None
    if report is not None:
        sync_gpu_copy(ds)
        report["condition"] = null
        report["edges"] = int(ds.W_base_csr.nnz)
    return ds, report


def turn_readout_probe(eng: WholeCNSEngine, ds: ConnectomeDataset, n_ms: int = 60) -> Dict[str, float]:
    probes = {
        "loom_left": [(ds.idx_lc16_L, 245.0)],
        "loom_right": [(ds.idx_lc16_R, 255.0)],
        "goal_left": [(ds.idx_pfl3_turnL, 240.0)],
        "goal_right": [(ds.idx_pfl3_turnR, 250.0)],
    }
    out: Dict[str, float] = {}
    for name, stim in probes.items():
        eng.reset_state()
        res = eng.step_population(
            n_steps_ms=n_ms,
            shared_stimuli=[(ds.idx_walk_tonic, 80.0)] + stim,
            bg_poisson_hz=0.0,
            bg_subset_indices=None,
        )
        out[name] = round(float(res["consensus_turn_hz"]), 4)
    out["loom_separation_hz"] = round(out["loom_left"] - out["loom_right"], 4)
    out["goal_separation_hz"] = round(out["goal_right"] - out["goal_left"], 4)
    return out


def summarise_episode(ep: Dict) -> Dict:
    traj = ep["trajectory"]
    cadence = np.array([f["cadence_hz"] for f in traj], dtype=float)
    turns = np.abs(np.array([f["turn_hz"] for f in traj], dtype=float))
    keys = (
        "intact_fraction", "mean_speed_mm_s", "net_displacement_mm", "target_progress_pct",
        "final_target_dist_mm", "roll_rms_deg", "pitch_rms_deg", "thorax_height_z_mm", "motor_turn_snr",
    )
    return {
        **{k: ep[k] for k in keys},
        "mean_cadence_hz": round(float(cadence.mean()), 3),
        "mean_fwd_drive": round(float(cadence.mean()) / 13.5, 4),
        "mean_abs_consensus_turn_hz": round(float(turns.mean()), 3),
        "consensus_turn_saturated_frac": round(float(np.mean(turns >= TURN_SATURATION_HZ)), 3),
    }


def build_dataset(sex: str, cond: str, seed: Optional[int]) -> Tuple[ConnectomeDataset, Optional[Dict]]:
    if cond in ("real", "zero_weights"):
        return fresh_dataset(None, cdir=compiled_dir(sex))
    if cond in MATCHED_TO:
        ref_ds, ref = fresh_dataset(seed, null=MATCHED_TO[cond], cdir=compiled_dir(sex))
        del ref_ds
        ds = ConnectomeDataset(compiled_dir(sex))
        rep = rewire_partial_matched(ds, seed, fraction=ref["fraction_edges_rewired"])
        sync_gpu_copy(ds)
        rep.update(condition=cond, matched_to=MATCHED_TO[cond], edges=int(ds.W_base_csr.nnz))
        return ds, rep
    return fresh_dataset(seed, null=cond, cdir=compiled_dir(sex))


def ladder_job(sex: str, cond: str, rewire_seed: Optional[int], n_engine_seeds: int, duration_s: float) -> Dict:
    t0 = time.perf_counter()
    ds, report = build_dataset(sex, cond, rewire_seed)
    if report is not None:
        report["sex"] = sex
    real_data = ds.W_base_csr.data.copy() if cond == "zero_weights" else None

    rows: List[Dict] = []
    for es in ENGINE_SEEDS[sex][:n_engine_seeds]:
        if cond == "zero_weights":
            ds.W_base_csr.data[:] = real_data
            sync_gpu_copy(ds)
        eng = WholeCNSEngine(ds, num_brains=1, enable_callosum=True, enable_kc_mbon_matrix=True, seed=es)
        if cond == "zero_weights":
            zero_synaptic_weights(ds)
        assert_gpu_matches_cpu(ds)
        probe = turn_readout_probe(eng, ds)
        ep = EmbodiedNeuroMechFly3DArena.run_single_fly_3d_episode(eng, **{**EPISODE, "duration_s": duration_s})
        rows.append({
            "sex": sex,
            "condition": cond,
            "rewire_seed": rewire_seed,
            "engine_seed": es,
            **summarise_episode(ep),
            **{f"probe_{k}": v for k, v in probe.items()},
        })

    speeds = ", ".join(f"{r['mean_speed_mm_s']:.2f}" for r in rows)
    print(
        f"  [{sex:<6}] {cond:<26} seed={str(rewire_seed):<4} speed={speeds} mm/s "
        f"progress={np.mean([r['target_progress_pct'] for r in rows]):5.1f}% "
        f"({time.perf_counter() - t0:.0f}s)",
        flush=True,
    )
    return {"rows": rows, "validation": report}


def summarise_ladder(rows: List[Dict]) -> Dict:
    metrics = (
        "mean_speed_mm_s", "target_progress_pct", "final_target_dist_mm", "roll_rms_deg",
        "mean_fwd_drive", "mean_abs_consensus_turn_hz", "probe_goal_separation_hz", "probe_loom_separation_hz",
    )
    out: Dict[str, Dict] = {}
    for sex in sorted({r["sex"] for r in rows}):
        out[sex] = {}
        for cond in LADDER_CONDITIONS:
            sel = [r for r in rows if r["sex"] == sex and r["condition"] == cond]
            if sel:
                out[sex][cond] = {"n": len(sel), **{m: round(float(np.mean([r[m] for r in sel])), 3) for m in metrics}}
    return out


def run_ladder(sexes, conditions, workers, n_rewire_seeds, n_engine_seeds, duration_s) -> Dict:
    jobs = []
    for sex in sexes:
        for cond in conditions:
            seeds = [None] if cond in ("real", "zero_weights") else list(REWIRE_SEEDS[:n_rewire_seeds])
            jobs += [(sex, cond, s) for s in seeds]
    jobs.sort(key=lambda j: (j[0] != "male", j[2] is None))

    args = [(sex, cond, s, n_engine_seeds, duration_s) for sex, cond, s in jobs]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            outcomes = list(pool.map(ladder_job, *zip(*args)))
    else:
        outcomes = [ladder_job(*a) for a in args]

    rows = [r for o in outcomes for r in o["rows"]]
    return {
        "settings": {
            **EPISODE,
            "duration_s": duration_s,
            "K": 1,
            "rewire_seeds": list(REWIRE_SEEDS[:n_rewire_seeds]),
            "engine_seeds": {s: list(ENGINE_SEEDS[s][:n_engine_seeds]) for s in sexes},
        },
        "matched_to": MATCHED_TO,
        "validation": [o["validation"] for o in outcomes if o["validation"] is not None],
        "rows": rows,
        "summary": summarise_ladder(rows),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Closed-loop 3D NeuroMechFly v2 null-model ladder.")
    p.add_argument("--sex", nargs="+", choices=["female", "male"], default=["female", "male"])
    p.add_argument("--conditions", nargs="+", choices=LADDER_CONDITIONS, default=list(LADDER_CONDITIONS))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--rewire-seeds", type=int, default=3)
    p.add_argument("--engine-seeds", type=int, default=3)
    p.add_argument("--duration", type=float, default=EPISODE["duration_s"])
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = {"ladder": run_ladder(args.sex, args.conditions, args.workers, args.rewire_seeds, args.engine_seeds, args.duration)}
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
