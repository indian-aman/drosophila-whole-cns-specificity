"""
Whole-CNS Structural Specificity Hierarchy, Parallel Virtual Split-GAL4 Screening,
and Unconfounded fru/dsx Sex-Switch Dissection on MaleCNS v1.0 and Female BANC v888.

Every null model preserves each neuron's exact in-degree, out-degree, outgoing
synapse-weight multiset, and Dale's-law neurotransmitter sign via block-constrained
CSR target permutation.
"""
from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import sparse

from src.connectome_engine import ConnectomeDataset, WholeCNSEngine, extract_female_social_pools


# ---------------------------------------------------------------------------
# CSR Snapshot, Restoration, and Degree/Weight/Sign-Preserving Null Operators
# ---------------------------------------------------------------------------
@dataclass
class CSRSnapshot:
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray

    @classmethod
    def capture(cls, ds: ConnectomeDataset) -> "CSRSnapshot":
        W = ds.W_base_csr
        return cls(
            indptr=W.indptr.copy(),
            indices=W.indices.copy(),
            data=W.data.copy(),
        )

    def restore(self, ds: ConnectomeDataset) -> None:
        W = ds.W_base_csr
        W.indptr[:] = self.indptr
        W.indices[:] = self.indices
        W.data[:] = self.data
        W.has_sorted_indices = True
        sync_gpu_csr(ds)


def sync_gpu_csr(ds: ConnectomeDataset) -> None:
    """Synchronize cached GPU CSR/COO tensors whenever CPU W_base_csr is mutated."""
    if not getattr(ds, "_cuda_ready", False):
        return

    W = ds.W_base_csr
    ds.csr_indptr_gpu = torch.from_numpy(W.indptr.astype(np.int32)).to("cuda")
    ds.csr_indices_gpu = torch.from_numpy(W.indices.astype(np.int32)).to("cuda")
    ds.csr_data_gpu = torch.from_numpy(W.data.astype(np.float32)).to("cuda")
    if getattr(ds, "W_T_gpu", None) is not None:
        W_T = W.T.tocoo()
        indices_2d = np.vstack((W_T.row, W_T.col)).astype(np.int64)
        ds.W_T_gpu = torch.sparse_coo_tensor(
            torch.from_numpy(indices_2d),
            torch.from_numpy(W_T.data.astype(np.float32)),
            size=(ds.N, ds.N),
            device="cuda",
        ).coalesce()


def _free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def fraction_edges_rewired(
    indptr: np.ndarray, old_indices: np.ndarray, new_indices: np.ndarray, n: int
) -> float:
    """Exact multiset fraction of (source, target) directed edges absent from original CSR."""
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    old_codes, old_counts = np.unique(rows * n + old_indices, return_counts=True)
    new_codes, new_counts = np.unique(rows * n + new_indices, return_counts=True)
    _, i_old, i_new = np.intersect1d(old_codes, new_codes, assume_unique=True, return_indices=True)
    kept = int(np.minimum(old_counts[i_old], new_counts[i_new]).sum())
    return 1.0 - kept / float(max(1, len(old_indices)))


def permute_within_blocks(ds: ConnectomeDataset, block_key: np.ndarray, seed: int) -> Dict[str, object]:
    """
    Permute CSR target indices strictly within equal `block_key` classes.
    Preserves exact out-degree, exact in-degree, per-row outgoing weight multiset,
    and per-row neurotransmitter sign.
    """
    W = ds.W_base_csr
    old_indices = W.indices.copy()
    out_deg_before = np.diff(W.indptr)
    in_deg_before = np.bincount(old_indices, minlength=ds.N)

    rng = np.random.default_rng(int(seed))
    rand = rng.random(W.nnz)
    pos = np.argsort(block_key, kind="stable")
    order = np.lexsort((rand, block_key))

    new_indices = old_indices.copy()
    new_indices[pos] = old_indices[order]
    W.indices = new_indices.astype(np.int32)
    W.has_sorted_indices = False
    W.sort_indices()
    sync_gpu_csr(ds)

    assert np.array_equal(out_deg_before, np.diff(W.indptr)), "out-degree invariant violated"
    assert np.array_equal(in_deg_before, np.bincount(W.indices, minlength=ds.N)), "in-degree invariant violated"

    rows = np.repeat(np.arange(ds.N, dtype=np.int32), out_deg_before)
    row_sign = np.sign(ds.nt_sign[rows])
    nz = W.data != 0.0
    sign_ok = float(np.mean(np.sign(W.data[nz]) == row_sign[nz])) if np.any(nz) else 1.0
    return {
        "rewire_seed": int(seed),
        "edges": int(W.nnz),
        "blocks": int(len(np.unique(block_key))),
        "fraction_edges_rewired": round(fraction_edges_rewired(W.indptr, old_indices, W.indices, ds.N), 6),
        "self_loops_created": int(np.sum(W.indices == rows)),
        "per_row_sign_preserved": round(sign_ok, 6),
        "out_degree_preserved": True,
        "in_degree_preserved": True,
    }


def rewire_degree_preserving(ds: ConnectomeDataset, seed: int) -> Dict[str, object]:
    """Full configuration-model target permutation across the entire CNS."""
    key = np.zeros(ds.W_base_csr.nnz, dtype=np.int64)
    rep = permute_within_blocks(ds, key, seed)
    rep["null"] = "rewired_full"
    return rep


def rewire_within_region(ds: ConnectomeDataset, seed: int) -> Dict[str, object]:
    """Preserve 5 x 5 anatomical super-region connectivity blocks."""
    rows = np.repeat(np.arange(ds.N, dtype=np.int64), np.diff(ds.W_base_csr.indptr))
    key = ds.region_id[rows].astype(np.int64) * 256 + ds.region_id[ds.W_base_csr.indices].astype(np.int64)
    rep = permute_within_blocks(ds, key, seed)
    rep["null"] = "rewired_region"
    return rep


def rewire_within_type(ds: ConnectomeDataset, seed: int) -> Dict[str, object]:
    """Preserve cell-type x cell-type blocks (mixes hemispheres for bilateral types)."""
    types = ds.meta_df["type"].fillna("").astype(str).to_numpy()
    _, tc = np.unique(types, return_inverse=True)
    tc = tc.astype(np.int64)
    n_t = int(tc.max()) + 1
    rows = np.repeat(np.arange(ds.N, dtype=np.int64), np.diff(ds.W_base_csr.indptr))
    key = tc[rows] * n_t + tc[ds.W_base_csr.indices]
    rep = permute_within_blocks(ds, key, seed)
    rep["null"] = "rewired_type"
    rep["n_groups"] = n_t
    return rep


def rewire_within_type_side(ds: ConnectomeDataset, seed: int) -> Dict[str, object]:
    """Preserve (type, somaSide) x (type, somaSide) blocks."""
    types = ds.meta_df["type"].fillna("").astype(str)
    sides = ds.meta_df["somaSide"].fillna("").astype(str)
    _, grp = np.unique((types + "|" + sides).to_numpy(), return_inverse=True)
    grp = grp.astype(np.int64)
    n_g = int(grp.max()) + 1
    rows = np.repeat(np.arange(ds.N, dtype=np.int64), np.diff(ds.W_base_csr.indptr))
    key = grp[rows] * n_g + grp[ds.W_base_csr.indices]
    rep = permute_within_blocks(ds, key, seed)
    rep["null"] = "rewired_type_side"
    rep["n_groups"] = n_g
    return rep


def rewire_within_instance(ds: ConnectomeDataset, seed: int) -> Dict[str, object]:
    """
    Preserve (type, somaSide, instance) x (type, somaSide, instance) blocks.
    Keeps glomerular/columnar wedge labels (e.g. EPG(PB08)_L5, Delta7(PB15)_L5R4_L)
    intact while permuting individual sister neurons within each sub-type instance.
    """
    types = ds.meta_df["type"].fillna("").astype(str)
    sides = ds.meta_df["somaSide"].fillna("").astype(str)
    insts = ds.meta_df["instance"].fillna("").astype(str)
    _, grp = np.unique((types + "|" + sides + "|" + insts).to_numpy(), return_inverse=True)
    grp = grp.astype(np.int64)
    n_g = int(grp.max()) + 1
    rows = np.repeat(np.arange(ds.N, dtype=np.int64), np.diff(ds.W_base_csr.indptr))
    key = grp[rows] * n_g + grp[ds.W_base_csr.indices]
    rep = permute_within_blocks(ds, key, seed)
    rep["null"] = "rewired_instance"
    rep["n_groups"] = n_g
    return rep


def rewire_partial_matched(ds: ConnectomeDataset, seed: int, fraction: float) -> Dict[str, object]:
    """Randomly permute a matched fraction of edges across the whole graph."""
    rng = np.random.default_rng(int(seed) + 1000)
    chosen = rng.random(ds.W_base_csr.nnz) < float(fraction)
    key = np.where(chosen, 0, 1 + np.arange(ds.W_base_csr.nnz, dtype=np.int64))
    rep = permute_within_blocks(ds, key, seed)
    rep["null"] = "rewired_partial_matched"
    rep["fraction_selected"] = round(float(fraction), 6)
    return rep


def rewire_pn_kc_random_claws(ds: ConnectomeDataset, seed: int) -> Dict[str, object]:
    """
    Caron et al. (2013) developmental random-claw test:
    Permute ALL olfactory projection neuron -> Kenyon cell (ALPN -> KC) edges in a single
    degree-preserving block across all glomeruli and both hemispheres, leaving the rest of
    the CNS untouched.
    """
    df = ds.meta_df
    alpn_m = (
        df["class"].fillna("").str.contains("ALPN|olfactory_projection", case=False, regex=True)
        | df["type"].fillna("").str.contains("_adPN|_lPN|_vPN|_lvPN", regex=True)
    ).to_numpy()
    kc_m = (
        df["class"].fillna("").str.contains("kenyon", case=False, regex=True)
        | df["type"].fillna("").str.startswith("KC")
    ).to_numpy()
    rows = np.repeat(np.arange(ds.N, dtype=np.int32), np.diff(ds.W_base_csr.indptr))
    cols = ds.W_base_csr.indices
    in_block = alpn_m[rows] & kc_m[cols]
    key = np.where(in_block, 0, 1 + np.arange(ds.W_base_csr.nnz, dtype=np.int64))
    rep = permute_within_blocks(ds, key, seed)
    rep["null"] = "rewired_pn_kc_random_claws"
    rep["pn_kc_edges_selected"] = int(in_block.sum())
    return rep


def rewire_kc_mbon_compartments(ds: ConnectomeDataset, seed: int) -> Dict[str, object]:
    """Permute ALL KC -> MBON edges across all MBON compartments while keeping PN -> KC intact."""
    df = ds.meta_df
    kc_m = (
        df["class"].fillna("").str.contains("kenyon", case=False, regex=True)
        | df["type"].fillna("").str.startswith("KC")
    ).to_numpy()
    mbon_m = df["type"].fillna("").str.startswith("MBON").to_numpy()
    rows = np.repeat(np.arange(ds.N, dtype=np.int32), np.diff(ds.W_base_csr.indptr))
    cols = ds.W_base_csr.indices
    in_block = kc_m[rows] & mbon_m[cols]
    key = np.where(in_block, 0, 1 + np.arange(ds.W_base_csr.nnz, dtype=np.int64))
    rep = permute_within_blocks(ds, key, seed)
    rep["null"] = "rewired_kc_mbon_compartments"
    rep["kc_mbon_edges_selected"] = int(in_block.sum())
    return rep


def extract_sex_switch_mask(ds: ConnectomeDataset) -> np.ndarray:
    """Boolean mask of neurons annotated as fru+, dsx+, sex-specific, or sexually dimorphic."""
    df = ds.meta_df
    fru = df["fruDsx"].fillna("").astype(str) != ""
    dim = df["dimorphism"].fillna("").astype(str) != ""
    return (fru | dim).to_numpy()


def rewire_fru_dsx_subgraph(ds: ConnectomeDataset, seed: int, mode: str = "touch") -> Dict[str, object]:
    """
    Targeted Maslov-Sneppen rewiring of genetically tagged fru+/dsx+/dimorphic edges:
      - mode='both': both presynaptic and postsynaptic neurons are fru+/dsx+/dimorphic
      - mode='touch': at least one endpoint is fru+/dsx+/dimorphic
    """
    sex_mask = extract_sex_switch_mask(ds)
    rows = np.repeat(np.arange(ds.N, dtype=np.int32), np.diff(ds.W_base_csr.indptr))
    cols = ds.W_base_csr.indices
    if mode == "both":
        chosen = sex_mask[rows] & sex_mask[cols]
    elif mode == "touch":
        chosen = sex_mask[rows] | sex_mask[cols]
    else:
        raise ValueError(f"Unknown fru/dsx mode: {mode}")
    key = np.where(chosen, 0, 1 + np.arange(ds.W_base_csr.nnz, dtype=np.int64))
    rep = permute_within_blocks(ds, key, seed)
    rep["null"] = f"rewired_fru_{mode}"
    rep["edges_selected"] = int(chosen.sum())
    rep["sex_switch_neurons"] = int(sex_mask.sum())
    return rep


def rewire_isomorphic_matched(ds: ConnectomeDataset, seed: int, n_edges_to_select: int) -> Dict[str, object]:
    """
    Edge-count-matched Maslov-Sneppen control strictly inside the isomorphic
    (fru-/dsx-/non-dimorphic) subgraph.
    """
    sex_mask = extract_sex_switch_mask(ds)
    rows = np.repeat(np.arange(ds.N, dtype=np.int32), np.diff(ds.W_base_csr.indptr))
    cols = ds.W_base_csr.indices
    iso_pool = np.flatnonzero((~sex_mask[rows]) & (~sex_mask[cols]))
    n_sel = min(int(n_edges_to_select), len(iso_pool))
    rng = np.random.default_rng(int(seed) + 5000)
    chosen = np.zeros(ds.W_base_csr.nnz, dtype=bool)
    if n_sel > 0:
        chosen[rng.choice(iso_pool, size=n_sel, replace=False)] = True
    key = np.where(chosen, 0, 1 + np.arange(ds.W_base_csr.nnz, dtype=np.int64))
    rep = permute_within_blocks(ds, key, seed)
    rep["null"] = "rewired_iso_matched"
    rep["edges_selected"] = int(n_sel)
    return rep


def rewire_all_isomorphic(ds: ConnectomeDataset, seed: int) -> Dict[str, object]:
    """
    Complement perturbation: rewire ALL purely isomorphic (fru-/dsx-) edges (~89-98% of CNS)
    while leaving every edge touching the fru/dsx/dimorphic subgraph 100% intact.
    """
    sex_mask = extract_sex_switch_mask(ds)
    rows = np.repeat(np.arange(ds.N, dtype=np.int32), np.diff(ds.W_base_csr.indptr))
    cols = ds.W_base_csr.indices
    iso_only = (~sex_mask[rows]) & (~sex_mask[cols])
    key = np.where(iso_only, 0, 1 + np.arange(ds.W_base_csr.nnz, dtype=np.int64))
    rep = permute_within_blocks(ds, key, seed)
    rep["null"] = "rewired_all_isomorphic"
    rep["edges_selected"] = int(iso_only.sum())
    return rep


# ---------------------------------------------------------------------------
# Cross-Trial Multi-Class Decoding & Cosine Contrast (Zero-Norm Safe)
# ---------------------------------------------------------------------------
def leave_one_trial_out_cosine_metrics(resp_r_c_d: np.ndarray) -> Dict[str, float]:
    """
    Evaluate multi-class population coding across R noisy trials x C classes x D neurons.
    If a trial or class centroid is completely silent (zero norm), assigns exact uniform
    chance (1 / C) rather than deterministic class-0 tie-breaking, preventing dead-network
    coding illusions.
    """
    R, C, D = resp_r_c_d.shape
    chance = 1.0 / float(C)
    if R < 2 or C < 2 or D == 0:
        return {"decode_acc": chance, "coding_contrast": 0.0, "same_cos": 0.0, "diff_cos": 0.0}

    # L2-normalized vectors per (r, c)
    norms = np.linalg.norm(resp_r_c_d, axis=2, keepdims=True)
    alive = norms[..., 0] > 1e-8
    unit = np.where(norms > 1e-8, resp_r_c_d / np.maximum(norms, 1e-12), 0.0)

    # Pairwise cross-trial cosine similarity (only across DIFFERENT trials r1 != r2)
    same_vals: List[float] = []
    diff_vals: List[float] = []
    for r1 in range(R):
        for r2 in range(r1 + 1, R):
            sim_cc = unit[r1] @ unit[r2].T  # (C, C)
            same_vals.extend(np.diag(sim_cc).tolist())
            off = sim_cc[~np.eye(C, dtype=bool)]
            diff_vals.extend(off.tolist())

    same_cos = float(np.mean(same_vals)) if same_vals else 0.0
    diff_cos = float(np.mean(diff_vals)) if diff_vals else 0.0

    # Leave-one-trial-out nearest-centroid classification
    total_score = 0.0
    total_cases = R * C
    sum_all = resp_r_c_d.sum(axis=0)  # (C, D)
    for r in range(R):
        train_mean = (sum_all - resp_r_c_d[r]) / float(R - 1)  # (C, D)
        c_norms = np.linalg.norm(train_mean, axis=1, keepdims=True)
        c_alive = (c_norms[:, 0] > 1e-8)
        c_unit = np.where(c_norms > 1e-8, train_mean / np.maximum(c_norms, 1e-12), 0.0)

        for c in range(C):
            if (not alive[r, c]) or (not np.any(c_alive)):
                total_score += chance
                continue
            sims = c_unit @ unit[r, c]  # (C,)
            sims = np.where(c_alive, sims, -2.0)
            best = float(sims.max())
            ties = np.flatnonzero(np.abs(sims - best) <= 1e-7)
            if c in ties:
                total_score += 1.0 / float(len(ties))

    return {
        "decode_acc": round(total_score / float(total_cases), 4),
        "coding_contrast": round(same_cos - diff_cos, 4),
        "same_cos": round(same_cos, 4),
        "diff_cos": round(diff_cos, 4),
    }


# ---------------------------------------------------------------------------
# Pillar 1: Three-Tier Whole-CNS Structural Specificity Probes (MaleCNS v1.0)
# ---------------------------------------------------------------------------
@dataclass
class ThreeCircuitAnatomy:
    epg_wedges_8: List[np.ndarray]          # 8 circular EB wedges (L_k + R_{9-k}, k=1..8)
    cx_readout_pools: Dict[str, np.ndarray] # 44 glomerular Delta7 (12) + PEN_a/b (32) instance pools
    odor_glom_names: List[str]              # 8 major ALPN olfactory glomeruli
    odor_pn_pools: List[np.ndarray]         # PN indices per glomerulus
    kc_indices: np.ndarray                  # 4,064 Kenyon Cell indices
    kc_plastic_slots: np.ndarray            # slot indices in eng.elig for exact per-KC spike counts
    mbon_readout_pools: Dict[str, np.ndarray]  # 36 bilateral MBON cell-type pools


def extract_three_circuit_anatomy(ds: ConnectomeDataset) -> ThreeCircuitAnatomy:
    df = ds.meta_df
    # 1. Central Complex: 8 anatomical EPG wedges (PB08_L1+R8 .. PB08_L8+R1) + 44 Delta7/PEN instance pools
    wedges: List[np.ndarray] = []
    for k in range(1, 9):
        w_idx = df[df["instance"].isin([f"EPG(PB08)_L{k}", f"EPG(PB08)_R{9 - k}"])].index.to_numpy(dtype=np.int32)
        wedges.append(w_idx)

    cx_pools: Dict[str, np.ndarray] = {}
    d7_df = df[df["type"].fillna("").str.startswith("Delta7")]
    for i, inst in enumerate(sorted(d7_df["instance"].dropna().unique().tolist())):
        cx_pools[f"d7_{i:02d}"] = d7_df[d7_df["instance"] == inst].index.to_numpy(dtype=np.int32)
    pen_df = df[df["type"].fillna("").str.startswith("PEN")]
    for i, inst in enumerate(sorted(pen_df["instance"].dropna().unique().tolist())):
        cx_pools[f"pen_{i:02d}"] = pen_df[pen_df["instance"] == inst].index.to_numpy(dtype=np.int32)

    # 2. Mushroom Body: 8 major ALPN glomeruli -> 4,064 KCs -> 97 MBONs
    glom_names = [
        "DA1_lPN", "VM5d_adPN", "DL3_lPN", "DA2_lPN",
        "DL2d_adPN", "VA1v_adPN", "DL2v_adPN", "VM7d_adPN",
    ]
    alpn_df = df[df["class"] == "ALPN"]
    pn_pools = [alpn_df[alpn_df["type"] == g].index.to_numpy(dtype=np.int32) for g in glom_names]
    kc_idx = df[df["class"] == "Kenyon_Cell"].index.to_numpy(dtype=np.int32)
    kc_slots = ds.plastic_slot[kc_idx]
    assert np.all(kc_slots >= 0), "All Kenyon Cells must map to valid eligibility slots"

    mbon_df = df[df["class"] == "MBON"]
    mbon_pools: Dict[str, np.ndarray] = {}
    for i, tp in enumerate(sorted(mbon_df["type"].dropna().unique().tolist())):
        mbon_pools[f"mbon_{i:02d}"] = mbon_df[mbon_df["type"] == tp].index.to_numpy(dtype=np.int32)

    return ThreeCircuitAnatomy(
        epg_wedges_8=wedges,
        cx_readout_pools=cx_pools,
        odor_glom_names=glom_names,
        odor_pn_pools=pn_pools,
        kc_indices=kc_idx,
        kc_plastic_slots=kc_slots,
        mbon_readout_pools=mbon_pools,
    )


def evaluate_three_circuits_open_loop(
    ds: ConnectomeDataset,
    anat: ThreeCircuitAnatomy,
    n_trials: int = 5,
    stim_ms: int = 60,
    bg_noise_hz: float = 20.0,
    seed: int = 500,
) -> Dict[str, float]:
    """
    Evaluate all three canonical circuits in a single whole-CNS LIF model with
    identical biophysical constants (tau_m = 20 ms, tau_syn = 5 ms) and independent
    noisy trials (R = n_trials) per stimulus class:
      1. Motor Relay (PFL3 -> DN & VNC leg MNs)
      2. Central Complex Topographic Compass (8 EPG wedges -> 44 Delta7/PEN instances + post-cue persistence)
      3. Mushroom Body Olfactory Expansion (8 ALPN glomeruli -> 4,064 KCs -> 97 MBONs)
    """
    C = 8
    K = n_trials * C  # e.g., 5 * 8 = 40 parallel whole-CNS trajectories
    eng = WholeCNSEngine(
        ds,
        num_brains=K,
        enable_callosum=False,
        homogeneous_tau_ms=20.0,
        common_random_numbers=False,
        seed=seed,
    )
    bg_sub = ds.plastic_neurons[::8]

    # --- 1. Motor Relay: PFL3_R (first n_trials slots) vs PFL3_L (next n_trials slots) ---
    eng.reset_state()
    r_walk = np.zeros(K, dtype=np.float32)
    r_walk[: 2 * n_trials] = 80.0
    r_pR = np.zeros(K, dtype=np.float32)
    r_pR[:n_trials] = 250.0
    r_pL = np.zeros(K, dtype=np.float32)
    r_pL[n_trials : 2 * n_trials] = 240.0
    res_steer = eng.step_population(
        n_steps_ms=stim_ms,
        shared_stimuli=[
            (ds.idx_walk_tonic, r_walk),
            (ds.idx_pfl3_turnR, r_pR),
            (ds.idx_pfl3_turnL, r_pL),
        ],
        bg_poisson_hz=bg_noise_hz,
        bg_subset_indices=bg_sub,
    )
    turn_R = float(res_steer["turn_hz_per_brain"][:n_trials].mean())
    turn_L = float(res_steer["turn_hz_per_brain"][n_trials : 2 * n_trials].mean())
    leg_asym = res_steer["leg_R_hz"] - res_steer["leg_L_hz"]
    leg_sep = float(leg_asym[:n_trials].mean() - leg_asym[n_trials : 2 * n_trials].mean())

    # --- 2. Central Complex Compass: R trials x 8 EPG wedges + 50ms cue-OFF persistence check ---
    eng.reset_state()
    epg_self_pools = {f"epg_w{c}": anat.epg_wedges_8[c] for c in range(C)}
    all_cx_pools = {**anat.cx_readout_pools, **epg_self_pools}
    cx_stims = []
    for c in range(C):
        rv = np.zeros(K, dtype=np.float32)
        for r in range(n_trials):
            rv[r * C + c] = 280.0
        cx_stims.append((anat.epg_wedges_8[c], rv))

    res_cx_on = eng.step_population(
        n_steps_ms=stim_ms,
        shared_stimuli=cx_stims,
        bg_poisson_hz=bg_noise_hz,
        bg_subset_indices=bg_sub,
        readout_pools=all_cx_pools,
    )
    res_cx_off = eng.step_population(
        n_steps_ms=50,
        shared_stimuli=[],
        bg_poisson_hz=bg_noise_hz,
        bg_subset_indices=bg_sub,
        readout_pools=epg_self_pools,
    )

    cx_keys = sorted(anat.cx_readout_pools.keys())
    cx_flat = np.stack([res_cx_on["pool_hz"][k] for k in cx_keys], axis=1)  # (K, 44)
    cx_rcd = cx_flat.reshape(n_trials, C, len(cx_keys))
    cx_dec = leave_one_trial_out_cosine_metrics(cx_rcd)

    # Trial-averaged antipodal modulation (cos_45deg - cos_180deg)
    cx_mean_c = cx_rcd.mean(axis=0)  # (8, 44)
    c_nrm = np.linalg.norm(cx_mean_c, axis=1, keepdims=True)
    if np.sum(c_nrm[:, 0] > 1e-8) >= 4:
        cu = np.where(c_nrm > 1e-8, cx_mean_c / np.maximum(c_nrm, 1e-12), 0.0)
        cs8 = cu @ cu.T
        adj = [0.5 * (cs8[i, (i + 1) % 8] + cs8[i, (i - 1) % 8]) for i in range(8)]
        opp = [cs8[i, (i + 4) % 8] for i in range(8)]
        cx_antipodal = float(np.mean(adj) - np.mean(opp))
    else:
        cx_antipodal = 0.0

    on_rates, off_rates = [], []
    for c in range(C):
        slots_c = [r * C + c for r in range(n_trials)]
        on_rates.append(float(res_cx_on["pool_hz"][f"epg_w{c}"][slots_c].mean()))
        off_rates.append(float(res_cx_off["pool_hz"][f"epg_w{c}"][slots_c].mean()))
    epg_on_hz = float(np.mean(on_rates))
    epg_off_hz = float(np.mean(off_rates))
    epg_persist_ratio = epg_off_hz / max(epg_on_hz, 1e-6)

    # --- 3. Mushroom Body Olfactory Expansion: R trials x 8 ALPN glomeruli -> 4,064 KCs -> 97 MBONs ---
    eng.reset_state()
    mb_stims = []
    for c in range(C):
        rv = np.zeros(K, dtype=np.float32)
        for r in range(n_trials):
            rv[r * C + c] = 280.0
        mb_stims.append((anat.odor_pn_pools[c], rv))

    res_mb = eng.step_population(
        n_steps_ms=stim_ms,
        shared_stimuli=mb_stims,
        bg_poisson_hz=bg_noise_hz,
        bg_subset_indices=bg_sub,
        readout_pools=anat.mbon_readout_pools,
    )
    if eng.device == "cuda":
        elig_np = eng.elig_t.cpu().numpy()
    else:
        elig_np = eng.elig
    kc_flat = elig_np[:, anat.kc_plastic_slots].copy()  # (K, 4064)
    kc_sparsity = float((kc_flat > 0.1).mean())
    kc_rcd = kc_flat.reshape(n_trials, C, len(anat.kc_indices))
    mb_dec = leave_one_trial_out_cosine_metrics(kc_rcd)

    mbon_keys = sorted(anat.mbon_readout_pools.keys())
    mbon_flat = np.stack([res_mb["pool_hz"][k] for k in mbon_keys], axis=1)  # (K, 36)
    mbon_rcd = mbon_flat.reshape(n_trials, C, len(mbon_keys))
    mbon_dec = leave_one_trial_out_cosine_metrics(mbon_rcd)
    del eng
    _free_cuda()

    return {
        "steer_goal_sep_hz": round(turn_R - turn_L, 3),
        "steer_leg_mn_sep_hz": round(leg_sep, 4),
        "cx_decode_acc": cx_dec["decode_acc"],
        "cx_coding_contrast": cx_dec["coding_contrast"],
        "cx_antipodal_mod": round(cx_antipodal, 4),
        "cx_mean_hz": round(float(cx_flat.mean()), 3),
        "cx_epg_on_hz": round(epg_on_hz, 2),
        "cx_epg_off_hz": round(epg_off_hz, 2),
        "cx_epg_persistence_ratio": round(epg_persist_ratio, 4),
        "mb_kc_decode_acc": mb_dec["decode_acc"],
        "mb_kc_coding_contrast": mb_dec["coding_contrast"],
        "mb_kc_sparsity": round(kc_sparsity, 4),
        "mb_mbon_decode_acc": mbon_dec["decode_acc"],
        "mb_mbon_coding_contrast": mbon_dec["coding_contrast"],
        "mb_mbon_mean_hz": round(float(mbon_flat.mean()), 3),
    }


# ---------------------------------------------------------------------------
# Pillar 2: High-Throughput Virtual Split-GAL4 Screen (All 486 DN Cell Types)
# ---------------------------------------------------------------------------
@dataclass
class DescendingNeuronCatalog:
    dn_types: List[str]
    dn_indices_per_type: List[np.ndarray]
    dn_neuron_counts: List[int]
    dn_vnc_out_syn: List[float]
    dn_is_sex_switch: List[bool]
    wing_mn_indices: np.ndarray


def extract_dn_screen_catalog(ds: ConnectomeDataset) -> DescendingNeuronCatalog:
    df = ds.meta_df
    W = ds.W_base_csr
    is_dn = (
        df["superclass"].fillna("").str.contains("descending", case=False)
        | df["class"].fillna("").str.contains("descending", case=False)
    )
    dn_df = df[is_dn & (df["type"].fillna("") != "")].copy()
    dn_types = sorted(dn_df["type"].unique().tolist())

    sex_mask = extract_sex_switch_mask(ds)
    is_vnc = (ds.region_id == 2)  # VNC intrinsic + motor

    idx_list: List[np.ndarray] = []
    cnt_list: List[int] = []
    vnc_syn_list: List[float] = []
    sex_list: List[bool] = []

    for tp in dn_types:
        arr = dn_df[dn_df["type"] == tp].index.to_numpy(dtype=np.int32)
        idx_list.append(arr)
        cnt_list.append(int(len(arr)))
        sub = W[arr]
        vnc_mask_e = is_vnc[sub.indices]
        vnc_syn_list.append(float(np.abs(sub.data[vnc_mask_e]).sum()))
        sex_list.append(bool(np.any(sex_mask[arr])))

    wing_mn = df[
        df["exitNerve"].fillna("").str.contains("ADMN|PDMN|wing", case=False, regex=True)
        | df["subclass"].fillna("").str.contains("wing", case=False, regex=True)
    ].index.to_numpy(dtype=np.int32)

    return DescendingNeuronCatalog(
        dn_types=dn_types,
        dn_indices_per_type=idx_list,
        dn_neuron_counts=cnt_list,
        dn_vnc_out_syn=vnc_syn_list,
        dn_is_sex_switch=sex_list,
        wing_mn_indices=wing_mn,
    )


def run_parallel_dn_perturbation_screen(
    ds: ConnectomeDataset,
    cat: DescendingNeuronCatalog,
    seeds: Sequence[int] = (700, 701, 702),
    stim_ms: int = 80,
    bg_noise_hz: float = 20.0,
) -> Dict[str, object]:
    """
    Screen ALL Descending Neuron cell types simultaneously in a single K = 1 + N_DN
    Shared-CSR engine using Common Random Numbers (CRN):
      - Slot 0: Intact wild-type control
      - Slots 1..N_DN: Single-DN-type acute Kir2.1 + TNT silencing
    Measures strictly downstream VNC Motor Neuron pools (leg_R - leg_L for steering,
    and 495 wing MNs for courtship song) so silenced DNs never overlap the readout!
    """
    n_dn = len(cat.dn_types)
    K = 1 + n_dn
    df = ds.meta_df
    lc10a_R = df[(df["type"] == "LC10a") & (df["somaSide"] == "R")].index.to_numpy(dtype=np.int32)
    va1v = df[df["type"].fillna("").str.contains("VA1v", regex=True)].index.to_numpy(dtype=np.int32)
    pC1_P1_aIPg = df[df["type"].fillna("").str.contains("^pC1|^P1_|^aIPg", regex=True)].index.to_numpy(dtype=np.int32)
    bg_sub = ds.plastic_neurons[::8]

    silenced_masks: List[Optional[np.ndarray]] = [None] + list(cat.dn_indices_per_type)

    # Mandatory CRN identity sanity check (16 unperturbed slots must have 0.000000 spread)
    chk_eng = WholeCNSEngine(
        ds, num_brains=16, enable_callosum=False, homogeneous_tau_ms=20.0, common_random_numbers=True, seed=seeds[0]
    )
    chk_res = chk_eng.step_population(
        25, shared_stimuli=[(ds.idx_walk_tonic, 80.0), (ds.idx_pfl3_turnR, 250.0)], bg_poisson_hz=bg_noise_hz, bg_subset_indices=bg_sub
    )
    crn_max_diff = float(np.max(np.abs(chk_res["turn_hz_per_brain"] - chk_res["turn_hz_per_brain"][0])))
    assert crn_max_diff == 0.0, f"CRN identity violated: max diff = {crn_max_diff}"
    del chk_eng
    _free_cuda()

    leg_sep_wt_seeds: List[float] = []
    turn_sep_wt_seeds: List[float] = []
    wing_wt_seeds: List[float] = []
    delta_leg_mat = np.zeros((len(seeds), n_dn), dtype=np.float64)
    delta_turn_mat = np.zeros((len(seeds), n_dn), dtype=np.float64)
    delta_wing_mat = np.zeros((len(seeds), n_dn), dtype=np.float64)

    for s_i, sd in enumerate(seeds):
        eng = WholeCNSEngine(
            ds,
            num_brains=K,
            enable_callosum=False,
            homogeneous_tau_ms=20.0,
            common_random_numbers=True,
            seed=int(sd),
        )
        eng.set_perturbation_masks(silenced_masks, learned_init=False)

        # 1. Right vs Left PFL3 Steering Screen -> VNC leg MN asymmetry
        eng.set_seed(int(sd) * 10 + 1)
        eng.reset_state()
        res_R = eng.step_population(
            stim_ms,
            shared_stimuli=[(ds.idx_walk_tonic, 80.0), (ds.idx_pfl3_turnR, 250.0)],
            bg_poisson_hz=bg_noise_hz,
            bg_subset_indices=bg_sub,
        )
        eng.set_seed(int(sd) * 10 + 2)
        eng.reset_state()
        res_L = eng.step_population(
            stim_ms,
            shared_stimuli=[(ds.idx_walk_tonic, 80.0), (ds.idx_pfl3_turnL, 240.0)],
            bg_poisson_hz=bg_noise_hz,
            bg_subset_indices=bg_sub,
        )
        leg_sep_all = (res_R["leg_R_hz"] - res_R["leg_L_hz"]) - (res_L["leg_R_hz"] - res_L["leg_L_hz"])
        turn_sep_all = res_R["turn_hz_per_brain"] - res_L["turn_hz_per_brain"]

        leg_wt = float(leg_sep_all[0])
        turn_wt = float(turn_sep_all[0])
        leg_sep_wt_seeds.append(leg_wt)
        turn_sep_wt_seeds.append(turn_wt)
        delta_leg_mat[s_i] = leg_wt - leg_sep_all[1:]
        delta_turn_mat[s_i] = turn_wt - turn_sep_all[1:]

        # 2. Social Courtship Song Screen -> VNC Wing Motor Neurons (495 MNs)
        eng.set_seed(int(sd) * 10 + 3)
        eng.reset_state()
        res_soc = eng.step_population(
            stim_ms,
            shared_stimuli=[(lc10a_R, 220.0), (va1v, 220.0), (pC1_P1_aIPg, 220.0)],
            bg_poisson_hz=bg_noise_hz,
            bg_subset_indices=bg_sub,
            readout_pools={"wing_mn": cat.wing_mn_indices},
        )
        wing_all = res_soc["pool_hz"]["wing_mn"]
        wing_wt = float(wing_all[0])
        wing_wt_seeds.append(wing_wt)
        delta_wing_mat[s_i] = wing_wt - wing_all[1:]
        del eng
        _free_cuda()

    mean_d_leg = delta_leg_mat.mean(axis=0)
    sd_d_leg = delta_leg_mat.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros(n_dn)
    mean_d_turn = delta_turn_mat.mean(axis=0)
    sd_d_turn = delta_turn_mat.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros(n_dn)
    mean_d_wing = delta_wing_mat.mean(axis=0)
    sd_d_wing = delta_wing_mat.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros(n_dn)

    rows = []
    for i, tp in enumerate(cat.dn_types):
        rows.append({
            "dn_type": tp,
            "n_neurons": cat.dn_neuron_counts[i],
            "vnc_out_syn": round(cat.dn_vnc_out_syn[i], 1),
            "is_fru_dsx_dimorphic": cat.dn_is_sex_switch[i],
            "delta_vnc_leg_sep_hz": round(float(mean_d_leg[i]), 4),
            "delta_vnc_leg_sep_sd": round(float(sd_d_leg[i]), 4),
            "delta_turn_readout_hz": round(float(mean_d_turn[i]), 4),
            "delta_turn_readout_sd": round(float(sd_d_turn[i]), 4),
            "delta_vnc_wing_mn_hz": round(float(mean_d_wing[i]), 4),
            "delta_vnc_wing_mn_sd": round(float(sd_d_wing[i]), 4),
        })

    # Correlation between VNC out-synapse count and causal impact (hub confound audit)
    vnc_syn_arr = np.asarray(cat.dn_vnc_out_syn, dtype=np.float64)
    r_leg_syn = float(np.corrcoef(vnc_syn_arr, mean_d_leg)[0, 1]) if np.std(mean_d_leg) > 1e-9 else 0.0
    r_wing_syn = float(np.corrcoef(vnc_syn_arr, mean_d_wing)[0, 1]) if np.std(mean_d_wing) > 1e-9 else 0.0

    return {
        "n_dn_types_screened": n_dn,
        "parallel_k_slots": K,
        "crn_unperturbed_max_diff_hz": crn_max_diff,
        "wt_vnc_leg_sep_hz_mean": round(float(np.mean(leg_sep_wt_seeds)), 4),
        "wt_turn_sep_hz_mean": round(float(np.mean(turn_sep_wt_seeds)), 4),
        "wt_vnc_wing_mn_hz_mean": round(float(np.mean(wing_wt_seeds)), 4),
        "pearson_r_vnc_syn_vs_leg_impact": round(r_leg_syn, 4),
        "pearson_r_vnc_syn_vs_wing_impact": round(r_wing_syn, 4),
        "dn_table": rows,
    }


# ---------------------------------------------------------------------------
# Pillar 3: Cross-Dataset Threshold Audit & Within-Connectome fru/dsx Switch
# ---------------------------------------------------------------------------
def compute_synapse_threshold_audit(ds_male: ConnectomeDataset, ds_female: ConnectomeDataset) -> Dict[str, object]:
    """
    Quantify directed-edge and synapse survival across identical minimum-synapse
    cutoffs (|w| >= 1, 2, 3, 5, 10, 20) in MaleCNS v1.0 vs Female BANC v888, plus
    open-loop steering and brain-activity gate probes on MaleCNS thresholded at >= 3.
    """
    thresholds = (1, 2, 3, 5, 10, 20)
    Wm = ds_male.W_base_csr
    Wf = ds_female.W_base_csr
    abs_m = np.abs(Wm.data)
    abs_f = np.abs(Wf.data)

    curve = []
    for th in thresholds:
        mm = abs_m >= th
        mf = abs_f >= th
        em = int(mm.sum())
        ef = int(mf.sum())
        sm = float(abs_m[mm].sum())
        sf = float(abs_f[mf].sum())
        curve.append({
            "min_synapses": th,
            "male_edges": em,
            "female_edges": ef,
            "male_to_female_edge_ratio": round(em / max(1, ef), 3),
            "male_abs_weight_sum": round(sm, 1),
            "female_abs_weight_sum": round(sf, 1),
            "male_mean_out_deg": round(em / float(ds_male.N), 2),
            "female_mean_out_deg": round(ef / float(ds_female.N), 2),
        })

    # Test MaleCNS pruned to |w| >= 3 (matching BANC's nominal compilation threshold)
    snap_m = CSRSnapshot.capture(ds_male)
    try:
        def _probe_gate_and_steer(ds_obj: ConnectomeDataset) -> Dict[str, float]:
            vals_sep, vals_brain, vals_leg = [], [], []
            for sd in (500, 501, 502, 503, 504):
                eng = WholeCNSEngine(ds_obj, num_brains=2, enable_callosum=False, homogeneous_tau_ms=20.0, seed=sd)
                r_w = np.array([80.0, 80.0], dtype=np.float32)
                r_R = np.array([250.0, 0.0], dtype=np.float32)
                r_L = np.array([0.0, 240.0], dtype=np.float32)
                res = eng.step_population(
                    60,
                    shared_stimuli=[(ds_obj.idx_walk_tonic, r_w), (ds_obj.idx_pfl3_turnR, r_R), (ds_obj.idx_pfl3_turnL, r_L)],
                    bg_poisson_hz=22.0,
                    bg_subset_indices=ds_obj.plastic_neurons[::6],
                )
                vals_sep.append(float(res["turn_hz_per_brain"][0] - res["turn_hz_per_brain"][1]))
                vals_brain.append(float(res["mean_brain_hz"].mean()))
                vals_leg.append(float(0.5 * (res["leg_L_hz"].mean() + res["leg_R_hz"].mean())))
                del eng
            _free_cuda()
            return {
                "goal_sep_hz_mean": round(float(np.mean(vals_sep)), 3),
                "goal_sep_hz_sd": round(float(np.std(vals_sep, ddof=1)), 3),
                "mean_brain_hz": round(float(np.mean(vals_brain)), 4),
                "mean_leg_mn_hz": round(float(np.mean(vals_leg)), 4),
                "exceeds_3d_brain_gate_0p02hz": bool(np.mean(vals_brain) >= 0.02),
            }

        male_ge1 = _probe_gate_and_steer(ds_male)
        ds_male.W_base_csr.data[np.abs(ds_male.W_base_csr.data) < 3.0] = 0.0
        sync_gpu_csr(ds_male)
        male_ge3 = _probe_gate_and_steer(ds_male)
        female_ge3 = _probe_gate_and_steer(ds_female)
    finally:
        snap_m.restore(ds_male)

    return {
        "threshold_curve": curve,
        "functional_gate_comparison": {
            "male_ge1_native": male_ge1,
            "male_ge3_threshold_matched": male_ge3,
            "female_banc_ge3_native": female_ge3,
        },
    }


def evaluate_sex_switch_double_dissociation_male(
    ds_m: ConnectomeDataset,
    engine_seeds: Sequence[int] = (600, 601, 602, 603, 604),
    stim_ms: int = 80,
    bg_noise_hz: float = 20.0,
) -> Dict[str, float]:
    """
    Measure Navigation (PFL3 -> DNa02 / turn readout) vs Courtship Song
    (pC1/P1/aIPg + LC10a_R + VA1v -> pIP10 & VNC Wing MNs, plus unaroused sensory gating)
    on MaleCNS v1.0.
    """
    df = ds_m.meta_df
    lc10a_R = df[(df["type"] == "LC10a") & (df["somaSide"] == "R")].index.to_numpy(dtype=np.int32)
    va1v = df[df["type"].fillna("").str.contains("VA1v", regex=True)].index.to_numpy(dtype=np.int32)
    pC1_P1_aIPg = df[df["type"].fillna("").str.contains("^pC1|^P1_|^aIPg", regex=True)].index.to_numpy(dtype=np.int32)
    pip10 = df[df["type"] == "pIP10"].index.to_numpy(dtype=np.int32)
    dna02_L = df[(df["type"] == "DNa02") & (df["somaSide"] == "L")].index.to_numpy(dtype=np.int32)
    dna02_R = df[(df["type"] == "DNa02") & (df["somaSide"] == "R")].index.to_numpy(dtype=np.int32)
    wing_mn = df[
        df["exitNerve"].fillna("").str.contains("ADMN|PDMN|wing", case=False, regex=True)
        | df["subclass"].fillna("").str.contains("wing", case=False, regex=True)
    ].index.to_numpy(dtype=np.int32)
    bg_sub = ds_m.plastic_neurons[::8]

    K = 4 * len(engine_seeds)
    eng = WholeCNSEngine(
        ds_m, num_brains=K, enable_callosum=False, homogeneous_tau_ms=20.0, common_random_numbers=False, seed=int(engine_seeds[0])
    )
    # Within each seed block b=0..S-1:
    #   4*b + 0 = Nav Right (PFL3_R + walk)
    #   4*b + 1 = Nav Left  (PFL3_L + walk)
    #   4*b + 2 = Unaroused Social Sensory (LC10a_R + VA1v only)
    #   4*b + 3 = Aroused Courtship Command (pC1/P1/aIPg + LC10a_R + VA1v)
    r_walk = np.zeros(K, dtype=np.float32)
    r_pR = np.zeros(K, dtype=np.float32)
    r_pL = np.zeros(K, dtype=np.float32)
    r_vis = np.zeros(K, dtype=np.float32)
    r_olf = np.zeros(K, dtype=np.float32)
    r_aro = np.zeros(K, dtype=np.float32)
    for b in range(len(engine_seeds)):
        r_walk[4 * b : 4 * b + 2] = 80.0
        r_pR[4 * b + 0] = 250.0
        r_pL[4 * b + 1] = 240.0
        r_vis[4 * b + 2] = 260.0
        r_olf[4 * b + 2] = 260.0
        r_vis[4 * b + 3] = 220.0
        r_olf[4 * b + 3] = 220.0
        r_aro[4 * b + 3] = 220.0

    res = eng.step_population(
        stim_ms,
        shared_stimuli=[
            (ds_m.idx_walk_tonic, r_walk),
            (ds_m.idx_pfl3_turnR, r_pR),
            (ds_m.idx_pfl3_turnL, r_pL),
            (lc10a_R, r_vis),
            (va1v, r_olf),
            (pC1_P1_aIPg, r_aro),
        ],
        bg_poisson_hz=bg_noise_hz,
        bg_subset_indices=bg_sub,
        readout_pools={"pip10": pip10, "wing_mn": wing_mn, "dna02_L": dna02_L, "dna02_R": dna02_R},
    )

    s0 = slice(0, K, 4)
    s1 = slice(1, K, 4)
    s2 = slice(2, K, 4)
    s3 = slice(3, K, 4)

    nav_sep = res["turn_hz_per_brain"][s0] - res["turn_hz_per_brain"][s1]
    dna02_sep = (res["pool_hz"]["dna02_R"][s0] - res["pool_hz"]["dna02_L"][s0]) - (
        res["pool_hz"]["dna02_R"][s1] - res["pool_hz"]["dna02_L"][s1]
    )
    pip10_unaroused = res["pool_hz"]["pip10"][s2]
    pip10_aroused = res["pool_hz"]["pip10"][s3]
    wing_aroused = res["pool_hz"]["wing_mn"][s3]
    del eng
    _free_cuda()

    return {
        "nav_turn_sep_hz": round(float(nav_sep.mean()), 3),
        "nav_dna02_sep_hz": round(float(dna02_sep.mean()), 3),
        "courtship_pip10_aroused_hz": round(float(pip10_aroused.mean()), 3),
        "courtship_pip10_unaroused_leak_hz": round(float(pip10_unaroused.mean()), 3),
        "courtship_selective_gain_hz": round(float((pip10_aroused - pip10_unaroused).mean()), 3),
        "courtship_wing_mn_hz": round(float(wing_aroused.mean()), 4),
    }


def evaluate_sex_switch_double_dissociation_female(
    ds_f: ConnectomeDataset,
    engine_seeds: Sequence[int] = (600, 601, 602, 603, 604),
    stim_ms: int = 80,
    bg_noise_hz: float = 20.0,
) -> Dict[str, float]:
    """
    Measure Female Navigation (PFL3 -> steering DNs) vs Female Courtship Receptivity
    (pC1a-e + auditory JO-B -> vpoDN vaginal-plate-opening DN) on Female BANC v888.
    """
    
    fp = extract_female_social_pools(ds_f)
    bg_sub = ds_f.plastic_neurons[::6]
    K = 3 * len(engine_seeds)
    eng = WholeCNSEngine(
        ds_f, num_brains=K, enable_callosum=False, homogeneous_tau_ms=20.0, common_random_numbers=False, seed=int(engine_seeds[0])
    )
    r_walk = np.zeros(K, dtype=np.float32)
    r_pR = np.zeros(K, dtype=np.float32)
    r_pL = np.zeros(K, dtype=np.float32)
    r_soc = np.zeros(K, dtype=np.float32)
    for b in range(len(engine_seeds)):
        r_walk[3 * b : 3 * b + 2] = 80.0
        r_pR[3 * b + 0] = 250.0
        r_pL[3 * b + 1] = 240.0
        r_soc[3 * b + 2] = 260.0

    res = eng.step_population(
        stim_ms,
        shared_stimuli=[
            (ds_f.idx_walk_tonic, r_walk),
            (ds_f.idx_pfl3_turnR, r_pR),
            (ds_f.idx_pfl3_turnL, r_pL),
            (fp["pc1_abc"], r_soc),
            (fp["jo_b"], r_soc),
        ],
        bg_poisson_hz=bg_noise_hz,
        bg_subset_indices=bg_sub,
        readout_pools={"vpodn": fp["vpodn"], "dnp13": fp["dnp13"]},
    )

    s0 = slice(0, K, 3)
    s1 = slice(1, K, 3)
    s2 = slice(2, K, 3)
    nav_sep = res["turn_hz_per_brain"][s0] - res["turn_hz_per_brain"][s1]
    vpodn_hz = res["pool_hz"]["vpodn"][s2]
    del eng
    _free_cuda()

    return {
        "nav_turn_sep_hz": round(float(nav_sep.mean()), 3),
        "receptivity_vpodn_hz": round(float(vpodn_hz.mean()), 3),
    }
