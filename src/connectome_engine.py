"""
Whole-CNS Spiking LIF & Common-Random-Numbers Perturbation Engine (MaleCNS v1.0 & Female BANC v888)
===================================================================================================
Simulates K = 1 .. K_max adult male (MaleCNS v1.0; 167,565 neurons, 25.6M directed connections) and
adult female (BANC v888; 157,789 neurons, 3.0M directed connections) whole-CNS connectomes on CPU or GPU:
  - Shared read-only CSR synaptic matrix across K independent state slots
  - Event-driven active-row CUDA Triton gather-scatter kernel (with PyTorch CSR/COO fallback)
  - Common-Random-Numbers (CRN) broadcast noise and per-slot acute silencing masks for K = 486 screens
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from scipy import sparse

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False


if _HAS_TRITON:
    @triton.jit
    def _active_csr_scatter_kernel(
        ActiveU_ptr,
        WAct_ptr,
        Indptr_ptr,
        Indices_ptr,
        Data_ptr,
        OutNK_ptr,
        K,
        BLOCK_K: tl.constexpr,
    ):
        """
        Coalesced event-driven CSR row gather-scatter across K whole connectomes.
        Only active presynaptic rows u (typically <3% of N per 1 ms bin) are visited,
        and K-blocks with zero spikes across their 128 brains skip edge traversal entirely.
        """
        pid_m = tl.program_id(0).to(tl.int64)
        pid_kb = tl.program_id(1).to(tl.int64)
        ks = pid_kb * BLOCK_K + tl.arange(0, BLOCK_K).to(tl.int64)
        k_mask = ks < K

        pre_v = tl.load(WAct_ptr + pid_m * K + ks, mask=k_mask, other=0.0)
        if tl.max(tl.abs(pre_v), axis=0) > 0.0:
            u = tl.load(ActiveU_ptr + pid_m)
            s0 = tl.load(Indptr_ptr + u)
            s1 = tl.load(Indptr_ptr + u + 1)
            for e in range(s0, s1):
                v = tl.load(Indices_ptr + e).to(tl.int64)
                w = tl.load(Data_ptr + e)
                tl.atomic_add(OutNK_ptr + v * K + ks, pre_v * w, mask=k_mask)


CASTE_DEFINITIONS = [
    {"caste_id": 0, "code": "C0", "name": "Ultra-Fast Reflex",     "tau_range": (8.5, 11.5),  "lr_mult": 1.35, "gaze_deg":  0.0,  "horizon_mm":  52.0, "vote_w": 1.15},
    {"caste_id": 1, "code": "C1", "name": "Optic Detour LC16",     "tau_range": (16.0, 21.0), "lr_mult": 1.15, "gaze_deg":  0.0,  "horizon_mm": 100.0, "vote_w": 1.15},
    {"caste_id": 2, "code": "C2", "name": "Long-Horizon Scout",    "tau_range": (22.0, 27.5), "lr_mult": 1.00, "gaze_deg":  0.0,  "horizon_mm": 210.0, "vote_w": 0.95},
    {"caste_id": 3, "code": "C3", "name": "1-Shot KCg Flash LTP",  "tau_range": (14.5, 18.5), "lr_mult": 3.50, "gaze_deg":  0.0,  "horizon_mm": 115.0, "vote_w": 1.40},
    {"caste_id": 4, "code": "C4", "name": "Consolidated KCab LTM", "tau_range": (21.0, 26.0), "lr_mult": 0.80, "gaze_deg":  0.0,  "horizon_mm": 120.0, "vote_w": 1.20},
    {"caste_id": 5, "code": "C5", "name": "PAM Appetitive Corridor","tau_range": (18.0, 22.0),"lr_mult": 1.15, "gaze_deg":  0.0,  "horizon_mm": 115.0, "vote_w": 1.00},
    {"caste_id": 6, "code": "C6", "name": "Left Fovea Predictor",  "tau_range": (13.5, 17.5), "lr_mult": 1.25, "gaze_deg": -24.0, "horizon_mm": 145.0, "vote_w": 1.05},
    {"caste_id": 7, "code": "C7", "name": "Right Fovea Predictor", "tau_range": (13.5, 17.5), "lr_mult": 1.25, "gaze_deg": +24.0, "horizon_mm": 145.0, "vote_w": 1.05},
    {"caste_id": 8, "code": "C8", "name": "EPG Place-Cell Grid",   "tau_range": (22.0, 28.0), "lr_mult": 1.20, "gaze_deg":  0.0,  "horizon_mm": 135.0, "vote_w": 1.30},
    {"caste_id": 9, "code": "C9", "name": "Callosum Executive",    "tau_range": (16.5, 20.5), "lr_mult": 1.30, "gaze_deg":  0.0,  "horizon_mm": 115.0, "vote_w": 1.65},
]


def resolve_compiled_connectome_dir(explicit_dir: Optional[str] = None) -> Path:
    candidates = []
    if explicit_dir:
        candidates.append(Path(explicit_dir))
    if os.environ.get("MALECNS_COMPILED_DIR"):
        candidates.append(Path(os.environ["MALECNS_COMPILED_DIR"]))
    repo_root = Path(__file__).resolve().parent.parent
    candidates.extend([
        repo_root / "data" / "compiled",
        Path("/content/drive/MyDrive/Drosophila_Whole_CNS_Specificity/compiled"),
        Path(r"D:\male-fly-brain\data\compiled"),
        Path("/content/drosophila_cns_data/compiled"),
        Path("/content/malecns_data/compiled"),
    ])
    for c in candidates:
        if (c / "connectome_compiled.npz").exists() and (c / "neurons_meta.parquet").exists():
            return c
    raise FileNotFoundError(
        "Compiled MaleCNS v1.0 files (connectome_compiled.npz, neurons_meta.parquet) not found. "
        "Set MALECNS_COMPILED_DIR or place them in data/compiled/."
    )


def resolve_female_connectome_dir(explicit_dir: Optional[str] = None) -> Path:
    candidates = []
    if explicit_dir:
        candidates.append(Path(explicit_dir))
    if os.environ.get("FEMALE_COMPILED_DIR"):
        candidates.append(Path(os.environ["FEMALE_COMPILED_DIR"]))
    repo_root = Path(__file__).resolve().parent.parent
    candidates.extend([
        repo_root / "data" / "female_compiled",
        Path("/content/drive/MyDrive/Drosophila_Whole_CNS_Specificity/female_compiled"),
        Path("/content/drosophila_cns_data/female_compiled"),
        Path("/content/malecns_data/female_compiled"),
    ])
    for c in candidates:
        if (c / "connectome_compiled.npz").exists() and (c / "neurons_meta.parquet").exists():
            return c
    raise FileNotFoundError(
        "Compiled Female BANC v888 files (connectome_compiled.npz, neurons_meta.parquet) not found. "
        "Run `python scripts/compile_female_banc.py` or set FEMALE_COMPILED_DIR."
    )


class ConnectomeDataset:
    """Loads the 167,565-neuron MaleCNS v1.0 connectome once and extracts anatomical cell-type masks."""

    def __init__(self, compiled_dir: Optional[str] = None):
        self.compiled_dir = resolve_compiled_connectome_dir(compiled_dir)
        t0 = time.perf_counter()
        with np.load(self.compiled_dir / "connectome_compiled.npz") as npz:
            out_weights = npz["out_weights"].astype(np.float32)
            out_indices = npz["out_indices"].astype(np.int32)
            out_indptr = npz["out_indptr"].astype(np.int32)
            self.nt_sign = npz["nt_sign"].astype(np.float32)
            self.region_id = npz["region_id"].astype(np.uint8)
            self.nt_id = npz["nt_id"].astype(np.uint8)
            self.dimorphism_id = npz["dimorphism_id"].astype(np.uint8)
            self.is_motor = npz["is_motor"].astype(bool)

        self.meta_df = pq.read_table(self.compiled_dir / "neurons_meta.parquet").to_pandas()
        stats_path = self.compiled_dir / "stats_summary.json"
        self.stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}

        self.N = int(len(self.meta_df))
        capped = np.minimum(out_weights, 400.0)
        row_signs = np.repeat(self.nt_sign, np.diff(out_indptr))
        signed_weights = (capped * row_signs).astype(np.float32)

        # Shared outgoing CSR: row u -> postsynaptic targets v
        self.W_base_csr = sparse.csr_matrix(
            (signed_weights, out_indices, out_indptr),
            shape=(self.N, self.N),
            dtype=np.float32,
        )
        self.nnz = int(self.W_base_csr.nnz)
        self.proofread_synapses = int(self.stats.get("proofread_synapses", int(out_weights.sum())))

        # Extract canonical Drosophila sensory, associative, modulatory, dimorphic, and motor populations
        t = self.meta_df["type"].fillna("").astype(str)
        cls = self.meta_df["class"].fillna("").astype(str)
        side = self.meta_df["somaSide"].fillna("").astype(str)
        nt = self.meta_df["nt_name"].fillna("").astype(str)

        self.idx_lc16_L = self.meta_df.loc[t.isin(["LC16", "LPLC1"]) & (side == "L"), "idx"].to_numpy(dtype=np.int32)
        self.idx_lc16_R = self.meta_df.loc[t.isin(["LC16", "LPLC1"]) & (side == "R"), "idx"].to_numpy(dtype=np.int32)
        self.idx_loom_L = self.meta_df.loc[t.isin(["LC4", "LPLC2"]) & (side == "L"), "idx"].to_numpy(dtype=np.int32)
        self.idx_loom_R = self.meta_df.loc[t.isin(["LC4", "LPLC2"]) & (side == "R"), "idx"].to_numpy(dtype=np.int32)

        self.idx_kc_vis_L = self.meta_df.loc[(t == "KCg-d") & (side == "L"), "idx"].to_numpy(dtype=np.int32)
        self.idx_kc_vis_R = self.meta_df.loc[(t == "KCg-d") & (side == "R"), "idx"].to_numpy(dtype=np.int32)

        # Central Complex PFL3 compass steering populations (contralateral LAL -> DNa02 projection)
        self.idx_pfl3_turnR = self.meta_df.loc[t.str.startswith("PFL3") & (side == "L"), "idx"].to_numpy(dtype=np.int32)
        self.idx_pfl3_turnL = self.meta_df.loc[t.str.startswith("PFL3") & (side == "R"), "idx"].to_numpy(dtype=np.int32)

        # Tonic forward walking Descending Neurons (bilateral DNp09 L+R for zero unilateral turn drift)
        self.idx_walk_tonic = self.meta_df.loc[t.str.startswith("DNp09"), "idx"].to_numpy(dtype=np.int32)

        # Steering & Backward Escape Descending Neurons + VNC Leg Motor Pools
        dn_steer_types = ("DNa01", "DNa02", "DNa03", "DNa04", "DNb01", "DNb02", "DNg13")
        mask_dn_steer = t.str.startswith(dn_steer_types)
        self.mask_dn_L = (mask_dn_steer & (side == "L")).to_numpy(dtype=bool)
        self.mask_dn_R = (mask_dn_steer & (side == "R")).to_numpy(dtype=bool)
        self.idx_dn_L = np.flatnonzero(self.mask_dn_L).astype(np.int32)
        self.idx_dn_R = np.flatnonzero(self.mask_dn_R).astype(np.int32)
        self.n_dn_L = max(1, int(self.mask_dn_L.sum()))
        self.n_dn_R = max(1, int(self.mask_dn_R.sum()))

        self.mask_leg_L = (self.is_motor & (side == "L").to_numpy()).astype(bool)
        self.mask_leg_R = (self.is_motor & (side == "R").to_numpy()).astype(bool)
        self.idx_leg_L = np.flatnonzero(self.mask_leg_L).astype(np.int32)
        self.idx_leg_R = np.flatnonzero(self.mask_leg_R).astype(np.int32)
        self.n_leg_L = max(1, int(self.mask_leg_L.sum()))
        self.n_leg_R = max(1, int(self.mask_leg_R.sum()))
        self.is_brain_only = bool(int(self.is_motor.sum()) < 200)
        self.motor_turn_scale = 0.16 if self.is_brain_only else 1.0

        self.idx_mdn = self.meta_df.loc[t.str.startswith("MDN"), "idx"].to_numpy(dtype=np.int32)
        self.idx_ppl1_dan = self.meta_df.loc[t.str.startswith("PPL1"), "idx"].to_numpy(dtype=np.int32)
        self.idx_pam_dan = self.meta_df.loc[t.str.startswith("PAM"), "idx"].to_numpy(dtype=np.int32)
        self.idx_apl = self.meta_df.loc[t == "APL", "idx"].to_numpy(dtype=np.int32)

        # Male-specific social/arousal command clusters (P1 / pC1 / TN1)
        self.idx_male_p1 = self.meta_df.loc[t.str.startswith(("P1_", "pC1")), "idx"].to_numpy(dtype=np.int32)
        self.mask_male_specific = (self.dimorphism_id == 2)
        self.mask_sex_dimorphic = (self.dimorphism_id == 1)
        self.mask_isomorphic = (self.dimorphism_id == 0)

        # 5-Tier Plasticity Masks (STM KCg + Optic, ITM KCa'b', LTM KCab, MBON Valence, Central Complex)
        is_vis_avoid = t.isin(["LC16", "LPLC1", "LC4", "KCg-d"]).to_numpy(dtype=bool)
        is_kcg = (t.isin(["LPLC2", "LC10a"]) | (t.str.startswith("KCg") & (t != "KCg-d"))).to_numpy(dtype=bool)
        is_stm = is_vis_avoid | is_kcg
        is_itm = t.str.startswith("KCa'b'").to_numpy(dtype=bool)
        is_ltm = (t.str.startswith("KCab") | (t == "KC")).to_numpy(dtype=bool)
        is_mbon = (cls == "MBON").to_numpy(dtype=bool)
        self.mask_mbon_app = is_mbon & (nt == "acetylcholine").to_numpy(dtype=bool)
        self.mask_mbon_avd = is_mbon & (~self.mask_mbon_app)
        is_cx = (
            t.str.startswith(("EPG", "PEN", "PEG", "Delta7", "PFL", "FC", "hDelta", "vDelta")).to_numpy(dtype=bool)
            & ~(is_stm | is_itm | is_ltm | is_mbon)
        )

        self.vis_avoid_neurons = np.flatnonzero(is_vis_avoid).astype(np.int32)
        self.kcg_neurons = np.flatnonzero(is_kcg).astype(np.int32)
        self.stm_neurons = np.flatnonzero(is_stm).astype(np.int32)
        self.itm_neurons = np.flatnonzero(is_itm).astype(np.int32)
        self.ltm_neurons = np.flatnonzero(is_ltm).astype(np.int32)
        self.mbon_app_neurons = np.flatnonzero(self.mask_mbon_app).astype(np.int32)
        self.mbon_avd_neurons = np.flatnonzero(self.mask_mbon_avd).astype(np.int32)
        self.cx_neurons = np.flatnonzero(is_cx).astype(np.int32)

        side_np = (side == "L").to_numpy()
        side_R_np = (side == "R").to_numpy()
        self.mask_mbon_app_L = (self.mask_mbon_app & side_np).astype(bool)
        self.mask_mbon_app_R = (self.mask_mbon_app & side_R_np).astype(bool)
        self.idx_mbon_app_L = np.flatnonzero(self.mask_mbon_app_L).astype(np.int32)
        self.idx_mbon_app_R = np.flatnonzero(self.mask_mbon_app_R).astype(np.int32)
        self.n_mbon_app_L = max(1, int(self.mask_mbon_app_L.sum()))
        self.n_mbon_app_R = max(1, int(self.mask_mbon_app_R.sum()))

        # Hemisphere-aligned pure Kenyon Cell pools (4 disjoint sparse coincidence pools per hemisphere;
        # excludes LPLC2/LC10a visual lobula neurons so calyx coincidence coding is 100% Mushroom Body KCs)
        pure_kcg_mask = (t.str.startswith("KCg") & (t != "KCg-d")).to_numpy(dtype=bool)
        pure_kcg_neurons = np.flatnonzero(pure_kcg_mask).astype(np.int32)
        kcg_L = pure_kcg_neurons[side_np[pure_kcg_neurons]]
        kcg_R = pure_kcg_neurons[side_R_np[pure_kcg_neurons]]
        itm_L = self.itm_neurons[side_np[self.itm_neurons]]
        itm_R = self.itm_neurons[side_R_np[self.itm_neurons]]
        ltm_L = self.ltm_neurons[side_np[self.ltm_neurons]]
        ltm_R = self.ltm_neurons[side_R_np[self.ltm_neurons]]

        # Hemisphere-specific APL homeostatic inhibition balance (Lin et al., Nat. Neurosci. 2014)
        self.idx_apl_L = self.idx_apl[side_np[self.idx_apl]]
        self.idx_apl_R = self.idx_apl[side_R_np[self.idx_apl]]
        if len(self.idx_apl_L) == 0:
            self.idx_apl_L = self.idx_apl
        if len(self.idx_apl_R) == 0:
            self.idx_apl_R = self.idx_apl
        apl_drive_L = float(np.abs(self.W_base_csr[self.idx_apl_L, :][:, kcg_L[:880]].sum()) / max(1, len(kcg_L[:880])))
        apl_drive_R = float(np.abs(self.W_base_csr[self.idx_apl_R, :][:, kcg_R[:880]].sum()) / max(1, len(kcg_R[:880])))
        apl_drive_mean = 0.5 * (apl_drive_L + apl_drive_R)
        self.apl_scale_L = float(np.clip(apl_drive_mean / max(1e-6, apl_drive_L), 0.65, 1.45))
        self.apl_scale_R = float(np.clip(apl_drive_mean / max(1e-6, apl_drive_R), 0.65, 1.45))

        n_sub = min(len(kcg_L), len(kcg_R), len(ltm_L), len(ltm_R)) // 4
        n_itm_sub = min(len(itm_L), len(itm_R)) // 4
        n_pair = 4 * n_sub
        self.n_kc_sub = int(n_sub)
        self.kcg_paired = np.concatenate([kcg_L[:n_pair], kcg_R[:n_pair]]).astype(np.int32)
        self.ltm_paired = np.concatenate([ltm_L[:n_pair], ltm_R[:n_pair]]).astype(np.int32)

        self.kc_pools_L: List[np.ndarray] = []
        self.kc_pools_R: List[np.ndarray] = []
        self.kc_homeo_scale = np.ones(self.N, dtype=np.float32)
        W_app_L = np.asarray(self.W_base_csr[:, self.mask_mbon_app_L].sum(axis=1)).ravel() / float(self.n_mbon_app_L)
        W_app_R = np.asarray(self.W_base_csr[:, self.mask_mbon_app_R].sum(axis=1)).ravel() / float(self.n_mbon_app_R)
        W_apl = np.abs(np.asarray(self.W_base_csr[self.idx_apl, :].sum(axis=0)).ravel())
        apl_att = 1.0 / (1.0 + 0.0022 * W_apl)
        for c_idx in range(4):
            sL, eL = c_idx * n_sub, (c_idx + 1) * n_sub
            siL, eiL = c_idx * n_itm_sub, (c_idx + 1) * n_itm_sub
            self.kc_pools_L.append(np.concatenate([kcg_L[sL:eL], itm_L[siL:eiL], ltm_L[sL:eL]]).astype(np.int32))
            self.kc_pools_R.append(np.concatenate([kcg_R[sL:eL], itm_R[siL:eiL], ltm_R[sL:eL]]).astype(np.int32))
            for subL, subR in [
                (kcg_L[sL:eL], kcg_R[sL:eL]),
                (itm_L[siL:eiL], itm_R[siL:eiL]),
                (ltm_L[sL:eL], ltm_R[sL:eL]),
            ]:
                dL = float((np.maximum(0.0, W_app_L[subL] - W_app_R[subL]) * apl_att[subL]).sum())
                dR = float((np.maximum(0.0, W_app_R[subR] - W_app_L[subR]) * apl_att[subR]).sum())
                if dL > 1e-6 and dR > 1e-6:
                    d_mean = 0.5 * (dL + dR)
                    self.kc_homeo_scale[subL] = np.clip(d_mean / dL, 0.45, 1.75)
                    self.kc_homeo_scale[subR] = np.clip(d_mean / dR, 0.45, 1.75)

        all_kc_L = np.concatenate(self.kc_pools_L)
        all_kc_R = np.concatenate(self.kc_pools_R)
        raw_kc_mbon = 0.5 * (float(W_app_L[all_kc_L].sum()) + float(W_app_R[all_kc_R].sum()))
        if 10.0 < raw_kc_mbon < 500.0:
            # Normalize Princeton BANC v888 cleft-detection density (85.2) to Janelia EM cleft scale (~1,950)
            cleft_comp = float(np.clip(1950.0 / raw_kc_mbon, 1.0, 23.0))
            self.kc_homeo_scale[all_kc_L] *= cleft_comp
            self.kc_homeo_scale[all_kc_R] *= cleft_comp
            if not self.is_brain_only:
                self.motor_turn_scale = 0.72

        self.homeo_kcg = self.kc_homeo_scale[self.kcg_neurons]
        self.homeo_itm = self.kc_homeo_scale[self.itm_neurons]
        self.homeo_ltm = self.kc_homeo_scale[self.ltm_neurons]
        self.homeo_kcg_paired = self.kc_homeo_scale[self.kcg_paired]
        self.homeo_ltm_paired = self.kc_homeo_scale[self.ltm_paired]
        self.pool_ltm_mask = np.concatenate([
            np.zeros(n_sub + n_itm_sub, dtype=np.float32),
            np.ones(n_sub, dtype=np.float32),
        ])

        all_plastic = is_stm | is_itm | is_ltm | is_mbon | is_cx
        self.plastic_neurons = np.flatnonzero(all_plastic).astype(np.int32)
        self.plastic_neuron_count = int(len(self.plastic_neurons))

        # Count exact plastic outgoing edges per brain
        indptr = self.W_base_csr.indptr
        self.plastic_synapses_per_brain = int(sum(int(indptr[u + 1] - indptr[u]) for u in self.plastic_neurons))

        self.plastic_slot = np.full(self.N, -1, dtype=np.int32)
        self.plastic_slot[self.plastic_neurons] = np.arange(self.plastic_neuron_count, dtype=np.int32)
        self.slot_vis = self.plastic_slot[self.vis_avoid_neurons]
        self.slot_kcg = self.plastic_slot[self.kcg_neurons]
        self.slot_itm = self.plastic_slot[self.itm_neurons]
        self.slot_ltm = self.plastic_slot[self.ltm_neurons]
        self.slot_kcg_paired = self.plastic_slot[self.kcg_paired]
        self.slot_ltm_paired = self.plastic_slot[self.ltm_paired]

        self.W_T_gpu = None
        self._cuda_ready = False
        if torch.cuda.is_available():
            self.ensure_cuda_tensors(need_coo_fallback=not _HAS_TRITON)

        self.load_seconds = time.perf_counter() - t0

    def ensure_cuda_tensors(self, need_coo_fallback: bool = False):
        if not self._cuda_ready:
            # Compact 196.13 MB zero-copy outgoing CSR arrays on GPU
            self.csr_indptr_gpu = torch.from_numpy(self.W_base_csr.indptr.astype(np.int32)).to("cuda")
            self.csr_indices_gpu = torch.from_numpy(self.W_base_csr.indices.astype(np.int32)).to("cuda")
            self.csr_data_gpu = torch.from_numpy(self.W_base_csr.data.astype(np.float32)).to("cuda")

            self.mask_dnL_gpu = torch.from_numpy(self.mask_dn_L).to("cuda")
            self.mask_dnR_gpu = torch.from_numpy(self.mask_dn_R).to("cuda")
            self.mask_lgL_gpu = torch.from_numpy(self.mask_leg_L).to("cuda")
            self.mask_lgR_gpu = torch.from_numpy(self.mask_leg_R).to("cuda")
            self.mask_mbon_app_L_gpu = torch.from_numpy(self.mask_mbon_app_L).to("cuda")
            self.mask_mbon_app_R_gpu = torch.from_numpy(self.mask_mbon_app_R).to("cuda")
            self.idx_dnL_gpu = torch.from_numpy(self.idx_dn_L).long().to("cuda")
            self.idx_dnR_gpu = torch.from_numpy(self.idx_dn_R).long().to("cuda")
            self.idx_p1_gpu = torch.from_numpy(self.idx_male_p1).long().to("cuda")
            self.plastic_neurons_gpu = torch.from_numpy(self.plastic_neurons).long().to("cuda")

            # Plasticity & tier readout index tensors (zero PCIe transfers during closed-loop episodes)
            self.idx_vis_gpu = torch.from_numpy(self.vis_avoid_neurons).long().to("cuda")
            self.idx_kcg_gpu = torch.from_numpy(self.kcg_neurons).long().to("cuda")
            self.idx_itm_gpu = torch.from_numpy(self.itm_neurons).long().to("cuda")
            self.idx_ltm_gpu = torch.from_numpy(self.ltm_neurons).long().to("cuda")
            self.idx_kcg_paired_gpu = torch.from_numpy(self.kcg_paired).long().to("cuda")
            self.idx_ltm_paired_gpu = torch.from_numpy(self.ltm_paired).long().to("cuda")
            self.idx_mbon_app_gpu = torch.from_numpy(self.mbon_app_neurons).long().to("cuda")
            self.slot_vis_gpu = torch.from_numpy(self.slot_vis).long().to("cuda")
            self.slot_kcg_gpu = torch.from_numpy(self.slot_kcg).long().to("cuda")
            self.slot_itm_gpu = torch.from_numpy(self.slot_itm).long().to("cuda")
            self.slot_ltm_gpu = torch.from_numpy(self.slot_ltm).long().to("cuda")
            self.homeo_kcg_gpu = torch.from_numpy(self.homeo_kcg).to("cuda")[None, :]
            self.homeo_itm_gpu = torch.from_numpy(self.homeo_itm).to("cuda")[None, :]
            self.homeo_ltm_gpu = torch.from_numpy(self.homeo_ltm).to("cuda")[None, :]
            self.homeo_kcg_paired_gpu = torch.from_numpy(self.homeo_kcg_paired).to("cuda")[None, :]
            self.homeo_ltm_paired_gpu = torch.from_numpy(self.homeo_ltm_paired).to("cuda")[None, :]
            self.pool_ltm_mask_gpu = torch.from_numpy(self.pool_ltm_mask).to("cuda")[None, :]
            self.kc_pools_L_gpu = [torch.from_numpy(p).long().to("cuda") for p in self.kc_pools_L]
            self.kc_pools_R_gpu = [torch.from_numpy(p).long().to("cuda") for p in self.kc_pools_R]
            self.slot_pools_L_gpu = [torch.from_numpy(self.plastic_slot[p]).long().to("cuda") for p in self.kc_pools_L]
            self.slot_pools_R_gpu = [torch.from_numpy(self.plastic_slot[p]).long().to("cuda") for p in self.kc_pools_R]
            self.homeo_pools_L_gpu = [torch.from_numpy(self.kc_homeo_scale[p]).to("cuda")[None, :] for p in self.kc_pools_L]
            self.homeo_pools_R_gpu = [torch.from_numpy(self.kc_homeo_scale[p]).to("cuda")[None, :] for p in self.kc_pools_R]
            self._cuda_ready = True

        if need_coo_fallback and self.W_T_gpu is None:
            W_T = self.W_base_csr.T.tocoo()
            indices_2d = np.vstack((W_T.row, W_T.col)).astype(np.int64)
            self.W_T_gpu = torch.sparse_coo_tensor(
                torch.from_numpy(indices_2d),
                torch.from_numpy(W_T.data.astype(np.float32)),
                size=(self.N, self.N),
                device="cuda",
            ).coalesce()


def _discrete_epsp_peak(tau_m_ms: float, tau_syn_ms: float = 5.0) -> float:
    dm = math.exp(-1.0 / tau_m_ms)
    ds = math.exp(-1.0 / tau_syn_ms)
    v_imp, g_imp, pk = 0.0, 1.0, 0.0
    for _ in range(35):
        v_imp = dm * v_imp + (1.0 - dm) * g_imp
        g_imp *= ds
        if v_imp > pk:
            pk = v_imp
    return pk


class WholeCNSEngine:
    """
    Shared-CSR Whole-CNS Spiking LIF & Parallel Perturbation Engine.
    Uses an event-driven active-row CUDA Triton gather-scatter kernel (with cuSPARSE fallback)
    automatically when a GPU is available, or event-driven CSR gather-scatter + AVX2 LIF on CPU.
    """
    NAIVE_VIS_GAIN = 0.04
    MAX_VIS_GAIN = 1.48
    KC_NAIVE_STM = 0.06
    KC_NAIVE_ITM = 0.08
    KC_NAIVE_LTM = 0.06
    KC_MAX_GAIN = 0.34
    ARENA_SYN_GAIN = 0.66
    W_SYN_BASE = 0.275

    def __init__(
        self,
        dataset: ConnectomeDataset,
        num_brains: int = 100,
        device: Optional[str] = None,
        enable_callosum: bool = True,
        seed: int = 2026,
        use_triton: Optional[bool] = None,
        enable_kc_mbon_matrix: bool = False,
        homogeneous_tau_ms: Optional[float] = None,
        common_random_numbers: bool = False,
    ):
        self.ds = dataset
        self.K = int(num_brains)
        self.N = dataset.N
        self.enable_callosum = bool(enable_callosum)
        self.homogeneous_tau_ms = float(homogeneous_tau_ms) if homogeneous_tau_ms is not None else None
        self.common_random_numbers = bool(common_random_numbers)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_triton = (
            bool(_HAS_TRITON and self.device == "cuda")
            if use_triton is None
            else bool(use_triton and _HAS_TRITON and self.device == "cuda")
        )

        self.total_neurons = self.K * self.N
        self.total_directed_edges = self.K * dataset.nnz
        self.total_biological_synapses = self.K * dataset.proofread_synapses
        self.total_plastic_synapses = self.K * dataset.plastic_synapses_per_brain

        # Assign each brain k in 0..K-1 to one of the 10 functional castes + column heterogeneity
        # (unless homogeneous_tau_ms is set for matched parallel perturbation screens)
        self.brain_specs: List[Dict] = []
        tau_list, lr_list, gaze_list, hor_list, vote_list, caste_ids = [], [], [], [], [], []
        for k in range(self.K):
            if self.K == 1 or self.homogeneous_tau_ms is not None:
                tau_val = self.homogeneous_tau_ms if self.homogeneous_tau_ms is not None else 20.0
                tau_k, lr_k, gaze_k, hor_k, vote_k, cid = tau_val, 1.0, 0.0, 115.0, 1.0, 1
                cname = "Single Whole-CNS" if self.K == 1 else "Homogeneous Whole-CNS"
                ccode = "C1"
            else:
                cid = (k * 10) // self.K if self.K < 10 else (k % 10)
                col_idx = k // 10
                n_cols = max(1, math.ceil(self.K / 10))
                frac = 0.5 if n_cols == 1 else (col_idx / max(1, n_cols - 1))
                cdef = CASTE_DEFINITIONS[cid]
                t_lo, t_hi = cdef["tau_range"]
                tau_k = t_lo + frac * (t_hi - t_lo)
                lr_k = cdef["lr_mult"] * (0.90 + 0.20 * frac)
                gaze_k = cdef["gaze_deg"] * (0.65 + 0.70 * frac) if cdef["gaze_deg"] != 0.0 else 0.0
                hor_k = cdef["horizon_mm"] * (0.85 + 0.30 * frac)
                vote_k = cdef["vote_w"]
                cname, ccode = cdef["name"], cdef["code"]

            self.brain_specs.append({
                "brain_idx": k, "caste_id": cid, "caste_code": ccode, "caste_name": cname,
                "tau_m_ms": round(tau_k, 2), "lr_mult": round(lr_k, 3),
                "gaze_deg": round(gaze_k, 2), "horizon_mm": round(hor_k, 1),
            })
            tau_list.append(tau_k)
            lr_list.append(lr_k)
            gaze_list.append(gaze_k)
            hor_list.append(hor_k)
            vote_list.append(vote_k)
            caste_ids.append(cid)

        self.caste_id_arr = np.array(caste_ids, dtype=np.int32)
        self.tau_m = np.array(tau_list, dtype=np.float32)
        self.lr_mult = np.array(lr_list, dtype=np.float32)
        self.gaze_rad = np.radians(np.array(gaze_list, dtype=np.float32))
        self.horizon_mm = np.array(hor_list, dtype=np.float32)
        raw_v = np.array(vote_list, dtype=np.float32)
        self.vote_weights = raw_v / raw_v.sum()

        # Biophysical LIF constants (Shiu et al., Nature 2024)
        self.v_rest = np.float32(-52.0)
        self.v_reset = np.float32(-52.0)
        self.v_th = np.float32(-45.0)
        self.decay_syn = np.float32(math.exp(-1.0 / 5.0))

        ref_pk = _discrete_epsp_peak(20.0)
        self.w_norm = np.array([ref_pk / _discrete_epsp_peak(float(tm)) for tm in self.tau_m], dtype=np.float32)

        decay_m = np.exp(-1.0 / self.tau_m[:, None]).astype(np.float32)
        one_minus_dm = (1.0 - decay_m).astype(np.float32)
        c_rest = (self.v_rest * one_minus_dm).astype(np.float32)

        # Active slot mask
        self.active_brain_mask = np.ones(self.K, dtype=bool)
        self.rng = np.random.default_rng(seed)

        if self.device == "cuda":
            self.ds.ensure_cuda_tensors(need_coo_fallback=not self.use_triton)
            self.W_T_gpu = self.ds.W_T_gpu
            self.cuda_gen = torch.Generator(device="cuda")
            self.cuda_gen.manual_seed(int(seed))
            self.v_t = torch.full((self.K, self.N), float(self.v_rest), dtype=torch.float32, device="cuda")
            self.g_t = torch.zeros((self.K, self.N), dtype=torch.float32, device="cuda")
            self.syn_gain_t = torch.ones((self.K, self.N), dtype=torch.float32, device="cuda")
            self.elig_t = torch.zeros((self.K, self.ds.plastic_neuron_count), dtype=torch.float32, device="cuda")
            self.elig_slow_t = torch.zeros((self.K, self.ds.plastic_neuron_count), dtype=torch.float32, device="cuda")
            self._post_nk_gpu = torch.zeros((self.N, self.K), dtype=torch.float32, device="cuda")
            self.dm_t = torch.from_numpy(decay_m).to("cuda")
            self.om_t = torch.from_numpy(one_minus_dm).to("cuda")
            self.cr_t = torch.from_numpy(c_rest).to("cuda")
            self.wnorm_t = torch.from_numpy(self.w_norm).to("cuda")
            self.lr_mult_gpu = torch.from_numpy(self.lr_mult).to("cuda")
            self.active_mask_gpu = torch.from_numpy(self.active_brain_mask).to("cuda")
            self.caste_id_gpu = torch.from_numpy(self.caste_id_arr).to("cuda")
            self.ring_gpu = [
                torch.zeros((self.K, self.N), dtype=torch.bool, device="cuda"),
                torch.zeros((self.K, self.N), dtype=torch.bool, device="cuda"),
            ]
        else:
            self.cuda_gen = None
            self.v = np.full((self.K, self.N), self.v_rest, dtype=np.float32)
            self.g = np.zeros((self.K, self.N), dtype=np.float32)
            self.neuron_syn_gain = np.ones((self.K, self.N), dtype=np.float32)
            self.elig = np.zeros((self.K, self.ds.plastic_neuron_count), dtype=np.float32)
            self.elig_slow = np.zeros((self.K, self.ds.plastic_neuron_count), dtype=np.float32)
            self.v_t = torch.from_numpy(self.v)
            self.g_t = torch.from_numpy(self.g)
            self.dm_t = torch.from_numpy(decay_m)
            self.om_t = torch.from_numpy(one_minus_dm)
            self.cr_t = torch.from_numpy(c_rest)
            self.delay_ring: List[List[np.ndarray]] = [
                [np.empty(0, dtype=np.int32) for _ in range(self.K)],
                [np.empty(0, dtype=np.int32) for _ in range(self.K)],
            ]

        self._silenced_k_cpu = np.empty(0, dtype=np.int64)
        self._silenced_n_cpu = np.empty(0, dtype=np.int64)
        self._silenced_k_gpu: Optional[torch.Tensor] = None
        self._silenced_n_gpu: Optional[torch.Tensor] = None
        self.kc_mbon_matrix = None
        self.reset_plasticity(learned_init=False)
        if enable_kc_mbon_matrix:
            self.attach_kc_mbon_matrix()

    def attach_kc_mbon_matrix(self, use_fp16_on_cuda: bool = True):
        """Attach compartment-factored KC x MBON synaptic matrix M_km in R^{K x N_KC x N_MBON}."""
        from src.kc_mbon_matrix import CompartmentKCMBONMatrix
        self.kc_mbon_matrix = CompartmentKCMBONMatrix(self, use_fp16_on_cuda=use_fp16_on_cuda)
        return self.kc_mbon_matrix

    def memory_footprint_mb(self) -> Dict[str, float]:
        shared_csr_mb = (
            self.ds.W_base_csr.data.nbytes + self.ds.W_base_csr.indices.nbytes + self.ds.W_base_csr.indptr.nbytes
        ) / (1024.0 ** 2)
        kc_mbon_mb = self.kc_mbon_matrix.vram_mb() if getattr(self, "kc_mbon_matrix", None) is not None else 0.0
        state_mb = (3 * self.K * self.N * 4 + self.K * self.ds.plastic_neuron_count * 4) / (1024.0 ** 2) + kc_mbon_mb
        out = {
            "shared_connectome_csr_mb": round(shared_csr_mb, 2),
            "per_ensemble_state_mb": round(state_mb, 2),
            "total_ram_or_vram_mb": round(shared_csr_mb + state_mb, 2),
            "naive_unshared_equivalent_mb": round(self.K * shared_csr_mb + state_mb, 2),
        }
        if kc_mbon_mb > 0.0:
            out["kc_mbon_matrix_mb"] = round(kc_mbon_mb, 2)
        return out

    def reset_state(self):
        if self.device == "cuda":
            self.v_t.fill_(float(self.v_rest))
            self.g_t.zero_()
            self.elig_t.zero_()
            self.elig_slow_t.zero_()
            self.ring_gpu[0].zero_()
            self.ring_gpu[1].zero_()
        else:
            self.v.fill(self.v_rest)
            self.g.fill(0.0)
            self.elig.fill(0.0)
            self.elig_slow.fill(0.0)
            self.delay_ring = [
                [np.empty(0, dtype=np.int32) for _ in range(self.K)],
                [np.empty(0, dtype=np.int32) for _ in range(self.K)],
            ]
        if getattr(self, "kc_mbon_matrix", None) is not None:
            self.kc_mbon_matrix.reset_eligibility()

    def reset_plasticity(self, learned_init: bool = False):
        vis_val = self.MAX_VIS_GAIN if learned_init else self.NAIVE_VIS_GAIN
        stm_val = self.KC_MAX_GAIN if learned_init else self.KC_NAIVE_STM
        itm_val = self.KC_MAX_GAIN if learned_init else self.KC_NAIVE_ITM
        ltm_val = self.KC_MAX_GAIN if learned_init else self.KC_NAIVE_LTM

        if self.device == "cuda":
            self.syn_gain_t.fill_(1.0)
            self.syn_gain_t[:, self.ds.idx_vis_gpu] = vis_val
            self.syn_gain_t[:, self.ds.idx_kcg_gpu] = stm_val
            self.syn_gain_t[:, self.ds.idx_itm_gpu] = itm_val
            self.syn_gain_t[:, self.ds.idx_ltm_gpu] = ltm_val
            if self._silenced_k_gpu is not None:
                self.syn_gain_t[self._silenced_k_gpu, self._silenced_n_gpu] = 0.0
        else:
            self.neuron_syn_gain.fill(1.0)
            self.neuron_syn_gain[:, self.ds.vis_avoid_neurons] = vis_val
            self.neuron_syn_gain[:, self.ds.kcg_neurons] = stm_val
            self.neuron_syn_gain[:, self.ds.itm_neurons] = itm_val
            self.neuron_syn_gain[:, self.ds.ltm_neurons] = ltm_val
            if len(self._silenced_k_cpu) > 0:
                self.neuron_syn_gain[self._silenced_k_cpu, self._silenced_n_cpu] = 0.0
        if getattr(self, "kc_mbon_matrix", None) is not None:
            self.kc_mbon_matrix.reset(base_gain=1.0)
        self.reset_state()

    def init_xor_mushroom_body(self, base_kcg_gain: float = 0.28, base_itm_gain: float = 0.16, base_ltm_gain: float = 0.04):
        """Initialize homeostatically balanced bilateral KC->MBON gains below KC->KC runaway threshold."""
        if self.device == "cuda":
            self.syn_gain_t.fill_(1.0)
            self.syn_gain_t[:, self.ds.idx_vis_gpu] = 0.22
            self.syn_gain_t[:, self.ds.idx_kcg_gpu] = float(base_kcg_gain) * self.ds.homeo_kcg_gpu
            self.syn_gain_t[:, self.ds.idx_itm_gpu] = float(base_itm_gain) * self.ds.homeo_itm_gpu
            self.syn_gain_t[:, self.ds.idx_ltm_gpu] = float(base_ltm_gain) * self.ds.homeo_ltm_gpu
        else:
            self.neuron_syn_gain.fill(1.0)
            self.neuron_syn_gain[:, self.ds.vis_avoid_neurons] = 0.22
            self.neuron_syn_gain[:, self.ds.kcg_neurons] = float(base_kcg_gain) * self.ds.homeo_kcg[None, :]
            self.neuron_syn_gain[:, self.ds.itm_neurons] = float(base_itm_gain) * self.ds.homeo_itm[None, :]
            self.neuron_syn_gain[:, self.ds.ltm_neurons] = float(base_ltm_gain) * self.ds.homeo_ltm[None, :]
        self.reset_state()

    def apply_contextual_dopamine(self, valence: float, strength: float = 1.0):
        """
        Purely local, zero-label 3-factor Mushroom Body synaptic plasticity:
          - valence < 0: PPL1 aversive dopamine induces LTD on fast Ca2+-eligible KCs (chosen shock arm)
                         and relief LTP on cAMP-tagged opposite-hemisphere KCs of the active XOR cue;
                         also extinguishes active KCab LTM during surprise rule reversals.
          - valence > 0: PAM appetitive dopamine potentiates fast Ca2+-eligible KCs (open corridor).
        """
        MAX_HOMEO_MULT = 0.54
        MIN_HOMEO_MULT = 0.03
        v_abs = abs(float(valence))
        if self.device == "cuda":
            lr_k = (0.14 * float(strength)) * (self.lr_mult_gpu * self.active_mask_gpu.float())[:, None]
            ltm_m = self.ds.pool_ltm_mask_gpu
            stm_m = 1.0 - ltm_m
            e_all_L = torch.cat([self.elig_t.index_select(1, s) for s in self.ds.slot_pools_L_gpu], dim=1).mean(dim=1, keepdim=True)
            e_all_R = torch.cat([self.elig_t.index_select(1, s) for s in self.ds.slot_pools_R_gpu], dim=1).mean(dim=1, keepdim=True)
            choose_L = (e_all_L >= e_all_R).float()
            for j in range(4):
                idxL, slL, hmL = self.ds.kc_pools_L_gpu[j], self.ds.slot_pools_L_gpu[j], self.ds.homeo_pools_L_gpu[j]
                idxR, slR, hmR = self.ds.kc_pools_R_gpu[j], self.ds.slot_pools_R_gpu[j], self.ds.homeo_pools_R_gpu[j]
                fL = torch.clamp(self.elig_t.index_select(1, slL) - 0.03, min=0.0, max=2.0)
                sL = torch.clamp(self.elig_slow_t.index_select(1, slL) - 0.05, min=0.0, max=2.0)
                fR = torch.clamp(self.elig_t.index_select(1, slR) - 0.03, min=0.0, max=2.0)
                sR = torch.clamp(self.elig_slow_t.index_select(1, slR) - 0.05, min=0.0, max=2.0)
                # Hybrid synapse + compartmental Dop1R1/Dop1R2 volume eligibility (Cohn et al., Neuron 2015)
                actL = 0.35 * torch.clamp(fL + 0.8 * sL, max=2.0) + 0.65 * (fL.mean(dim=1, keepdim=True) + 0.8 * sL.mean(dim=1, keepdim=True))
                actR = 0.35 * torch.clamp(fR + 0.8 * sR, max=2.0) + 0.65 * (fR.mean(dim=1, keepdim=True) + 0.8 * sR.mean(dim=1, keepdim=True))
                oppL = 0.35 * sL + 0.65 * sL.mean(dim=1, keepdim=True)
                oppR = 0.35 * sR + 0.65 * sR.mean(dim=1, keepdim=True)
                for idx_h, hm_h, is_ch, act_h, opp_h in [
                    (idxL, hmL, choose_L, actL, oppL),
                    (idxR, hmR, 1.0 - choose_L, actR, oppR),
                ]:
                    cur = self.syn_gain_t.index_select(1, idx_h)
                    if valence < 0.0:
                        ltd = (14.0 * ltm_m + 3.40 * stm_m) * v_abs
                        delta = -is_ch * ltd * act_h + (1.0 - is_ch) * (2.50 * stm_m * v_abs) * opp_h
                    else:
                        ext = (12.0 * ltm_m + 2.80 * stm_m) * v_abs
                        delta = is_ch * (2.40 * stm_m * v_abs) * act_h - (1.0 - is_ch) * ext * opp_h
                    self.syn_gain_t[:, idx_h] = torch.clamp(cur + lr_k * hm_h * delta, min=MIN_HOMEO_MULT * hm_h, max=MAX_HOMEO_MULT * hm_h)
            return

        intact = self.active_brain_mask[:, None].astype(np.float32)
        lr_k = (0.14 * float(strength) * self.lr_mult[:, None]) * intact
        ltm_m = self.ds.pool_ltm_mask[None, :]
        stm_m = 1.0 - ltm_m
        e_all_L = np.concatenate([self.elig[:, self.ds.plastic_slot[p]] for p in self.ds.kc_pools_L], axis=1).mean(axis=1, keepdims=True)
        e_all_R = np.concatenate([self.elig[:, self.ds.plastic_slot[p]] for p in self.ds.kc_pools_R], axis=1).mean(axis=1, keepdims=True)
        choose_L = (e_all_L >= e_all_R).astype(np.float32)
        for j in range(4):
            pL, pR = self.ds.kc_pools_L[j], self.ds.kc_pools_R[j]
            slL, slR = self.ds.plastic_slot[pL], self.ds.plastic_slot[pR]
            hmL, hmR = self.ds.kc_homeo_scale[pL][None, :], self.ds.kc_homeo_scale[pR][None, :]
            fL = np.clip(self.elig[:, slL] - 0.03, 0.0, 2.0)
            sL = np.clip(self.elig_slow[:, slL] - 0.05, 0.0, 2.0)
            fR = np.clip(self.elig[:, slR] - 0.03, 0.0, 2.0)
            sR = np.clip(self.elig_slow[:, slR] - 0.05, 0.0, 2.0)
            actL = 0.35 * np.minimum(2.0, fL + 0.8 * sL) + 0.65 * (fL.mean(axis=1, keepdims=True) + 0.8 * sL.mean(axis=1, keepdims=True))
            actR = 0.35 * np.minimum(2.0, fR + 0.8 * sR) + 0.65 * (fR.mean(axis=1, keepdims=True) + 0.8 * sR.mean(axis=1, keepdims=True))
            oppL = 0.35 * sL + 0.65 * sL.mean(axis=1, keepdims=True)
            oppR = 0.35 * sR + 0.65 * sR.mean(axis=1, keepdims=True)
            for p_h, hm_h, is_ch, act_h, opp_h in [
                (pL, hmL, choose_L, actL, oppL),
                (pR, hmR, 1.0 - choose_L, actR, oppR),
            ]:
                cur = self.neuron_syn_gain[:, p_h]
                if valence < 0.0:
                    ltd = (14.0 * ltm_m + 3.40 * stm_m) * v_abs
                    delta = -is_ch * ltd * act_h + (1.0 - is_ch) * (2.50 * stm_m * v_abs) * opp_h
                else:
                    ext = (12.0 * ltm_m + 2.80 * stm_m) * v_abs
                    delta = is_ch * (2.40 * stm_m * v_abs) * act_h - (1.0 - is_ch) * ext * opp_h
                self.neuron_syn_gain[:, p_h] = np.clip(cur + lr_k * hm_h * delta, MIN_HOMEO_MULT * hm_h, MAX_HOMEO_MULT * hm_h)

    def run_topographic_sleep_replay(self, n_waves: int = 24) -> Dict[str, float]:
        """
        Topographic Neuron-to-Neuron Offline Sleep Replay (0.5 Hz Slow Waves):
        Transfers compartment-specific 8-pool XOR engrams from C3 KCg Flash Learners
        into homologous KCab LTM neurons across all intact brains without external labels.
        """
        if self.device == "cuda":
            flash_mask = (self.caste_id_gpu == 3) & self.active_mask_gpu
            donor_mask = flash_mask if bool(flash_mask.any().item()) else self.active_mask_gpu
            donor_idx = torch.nonzero(donor_mask, as_tuple=False).squeeze(-1)
            donor_kcg = self.syn_gain_t.index_select(0, donor_idx).index_select(1, self.ds.idx_kcg_paired_gpu).mean(dim=0, keepdim=True)
            donor_norm = donor_kcg / self.ds.homeo_kcg_paired_gpu
            has_castes = 1.0 if bool(flash_mask.any().item()) else 0.35
            sharp = 1.65 if has_castes > 0.5 else 0.75
            sharpened_norm = torch.clamp(0.28 + sharp * (donor_norm - 0.28), min=0.04, max=0.58)
            target_ltm = sharpened_norm * self.ds.homeo_ltm_paired_gpu
            pre_contrast = float(self.syn_gain_t.index_select(1, self.ds.idx_ltm_paired_gpu).std().item())
            intact_f = self.active_mask_gpu.float()[:, None]
            ltm_rate = torch.where(self.caste_id_gpu == 4, 0.115, 0.075 * has_castes).float()[:, None] * intact_f
            hm_ltm = self.ds.homeo_ltm_paired_gpu
            hm_kcg = self.ds.homeo_kcg_paired_gpu
            for _ in range(n_waves):
                cur_ltm = self.syn_gain_t.index_select(1, self.ds.idx_ltm_paired_gpu)
                self.syn_gain_t[:, self.ds.idx_ltm_paired_gpu] = torch.clamp(
                    cur_ltm + ltm_rate * (target_ltm - cur_ltm), min=0.03 * hm_ltm, max=0.58 * hm_ltm
                )
                cur_kcg = self.syn_gain_t.index_select(1, self.ds.idx_kcg_paired_gpu)
                self.syn_gain_t[:, self.ds.idx_kcg_paired_gpu] = torch.clamp(
                    cur_kcg + (0.045 * intact_f) * (donor_kcg - cur_kcg), min=0.03 * hm_kcg, max=0.58 * hm_kcg
                )
            post_contrast = float(self.syn_gain_t.index_select(1, self.ds.idx_ltm_paired_gpu).std().item())
            return {"pre_ltm_std": round(pre_contrast, 4), "post_ltm_std": round(post_contrast, 4)}

        flash_mask = (self.caste_id_arr == 3) & self.active_brain_mask
        donor_mask = flash_mask if np.any(flash_mask) else self.active_brain_mask
        donor_kcg = self.neuron_syn_gain[donor_mask][:, self.ds.kcg_paired].mean(axis=0, keepdims=True)
        donor_norm = donor_kcg / self.ds.homeo_kcg_paired[None, :]
        has_castes = 1.0 if np.any(flash_mask) else 0.35
        sharp = 1.65 if has_castes > 0.5 else 0.75
        sharpened_norm = np.clip(0.28 + sharp * (donor_norm - 0.28), 0.04, 0.58)
        target_ltm = sharpened_norm * self.ds.homeo_ltm_paired[None, :]
        pre_contrast = float(self.neuron_syn_gain[:, self.ds.ltm_paired].std())
        intact_f = self.active_brain_mask[:, None].astype(np.float32)
        ltm_rate = np.where(self.caste_id_arr == 4, 0.115, 0.075 * has_castes).astype(np.float32)[:, None] * intact_f
        hm_ltm = self.ds.homeo_ltm_paired[None, :]
        hm_kcg = self.ds.homeo_kcg_paired[None, :]
        for _ in range(n_waves):
            cur_ltm = self.neuron_syn_gain[:, self.ds.ltm_paired]
            self.neuron_syn_gain[:, self.ds.ltm_paired] = np.clip(cur_ltm + ltm_rate * (target_ltm - cur_ltm), 0.03 * hm_ltm, 0.58 * hm_ltm)
            cur_kcg = self.neuron_syn_gain[:, self.ds.kcg_paired]
            self.neuron_syn_gain[:, self.ds.kcg_paired] = np.clip(cur_kcg + (0.045 * intact_f) * (donor_kcg - cur_kcg), 0.03 * hm_kcg, 0.58 * hm_kcg)
        post_contrast = float(self.neuron_syn_gain[:, self.ds.ltm_paired].std())
        return {"pre_ltm_std": round(pre_contrast, 4), "post_ltm_std": round(post_contrast, 4)}

    def washout_kc_stm(self, washout_fraction: float = 0.80, base_kcg_gain: float = 0.28, base_itm_gain: float = 0.16):
        """Erase waking KCg and KCa'b' associative STM back toward homeostatic baseline to isolate KCab LTM."""
        keep = float(1.0 - washout_fraction)
        if self.device == "cuda":
            base_g = float(base_kcg_gain) * self.ds.homeo_kcg_gpu
            cur_g = self.syn_gain_t.index_select(1, self.ds.idx_kcg_gpu)
            self.syn_gain_t[:, self.ds.idx_kcg_gpu] = base_g + keep * (cur_g - base_g)
            base_i = float(base_itm_gain) * self.ds.homeo_itm_gpu
            cur_i = self.syn_gain_t.index_select(1, self.ds.idx_itm_gpu)
            self.syn_gain_t[:, self.ds.idx_itm_gpu] = base_i + keep * (cur_i - base_i)
        else:
            base_g = float(base_kcg_gain) * self.ds.homeo_kcg[None, :]
            cur_g = self.neuron_syn_gain[:, self.ds.kcg_neurons]
            self.neuron_syn_gain[:, self.ds.kcg_neurons] = base_g + keep * (cur_g - base_g)
            base_i = float(base_itm_gain) * self.ds.homeo_itm[None, :]
            cur_i = self.neuron_syn_gain[:, self.ds.itm_neurons]
            self.neuron_syn_gain[:, self.ds.itm_neurons] = base_i + keep * (cur_i - base_i)

    def clone_plasticity_state(self):
        if self.device == "cuda":
            return self.syn_gain_t.detach().clone()
        return self.neuron_syn_gain.copy()

    def load_plasticity_state(self, state):
        if self.device == "cuda":
            if isinstance(state, torch.Tensor):
                self.syn_gain_t.copy_(state.to("cuda"))
            else:
                self.syn_gain_t.copy_(torch.from_numpy(state).to("cuda"))
        else:
            if isinstance(state, torch.Tensor):
                self.neuron_syn_gain[:] = state.cpu().numpy()
            else:
                self.neuron_syn_gain[:] = state

    def washout_waking_stm(self, washout_fraction: float = 0.65):
        """Wash out a fraction (default 65%) of waking visual avoidance STM to test KCab LTM retention."""
        keep = float(1.0 - washout_fraction)
        if self.device == "cuda":
            cur_v = self.syn_gain_t.index_select(1, self.ds.idx_vis_gpu)
            self.syn_gain_t[:, self.ds.idx_vis_gpu] = float(self.NAIVE_VIS_GAIN) + keep * (cur_v - float(self.NAIVE_VIS_GAIN))
        else:
            cur_v = self.neuron_syn_gain[:, self.ds.vis_avoid_neurons]
            self.neuron_syn_gain[:, self.ds.vis_avoid_neurons] = self.NAIVE_VIS_GAIN + keep * (cur_v - self.NAIVE_VIS_GAIN)

    def set_active_slot_mask(self, active_mask: np.ndarray):
        assert len(active_mask) == self.K
        self.active_brain_mask = active_mask.astype(bool).copy()
        if self.device == "cuda":
            self.active_mask_gpu = torch.from_numpy(self.active_brain_mask).to("cuda")

    def clear_perturbations(self, learned_init: bool = False):
        """Clear per-slot cell-type silencing masks and restore baseline synaptic gains."""
        self._silenced_k_cpu = np.empty(0, dtype=np.int64)
        self._silenced_n_cpu = np.empty(0, dtype=np.int64)
        self._silenced_k_gpu = None
        self._silenced_n_gpu = None
        self.reset_plasticity(learned_init=learned_init)

    def set_perturbation_masks(
        self,
        silenced_per_brain: Optional[List[Optional[np.ndarray]]] = None,
        learned_init: bool = False,
    ):
        """
        Configure parallel virtual Split-GAL4 silencing (Kir2.1 voltage clamp + TNT release block)
        independently across each of the K shared-CSR connectome slots.
        """
        self.clear_perturbations(learned_init=learned_init)
        if not silenced_per_brain:
            return
        assert len(silenced_per_brain) == self.K, f"Expected {self.K} masks, got {len(silenced_per_brain)}"
        ks_list, ns_list = [], []
        for k_idx, idx_arr in enumerate(silenced_per_brain):
            if idx_arr is None or len(idx_arr) == 0:
                continue
            arr = np.unique(np.asarray(idx_arr, dtype=np.int64))
            ks_list.append(np.full(len(arr), k_idx, dtype=np.int64))
            ns_list.append(arr)
        if not ks_list:
            return
        sk = np.concatenate(ks_list)
        sn = np.concatenate(ns_list)
        self._silenced_k_cpu = sk
        self._silenced_n_cpu = sn
        if self.device == "cuda":
            self._silenced_k_gpu = torch.from_numpy(sk).long().to("cuda")
            self._silenced_n_gpu = torch.from_numpy(sn).long().to("cuda")
            self.syn_gain_t[self._silenced_k_gpu, self._silenced_n_gpu] = 0.0
        else:
            self.neuron_syn_gain[sk, sn] = 0.0

    def set_seed(self, seed: int):
        self.rng = np.random.default_rng(int(seed))
        if self.device == "cuda" and self.cuda_gen is not None:
            self.cuda_gen.manual_seed(int(seed))

    def apply_ppl1_punishment_ltp(self, strength: float = 1.0):
        """Three-factor PPL1 Dopamine LTP across all intact brains weighted by caste plasticity rate."""
        if self.device == "cuda":
            lr_k = (0.38 * float(strength)) * (self.lr_mult_gpu * self.active_mask_gpu.float())[:, None]
            vis_boost = lr_k * (0.35 + torch.clamp(self.elig_t.index_select(1, self.ds.slot_vis_gpu), max=1.5))
            self.syn_gain_t[:, self.ds.idx_vis_gpu] = torch.clamp(
                self.syn_gain_t.index_select(1, self.ds.idx_vis_gpu) + vis_boost,
                min=self.NAIVE_VIS_GAIN, max=self.MAX_VIS_GAIN,
            )
            stm_boost = 0.22 * lr_k * (0.30 + torch.clamp(self.elig_t.index_select(1, self.ds.slot_kcg_gpu), max=1.5))
            self.syn_gain_t[:, self.ds.idx_kcg_gpu] = torch.clamp(
                self.syn_gain_t.index_select(1, self.ds.idx_kcg_gpu) + stm_boost,
                min=self.KC_NAIVE_STM, max=self.KC_MAX_GAIN,
            )
            itm_boost = 0.14 * lr_k * (0.25 + torch.clamp(self.elig_t.index_select(1, self.ds.slot_itm_gpu), max=1.5))
            self.syn_gain_t[:, self.ds.idx_itm_gpu] = torch.clamp(
                self.syn_gain_t.index_select(1, self.ds.idx_itm_gpu) + itm_boost,
                min=self.KC_NAIVE_ITM, max=self.KC_MAX_GAIN,
            )
            self.syn_gain_t[:, self.ds.idx_mbon_app_gpu] = torch.clamp(
                self.syn_gain_t.index_select(1, self.ds.idx_mbon_app_gpu) - 0.10 * lr_k,
                min=0.48, max=1.35,
            )
            return

        gains = self.neuron_syn_gain
        elig = self.elig
        intact = self.active_brain_mask[:, None].astype(np.float32)
        lr_k = (0.38 * strength * self.lr_mult[:, None]) * intact
        vis_boost = lr_k * (0.35 + np.minimum(1.5, elig[:, self.ds.slot_vis]))
        gains[:, self.ds.vis_avoid_neurons] = np.clip(
            gains[:, self.ds.vis_avoid_neurons] + vis_boost, self.NAIVE_VIS_GAIN, self.MAX_VIS_GAIN
        )
        stm_boost = 0.22 * lr_k * (0.30 + np.minimum(1.5, elig[:, self.ds.slot_kcg]))
        gains[:, self.ds.kcg_neurons] = np.clip(
            gains[:, self.ds.kcg_neurons] + stm_boost, self.KC_NAIVE_STM, self.KC_MAX_GAIN
        )
        itm_boost = 0.14 * lr_k * (0.25 + np.minimum(1.5, elig[:, self.ds.slot_itm]))
        gains[:, self.ds.itm_neurons] = np.clip(
            gains[:, self.ds.itm_neurons] + itm_boost, self.KC_NAIVE_ITM, self.KC_MAX_GAIN
        )
        gains[:, self.ds.mbon_app_neurons] = np.clip(
            gains[:, self.ds.mbon_app_neurons] - 0.10 * lr_k, 0.48, 1.35
        )

    def apply_pam_reward_ltp(self, strength: float = 1.0):
        """Three-factor PAM Dopamine corridor stabilization on ITM & Approach MBONs."""
        if self.device == "cuda":
            lr_k = (0.18 * float(strength)) * (self.lr_mult_gpu * self.active_mask_gpu.float())[:, None]
            self.syn_gain_t[:, self.ds.idx_itm_gpu] = torch.clamp(
                self.syn_gain_t.index_select(1, self.ds.idx_itm_gpu) + 0.16 * lr_k,
                min=self.KC_NAIVE_ITM, max=self.KC_MAX_GAIN,
            )
            return

        gains = self.neuron_syn_gain
        intact = self.active_brain_mask[:, None].astype(np.float32)
        lr_k = (0.18 * strength * self.lr_mult[:, None]) * intact
        gains[:, self.ds.itm_neurons] = np.clip(
            gains[:, self.ds.itm_neurons] + 0.16 * lr_k, self.KC_NAIVE_ITM, self.KC_MAX_GAIN
        )

    def run_offline_sleep_replay(self, n_waves: int = 24) -> Dict[str, float]:
        """
        24-Wave DPM/APL-Inspired Offline Sleep Replay Consolidation:
        Transfers waking STM (KCg + LC16) and ITM (KCa'b') engrams from fast C3 Flash Learners
        across the Callosum into permanent Alpha/Beta KCab LTM across the ensemble.
        """
        if self.device == "cuda":
            pre_ltm = float(self.syn_gain_t.index_select(1, self.ds.idx_ltm_gpu).mean().item())
            flash_mask = (self.caste_id_gpu == 3) & self.active_mask_gpu
            donor_mask = flash_mask if bool(flash_mask.any().item()) else self.active_mask_gpu
            donor_idx = torch.nonzero(donor_mask, as_tuple=False).squeeze(-1)
            donor_sub = self.syn_gain_t.index_select(0, donor_idx)
            donor_vis = donor_sub.index_select(1, self.ds.idx_vis_gpu).mean(dim=0, keepdim=True)
            donor_itm = donor_sub.index_select(1, self.ds.idx_itm_gpu).mean(dim=0, keepdim=True)

            stm_frac = float(np.clip(
                (float(donor_vis.mean().item()) - self.NAIVE_VIS_GAIN) / max(1e-6, self.MAX_VIS_GAIN - self.NAIVE_VIS_GAIN),
                0.0, 1.0,
            ))
            if stm_frac > 0.01:
                intact_f = self.active_mask_gpu.float()[:, None]
                ltm_rate = torch.where(self.caste_id_gpu == 4, 0.065, 0.042).float()[:, None] * intact_f
                target_ltm = float(self.KC_NAIVE_LTM + stm_frac * (self.KC_MAX_GAIN - self.KC_NAIVE_LTM))
                for _ in range(n_waves):
                    cur_v = self.syn_gain_t.index_select(1, self.ds.idx_vis_gpu)
                    self.syn_gain_t[:, self.ds.idx_vis_gpu] = torch.clamp(
                        cur_v + (0.045 * intact_f) * torch.clamp(donor_vis - cur_v, min=0.0),
                        min=self.NAIVE_VIS_GAIN, max=self.MAX_VIS_GAIN,
                    )
                    cur_i = self.syn_gain_t.index_select(1, self.ds.idx_itm_gpu)
                    self.syn_gain_t[:, self.ds.idx_itm_gpu] = torch.clamp(
                        cur_i + (0.040 * intact_f) * torch.clamp(donor_itm - cur_i, min=0.0),
                        min=self.KC_NAIVE_ITM, max=self.KC_MAX_GAIN,
                    )
                    cur_l = self.syn_gain_t.index_select(1, self.ds.idx_ltm_gpu)
                    self.syn_gain_t[:, self.ds.idx_ltm_gpu] = torch.clamp(
                        cur_l + ltm_rate * (target_ltm - cur_l),
                        min=self.KC_NAIVE_LTM, max=self.KC_MAX_GAIN,
                    )
            post_ltm = float(self.syn_gain_t.index_select(1, self.ds.idx_ltm_gpu).mean().item())
            return {"pre_ltm_gain": round(pre_ltm, 4), "post_ltm_gain": round(post_ltm, 4), "stm_frac": round(stm_frac, 4)}

        gains = self.neuron_syn_gain
        pre_ltm = float(gains[:, self.ds.ltm_neurons].mean())
        flash_mask = (self.caste_id_arr == 3) & self.active_brain_mask
        donor_mask = flash_mask if np.any(flash_mask) else self.active_brain_mask

        donor_vis = gains[donor_mask][:, self.ds.vis_avoid_neurons].mean(axis=0)
        donor_itm = gains[donor_mask][:, self.ds.itm_neurons].mean(axis=0)

        stm_frac = float(np.clip(
            (donor_vis.mean() - self.NAIVE_VIS_GAIN) / max(1e-6, self.MAX_VIS_GAIN - self.NAIVE_VIS_GAIN), 0.0, 1.0
        ))
        if stm_frac > 0.01:
            for _ in range(n_waves):
                for k in range(self.K):
                    if not self.active_brain_mask[k]:
                        continue
                    cur_v = gains[k, self.ds.vis_avoid_neurons]
                    gains[k, self.ds.vis_avoid_neurons] = np.clip(
                        cur_v + 0.045 * np.maximum(0.0, donor_vis - cur_v), self.NAIVE_VIS_GAIN, self.MAX_VIS_GAIN
                    )
                    cur_i = gains[k, self.ds.itm_neurons]
                    gains[k, self.ds.itm_neurons] = np.clip(
                        cur_i + 0.040 * np.maximum(0.0, donor_itm - cur_i), self.KC_NAIVE_ITM, self.KC_MAX_GAIN
                    )
                    target_ltm = self.KC_NAIVE_LTM + stm_frac * (self.KC_MAX_GAIN - self.KC_NAIVE_LTM)
                    ltm_rate = 0.065 if self.caste_id_arr[k] == 4 else 0.042
                    gains[k, self.ds.ltm_neurons] = np.clip(
                        gains[k, self.ds.ltm_neurons] + ltm_rate * (target_ltm - gains[k, self.ds.ltm_neurons]),
                        self.KC_NAIVE_LTM, self.KC_MAX_GAIN,
                    )

        post_ltm = float(gains[:, self.ds.ltm_neurons].mean())
        return {"pre_ltm_gain": round(pre_ltm, 4), "post_ltm_gain": round(post_ltm, 4), "stm_frac": round(stm_frac, 4)}

    def tier_mean_gains(self) -> Dict[str, float]:
        if self.device == "cuda":
            intact_idx = torch.nonzero(self.active_mask_gpu, as_tuple=False).squeeze(-1)
            g_sub = self.syn_gain_t.index_select(0, intact_idx)
            means = torch.stack([
                g_sub.index_select(1, self.ds.idx_vis_gpu).mean(),
                g_sub.index_select(1, self.ds.idx_kcg_gpu).mean(),
                g_sub.index_select(1, self.ds.idx_itm_gpu).mean(),
                g_sub.index_select(1, self.ds.idx_ltm_gpu).mean(),
            ]).cpu().tolist()
            return {
                "vis_stm": float(means[0]),
                "kcg_stm": float(means[1]),
                "kcabp_itm": float(means[2]),
                "kcab_ltm": float(means[3]),
            }

        gains = self.neuron_syn_gain
        m = self.active_brain_mask
        return {
            "vis_stm": float(gains[m][:, self.ds.vis_avoid_neurons].mean()),
            "kcg_stm": float(gains[m][:, self.ds.kcg_neurons].mean()),
            "kcabp_itm": float(gains[m][:, self.ds.itm_neurons].mean()),
            "kcab_ltm": float(gains[m][:, self.ds.ltm_neurons].mean()),
        }

    def step_population(
        self,
        n_steps_ms: int,
        stimuli_per_brain: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        shared_stimuli: Optional[List[Tuple[np.ndarray, float]]] = None,
        bg_poisson_hz: float = 0.0,
        bg_subset_indices: Optional[np.ndarray] = None,
        readout_pools: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Advance all K connectomes by `n_steps_ms` (dt = 1.0 ms).
        - shared_stimuli: list of (neuron_indices_1d, rate_hz_scalar_or_K_array[, amp_val])
        - bg_poisson_hz: independent uncorrelated synaptic Poisson noise per brain
        - readout_pools: optional dict of {name: 1D neuron index array} for custom circuit readouts
        """
        w_eff = np.float32(self.W_SYN_BASE * self.ARENA_SYN_GAIN)
        dn_L_counts = np.zeros(self.K, dtype=np.float32)
        dn_R_counts = np.zeros(self.K, dtype=np.float32)
        leg_L_counts = np.zeros(self.K, dtype=np.float32)
        leg_R_counts = np.zeros(self.K, dtype=np.float32)
        mbon_L_counts = np.zeros(self.K, dtype=np.float32)
        mbon_R_counts = np.zeros(self.K, dtype=np.float32)
        male_p1_counts = np.zeros(self.K, dtype=np.float32)
        total_spike_counts = np.zeros(self.K, dtype=np.int64)
        step_turn_series = np.zeros((n_steps_ms, self.K), dtype=np.float32)

        exec_mask = (self.caste_id_arr == 9) & self.active_brain_mask
        non_exec_mask = (self.caste_id_arr != 9) & self.active_brain_mask

        if self.device == "cuda":
            return self._step_cuda(
                n_steps_ms, shared_stimuli, bg_poisson_hz, bg_subset_indices, w_eff,
                dn_L_counts, dn_R_counts, leg_L_counts, leg_R_counts, male_p1_counts,
                total_spike_counts, step_turn_series, exec_mask, non_exec_mask, readout_pools,
            )

        pool_counts = {k: np.zeros(self.K, dtype=np.float32) for k in readout_pools} if readout_pools else None
        pool_masks_cpu = None
        if readout_pools:
            pool_masks_cpu = {}
            for k_name, idx_p in readout_pools.items():
                m_bool = np.zeros(self.N, dtype=bool)
                if len(idx_p) > 0:
                    m_bool[idx_p] = True
                pool_masks_cpu[k_name] = (m_bool, max(1, len(idx_p)))

        W_base = self.ds.W_base_csr
        w_indptr = W_base.indptr
        w_indices = W_base.indices
        w_data = W_base.data
        intact = self.active_brain_mask

        for s_idx in range(n_steps_ms):
            arriving = self.delay_ring.pop(0)   # t - 2 spikes
            prev1 = self.delay_ring[0]          # t - 1 spikes

            self.g_t.mul_(float(self.decay_syn))
            callosum_L = 0.0
            callosum_R = 0.0

            # Event-driven CSR row gather-scatter (6.3x faster than SPA csr_matmat in sparse <0.3% LIF regime!)
            scale_k_cpu = (float(w_eff) * self.w_norm)[:, None].copy()
            for b in range(self.K):
                if not intact[b]:
                    scale_k_cpu[b, 0] = 0.0
                    continue
                sp_b = arriving[b]
                kb = len(sp_b)
                if kb > 0:
                    homeo = 1.0 / (1.0 + max(0, kb - 185) * 0.010)
                    sb = float(w_eff * self.w_norm[b] * homeo)
                    scale_k_cpu[b, 0] = sb
                    vals_b = sb * self.neuron_syn_gain[b, sp_b]
                    gb = self.g[b]
                    for u_idx, w_u in zip(sp_b, vals_b):
                        s0, s1 = w_indptr[u_idx], w_indptr[u_idx + 1]
                        if s1 > s0:
                            gb[w_indices[s0:s1]] += float(w_u) * w_data[s0:s1]
                    if self.enable_callosum and non_exec_mask[b]:
                        callosum_L += float(self.ds.mask_dn_L[sp_b].sum())
                        callosum_R += float(self.ds.mask_dn_R[sp_b].sum())

            if self.kc_mbon_matrix is not None:
                self.kc_mbon_matrix.inject_step_cpu(arriving, scale_k_cpu, self.g, self.neuron_syn_gain)

            # Commissural Callosum coupling from specialist castes C0..C8 onto executive caste C9!
            if self.enable_callosum and np.any(exec_mask) and np.any(non_exec_mask):
                n_senders = max(1.0, float(non_exec_mask.sum()))
                net_cal = (callosum_R - callosum_L) / math.sqrt(n_senders)
                if net_cal > 0.35:
                    for b in np.flatnonzero(exec_mask):
                        self.g[b, self.ds.idx_dn_R] += min(4.5, net_cal * 0.55) * float(self.w_norm[b])
                elif net_cal < -0.35:
                    for b in np.flatnonzero(exec_mask):
                        self.g[b, self.ds.idx_dn_L] += min(4.5, (-net_cal) * 0.55) * float(self.w_norm[b])

            # Vectorized 2D Poisson external & sensory stimuli across all K brains
            r_rows = 1 if self.common_random_numbers else self.K
            if shared_stimuli:
                for stim_item in shared_stimuli:
                    idx_arr, rate_spec = stim_item[0], stim_item[1]
                    amp_val = float(stim_item[2]) if len(stim_item) > 2 else 18.0
                    if len(idx_arr) == 0:
                        continue
                    if np.ndim(rate_spec) == 0:
                        if float(rate_spec) <= 0.5:
                            continue
                        p_mat = 1.0 - math.exp(-float(rate_spec) * 0.001)
                        draws = (self.rng.random((r_rows, len(idx_arr))) < p_mat) & intact[:, None]
                    else:
                        r_vec = np.asarray(rate_spec, dtype=np.float32)[:, None]
                        if float(r_vec.max()) <= 0.5:
                            continue
                        p_vec = 1.0 - np.exp(-np.maximum(0.0, r_vec) * 0.001)
                        draws = (self.rng.random((r_rows, len(idx_arr))) < p_vec) & intact[:, None]
                    ks, js = np.nonzero(draws)
                    if len(ks) > 0:
                        np.add.at(self.g, (ks, idx_arr[js]), amp_val * self.w_norm[ks])

            if bg_poisson_hz > 0.5 and bg_subset_indices is not None and len(bg_subset_indices) > 0:
                p_bg = 1.0 - math.exp(-bg_poisson_hz * 0.001)
                draws_bg = (self.rng.random((r_rows, len(bg_subset_indices))) < p_bg) & intact[:, None]
                ks, js = np.nonzero(draws_bg)
                if len(ks) > 0:
                    np.add.at(self.g, (ks, bg_subset_indices[js]), 14.5 * self.w_norm[ks])

            # Multi-core AVX2 Fused LIF Voltage Update
            self.g_t.clamp_(min=-65.0, max=95.0)
            self.v_t.mul_(self.dm_t).add_(self.cr_t).addcmul_(self.g_t, self.om_t).clamp_min_(-72.0)

            for b in range(self.K):
                if not intact[b]:
                    self.v[b].fill(self.v_rest)
                    continue
                if len(arriving[b]) > 0:
                    self.v[b, arriving[b]] = self.v_reset
                if len(prev1[b]) > 0:
                    self.v[b, prev1[b]] = self.v_reset

            if len(self._silenced_k_cpu) > 0:
                self.v[self._silenced_k_cpu, self._silenced_n_cpu] = self.v_reset

            sp_nz = (self.v_t >= -45.0).nonzero()
            new_slot = [np.empty(0, dtype=np.int32) for _ in range(self.K)]
            self.elig *= 0.991
            self.elig_slow *= 0.997

            if sp_nz.shape[0] > 0:
                sp_np = sp_nz.numpy()
                sp_b = sp_np[:, 0]
                sp_n = sp_np[:, 1]
                self.v[sp_b, sp_n] = self.v_reset

                # Update dual-timescale eligibility traces on plastic neurons
                pslots = self.ds.plastic_slot[sp_n]
                pm = pslots >= 0
                if np.any(pm):
                    self.elig[sp_b[pm], pslots[pm]] += 1.0
                    self.elig_slow[sp_b[pm], pslots[pm]] += 1.0

                order = np.argsort(sp_b, kind="mergesort")
                sp_b_s = sp_b[order]
                sp_n_s = sp_n[order]
                uniq_b, starts, counts = np.unique(sp_b_s, return_index=True, return_counts=True)
                for ub, st, ct in zip(uniq_b, starts, counts):
                    neurons_b = sp_n_s[st:st + ct].astype(np.int32)
                    new_slot[int(ub)] = neurons_b
                    total_spike_counts[int(ub)] += int(ct)
                    dnL = float(self.ds.mask_dn_L[neurons_b].sum()) / self.ds.n_dn_L
                    dnR = float(self.ds.mask_dn_R[neurons_b].sum()) / self.ds.n_dn_R
                    lgL = float(self.ds.mask_leg_L[neurons_b].sum()) / self.ds.n_leg_L
                    lgR = float(self.ds.mask_leg_R[neurons_b].sum()) / self.ds.n_leg_R
                    mbL = float(self.ds.mask_mbon_app_L[neurons_b].sum()) / self.ds.n_mbon_app_L
                    mbR = float(self.ds.mask_mbon_app_R[neurons_b].sum()) / self.ds.n_mbon_app_R
                    dn_L_counts[int(ub)] += dnL
                    dn_R_counts[int(ub)] += dnR
                    leg_L_counts[int(ub)] += lgL
                    leg_R_counts[int(ub)] += lgR
                    mbon_L_counts[int(ub)] += mbL
                    mbon_R_counts[int(ub)] += mbR
                    if len(self.ds.idx_male_p1) > 0:
                        male_p1_counts[int(ub)] += float(np.isin(neurons_b, self.ds.idx_male_p1).sum())
                    if pool_masks_cpu is not None:
                        for k_name, (m_bool, _) in pool_masks_cpu.items():
                            pool_counts[k_name][int(ub)] += float(m_bool[neurons_b].sum())
                    step_turn_series[s_idx, int(ub)] = ((dnR - dnL) + 0.30 * (lgR - lgL)) * 1000.0

            self.delay_ring.append(new_slot)

        dur_s = max(1e-6, n_steps_ms * 0.001)
        dn_L_hz = dn_L_counts / dur_s
        dn_R_hz = dn_R_counts / dur_s
        leg_L_hz = leg_L_counts / dur_s
        leg_R_hz = leg_R_counts / dur_s
        mbon_L_hz = mbon_L_counts / dur_s
        mbon_R_hz = mbon_R_counts / dur_s
        mbon_val_hz = mbon_L_hz - mbon_R_hz
        turn_hz = (dn_R_hz - dn_L_hz) + 0.30 * (leg_R_hz - leg_L_hz)
        mean_brain_hz = total_spike_counts.astype(np.float32) / (self.N * dur_s)

        active_w = self.vote_weights * intact.astype(np.float32)
        w_sum = float(active_w.sum())
        consensus_turn_hz = float(np.dot(turn_hz, active_w) / w_sum) if w_sum > 1e-8 else 0.0
        consensus_mbon_hz = float(np.dot(mbon_val_hz, active_w) / w_sum) if w_sum > 1e-8 else 0.0

        out_dict = {
            "dn_L_hz": dn_L_hz,
            "dn_R_hz": dn_R_hz,
            "leg_L_hz": leg_L_hz,
            "leg_R_hz": leg_R_hz,
            "mbon_app_L_hz": mbon_L_hz,
            "mbon_app_R_hz": mbon_R_hz,
            "consensus_mbon_valence_hz": consensus_mbon_hz,
            "turn_hz_per_brain": turn_hz,
            "consensus_turn_hz": consensus_turn_hz,
            "step_turn_series": step_turn_series,
            "mean_brain_hz": mean_brain_hz,
            "male_p1_hz": male_p1_counts / (max(1, len(self.ds.idx_male_p1)) * dur_s),
        }
        if pool_masks_cpu is not None:
            p_hz = {k: pool_counts[k] / (pool_masks_cpu[k][1] * dur_s) for k in pool_masks_cpu}
            out_dict["pool_hz"] = p_hz
            out_dict["pool_consensus_hz"] = {
                k: (float(np.dot(v, active_w) / w_sum) if w_sum > 1e-8 else 0.0) for k, v in p_hz.items()
            }
        return out_dict

    def _step_cuda(
        self, n_steps_ms, shared_stimuli, bg_poisson_hz, bg_subset_indices, w_eff,
        dn_L_counts, dn_R_counts, leg_L_counts, leg_R_counts, male_p1_counts,
        total_spike_counts, step_turn_series, exec_mask, non_exec_mask, readout_pools=None,
    ) -> Dict[str, np.ndarray]:
        """Hardware-accelerated event-driven Triton CSR / cuSPARSE SpMM path for NVIDIA GPUs."""
        intact_t = self.active_mask_gpu
        exec_t = torch.from_numpy(exec_mask).to("cuda")
        non_exec_t = torch.from_numpy(non_exec_mask).to("cuda")
        has_cal = bool(self.enable_callosum and np.any(exec_mask) and np.any(non_exec_mask))
        inv_sqrt_senders = 1.0 / math.sqrt(max(1.0, float(non_exec_mask.sum())))
        exec_idx = torch.nonzero(exec_t, as_tuple=False).squeeze(-1) if has_cal else None

        dnL_acc = torch.zeros(self.K, dtype=torch.float32, device="cuda")
        dnR_acc = torch.zeros(self.K, dtype=torch.float32, device="cuda")
        lgL_acc = torch.zeros(self.K, dtype=torch.float32, device="cuda")
        lgR_acc = torch.zeros(self.K, dtype=torch.float32, device="cuda")
        mbL_acc = torch.zeros(self.K, dtype=torch.float32, device="cuda")
        mbR_acc = torch.zeros(self.K, dtype=torch.float32, device="cuda")
        p1_acc = torch.zeros(self.K, dtype=torch.float32, device="cuda")
        tot_acc = torch.zeros(self.K, dtype=torch.int64, device="cuda")
        series_gpu = torch.zeros((n_steps_ms, self.K), dtype=torch.float32, device="cuda")

        if not hasattr(self, "_gpu_idx_cache"):
            self._gpu_idx_cache = {}

        def _cached_idx_tensor(tag: str, arr: np.ndarray) -> torch.Tensor:
            n_a = len(arr)
            key = (tag, id(arr), n_a, int(arr[0]), int(arr[n_a // 2]), int(arr[-1]))
            hit = self._gpu_idx_cache.get(key)
            if hit is not None and hit[0] is arr:
                return hit[1]
            t_gpu = torch.from_numpy(arr).long().to("cuda")
            if len(self._gpu_idx_cache) < 192:
                self._gpu_idx_cache[key] = (arr, t_gpu)
            return t_gpu

        pool_gpu = None
        if readout_pools:
            pool_gpu = []
            for p_name, p_arr in readout_pools.items():
                if len(p_arr) == 0:
                    pool_gpu.append((p_name, None, 1, torch.zeros(self.K, dtype=torch.float32, device="cuda")))
                else:
                    p_t = _cached_idx_tensor("pool", p_arr)
                    pool_gpu.append((p_name, p_t, len(p_arr), torch.zeros(self.K, dtype=torch.float32, device="cuda")))
        stim_gpu = []
        if shared_stimuli:
            for stim_item in shared_stimuli:
                idx_arr, rate_spec = stim_item[0], stim_item[1]
                amp_val = float(stim_item[2]) if len(stim_item) > 2 else 18.0
                if len(idx_arr) == 0:
                    continue
                idx_t = _cached_idx_tensor("stim", idx_arr)
                if np.ndim(rate_spec) == 0:
                    if float(rate_spec) <= 0.5:
                        continue
                    p_val = 1.0 - math.exp(-float(rate_spec) * 0.001)
                    stim_gpu.append((idx_t, p_val, amp_val))
                else:
                    r_np = np.asarray(rate_spec, dtype=np.float32)
                    if float(r_np.max()) <= 0.5:
                        continue
                    rv = torch.from_numpy(r_np).to("cuda")[:, None]
                    p_vec = 1.0 - torch.exp(-torch.clamp(rv, min=0.0) * 0.001)
                    stim_gpu.append((idx_t, p_vec, amp_val))

        if bg_poisson_hz > 0.5 and bg_subset_indices is not None and len(bg_subset_indices) > 0:
            bg_t = _cached_idx_tensor("bg", bg_subset_indices)
        else:
            bg_t = None
        p_bg = (1.0 - math.exp(-bg_poisson_hz * 0.001)) if bg_t is not None else 0.0
        block_k = 16 if self.K <= 16 else 128
        grid_kb = triton.cdiv(self.K, block_k) if self.use_triton else 0

        for s_idx in range(n_steps_ms):
            arriving = self.ring_gpu.pop(0) & intact_t[:, None]  # (K, N) bool
            prev1 = self.ring_gpu[0]

            kb = arriving.sum(dim=1).float()
            homeo = 1.0 / (1.0 + torch.clamp(kb - 185.0, min=0.0) * 0.010)
            scale_k = (float(w_eff) * self.wnorm_t * homeo)[:, None]

            self.g_t.mul_(float(self.decay_syn))
            if self.use_triton:
                active_u = torch.nonzero(arriving.any(dim=0), as_tuple=False).squeeze(-1)
                m_act = int(active_u.numel())
                if m_act > 0:
                    w_act = (
                        arriving.index_select(1, active_u).float()
                        * self.syn_gain_t.index_select(1, active_u)
                        * scale_k
                    ).t().contiguous()  # (M, K)
                    self._post_nk_gpu.zero_()
                    _active_csr_scatter_kernel[(m_act, grid_kb)](
                        active_u,
                        w_act,
                        self.ds.csr_indptr_gpu,
                        self.ds.csr_indices_gpu,
                        self.ds.csr_data_gpu,
                        self._post_nk_gpu,
                        self.K,
                        BLOCK_K=block_k,
                    )
                    self.g_t.add_(self._post_nk_gpu.t())
            else:
                weighted_pre = arriving.float() * self.syn_gain_t * scale_k
                post_drive = torch.sparse.mm(self.W_T_gpu, weighted_pre.t()).t()
                self.g_t.add_(post_drive)

            if self.kc_mbon_matrix is not None:
                self.kc_mbon_matrix.inject_step_cuda(arriving, scale_k, self.g_t, self.syn_gain_t)

            # Commissural Callosum coupling from specialist castes C0..C8 onto executive caste C9!
            if has_cal:
                cal_L = arriving[non_exec_t][:, self.ds.idx_dnL_gpu].sum().float()
                cal_R = arriving[non_exec_t][:, self.ds.idx_dnR_gpu].sum().float()
                net_cal = (cal_R - cal_L) * inv_sqrt_senders
                pos_boost = torch.clamp(net_cal * 0.55, min=0.0, max=4.5) * (net_cal > 0.35).float()
                neg_boost = torch.clamp((-net_cal) * 0.55, min=0.0, max=4.5) * (net_cal < -0.35).float()
                self.g_t[exec_idx[:, None], self.ds.idx_dnR_gpu[None, :]] += pos_boost * self.wnorm_t[exec_idx, None]
                self.g_t[exec_idx[:, None], self.ds.idx_dnL_gpu[None, :]] += neg_boost * self.wnorm_t[exec_idx, None]

            r_rows = 1 if self.common_random_numbers else self.K
            for idx_t, p_spec, amp_val in stim_gpu:
                r_draw = torch.rand((r_rows, len(idx_t)), device="cuda", generator=self.cuda_gen)
                draws = (r_draw < p_spec) & intact_t[:, None]
                self.g_t[:, idx_t] += draws.float() * (amp_val * self.wnorm_t[:, None])

            if bg_t is not None:
                r_bg = torch.rand((r_rows, len(bg_t)), device="cuda", generator=self.cuda_gen)
                draws_bg = (r_bg < p_bg) & intact_t[:, None]
                self.g_t[:, bg_t] += draws_bg.float() * (14.5 * self.wnorm_t[:, None])

            self.g_t.clamp_(min=-65.0, max=95.0)
            self.v_t.mul_(self.dm_t).add_(self.cr_t).addcmul_(self.g_t, self.om_t).clamp_min_(-72.0)
            self.v_t.masked_fill_(arriving | prev1 | (~intact_t[:, None]), float(self.v_reset))
            if self._silenced_k_gpu is not None:
                self.v_t[self._silenced_k_gpu, self._silenced_n_gpu] = float(self.v_reset)

            spikes = (self.v_t >= -45.0) & intact_t[:, None]
            self.v_t.masked_fill_(spikes, float(self.v_reset))
            pl_sp = spikes[:, self.ds.plastic_neurons_gpu].float()
            self.elig_t.mul_(0.991).add_(pl_sp)
            self.elig_slow_t.mul_(0.997).add_(pl_sp)

            dnL = spikes[:, self.ds.mask_dnL_gpu].sum(dim=1).float() / self.ds.n_dn_L
            dnR = spikes[:, self.ds.mask_dnR_gpu].sum(dim=1).float() / self.ds.n_dn_R
            lgL = spikes[:, self.ds.mask_lgL_gpu].sum(dim=1).float() / self.ds.n_leg_L
            lgR = spikes[:, self.ds.mask_lgR_gpu].sum(dim=1).float() / self.ds.n_leg_R
            mbL = spikes[:, self.ds.mask_mbon_app_L_gpu].sum(dim=1).float() / self.ds.n_mbon_app_L
            mbR = spikes[:, self.ds.mask_mbon_app_R_gpu].sum(dim=1).float() / self.ds.n_mbon_app_R
            dnL_acc.add_(dnL)
            dnR_acc.add_(dnR)
            lgL_acc.add_(lgL)
            lgR_acc.add_(lgR)
            mbL_acc.add_(mbL)
            mbR_acc.add_(mbR)
            if len(self.ds.idx_p1_gpu) > 0:
                p1_acc.add_(spikes[:, self.ds.idx_p1_gpu].sum(dim=1).float())
            if pool_gpu is not None:
                for _, p_t, _, p_acc in pool_gpu:
                    if p_t is not None:
                        p_acc.add_(spikes.index_select(1, p_t).sum(dim=1).float())
            tot_acc.add_(spikes.sum(dim=1))
            series_gpu[s_idx] = ((dnR - dnL) + 0.30 * (lgR - lgL)) * 1000.0
            self.ring_gpu.append(spikes)

        dur_s = max(1e-6, n_steps_ms * 0.001)
        dn_L_hz = dnL_acc.cpu().numpy() / dur_s
        dn_R_hz = dnR_acc.cpu().numpy() / dur_s
        leg_L_hz = lgL_acc.cpu().numpy() / dur_s
        leg_R_hz = lgR_acc.cpu().numpy() / dur_s
        mbon_L_hz = mbL_acc.cpu().numpy() / dur_s
        mbon_R_hz = mbR_acc.cpu().numpy() / dur_s
        mbon_val_hz = mbon_L_hz - mbon_R_hz
        turn_hz = (dn_R_hz - dn_L_hz) + 0.30 * (leg_R_hz - leg_L_hz)
        active_w = self.vote_weights * self.active_brain_mask.astype(np.float32)
        w_sum = float(active_w.sum())
        out_dict = {
            "dn_L_hz": dn_L_hz,
            "dn_R_hz": dn_R_hz,
            "leg_L_hz": leg_L_hz,
            "leg_R_hz": leg_R_hz,
            "mbon_app_L_hz": mbon_L_hz,
            "mbon_app_R_hz": mbon_R_hz,
            "consensus_mbon_valence_hz": float(np.dot(mbon_val_hz, active_w) / w_sum) if w_sum > 1e-8 else 0.0,
            "turn_hz_per_brain": turn_hz,
            "consensus_turn_hz": float(np.dot(turn_hz, active_w) / w_sum) if w_sum > 1e-8 else 0.0,
            "step_turn_series": series_gpu.cpu().numpy(),
            "mean_brain_hz": tot_acc.cpu().numpy().astype(np.float32) / (self.N * dur_s),
            "male_p1_hz": p1_acc.cpu().numpy() / (max(1, len(self.ds.idx_male_p1)) * dur_s),
        }
        if pool_gpu is not None:
            p_hz = {p_name: (p_acc.cpu().numpy() / (p_len * dur_s)) for p_name, _, p_len, p_acc in pool_gpu}
            out_dict["pool_hz"] = p_hz
            out_dict["pool_consensus_hz"] = {
                k: (float(np.dot(v, active_w) / w_sum) if w_sum > 1e-8 else 0.0) for k, v in p_hz.items()
            }
        return out_dict


def extract_male_social_pools(ds_m: ConnectomeDataset) -> Dict[str, np.ndarray]:
    """Extract canonical courtship and sensory-gating cell-type indices in MaleCNS v1.0."""
    mt = ds_m.meta_df["type"].fillna("").astype(str)
    ms = ds_m.meta_df["somaSide"].fillna("").astype(str)
    pip10 = np.flatnonzero(mt.str.contains("pIP10", case=False)).astype(np.int32)
    dpr_vpr = np.flatnonzero(mt.isin(["dPR1", "vPR6"])).astype(np.int32)
    return {
        "va1v": np.flatnonzero(mt.str.contains("VA1v", case=False)).astype(np.int32),
        "da1": np.flatnonzero(mt.str.contains("DA1", case=False)).astype(np.int32),
        "lc10a_L": np.flatnonzero((mt == "LC10a") & (ms == "L")).astype(np.int32),
        "lc10a_R": np.flatnonzero((mt == "LC10a") & (ms == "R")).astype(np.int32),
        "p1": np.flatnonzero(mt.str.startswith(("P1", "pC1"))).astype(np.int32),
        "mal_m": np.flatnonzero(mt.str.startswith("mAL")).astype(np.int32),
        "pip10": pip10,
        "pip10_song": np.unique(np.concatenate([pip10, dpr_vpr])).astype(np.int32),
        "mdn": ds_m.idx_mdn,
    }


def extract_female_social_pools(ds_f: ConnectomeDataset) -> Dict[str, np.ndarray]:
    """Extract canonical auditory song-relay and vpoDN receptivity indices in Female BANC v888."""
    ft = ds_f.meta_df["type"].fillna("").astype(str)
    ffw = ds_f.meta_df["flywireType"].fillna("").astype(str)
    fs = ds_f.meta_df["somaSide"].fillna("").astype(str)
    pc2l_pre = np.flatnonzero(
        ffw.str.contains("pC2l|AN_AVLP_GNG_19|AVLP370|AN17A015|PVLP093", case=False)
    ).astype(np.int32)
    cb_aud_relay = np.flatnonzero(
        ffw.str.contains("CB1484|WED118_a|CB1908|AN01A055|CB1078|WED104|CB1383", case=False)
    ).astype(np.int32)
    return {
        "jo_b": np.flatnonzero(ft == "JO-B").astype(np.int32),
        "cb1484": cb_aud_relay,
        "da1": np.flatnonzero(ffw.str.contains("DA1", case=False)).astype(np.int32),
        "sag": np.flatnonzero(ffw.str.contains(r"\bSAG(?:-like)?\b", case=False)).astype(np.int32),
        "pc1_abc": np.flatnonzero(ft.isin(["pC1a", "pC1b", "pC1c", "pC1e"])).astype(np.int32),
        "pc1d": np.flatnonzero(ft == "pC1d").astype(np.int32),
        "pc2l_pre": pc2l_pre,
        "aipg": np.flatnonzero(ffw.str.contains(r"aIP[-_]?g|CB1127|CB2131", case=False)).astype(np.int32),
        "vpoen": np.flatnonzero(ft == "vpoEN").astype(np.int32),
        "vpodn": np.flatnonzero(ffw.str.contains(r"\bvpoDN\b|\bDNp37\b", case=False)).astype(np.int32),
        "dnp13": np.flatnonzero(ft == "DNp13").astype(np.int32),
        "ovidn": np.flatnonzero(ffw.str.contains("oviDN", case=False)).astype(np.int32),
        "vnc_accept": np.flatnonzero(ft.isin(["INXXX089", "INXXX910_f", "IN13A011", "INXXX866_f"])).astype(np.int32),
        "vnc_reject": np.flatnonzero(
            ft.isin(["INXXX247", "INXXX985_f", "ab-PDF"]) | ft.str.startswith(("MNad", "EFFabg"))
        ).astype(np.int32),
        "lc10a_L": np.flatnonzero((ft == "LC10a") & (fs == "L")).astype(np.int32),
        "lc10a_R": np.flatnonzero((ft == "LC10a") & (fs == "R")).astype(np.int32),
        "mdn": ds_f.idx_mdn,
    }
