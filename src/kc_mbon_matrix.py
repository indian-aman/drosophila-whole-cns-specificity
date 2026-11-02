"""
Compartment-Factored Mushroom-Body KC x MBON Synaptic Matrix (MaleCNS v1.0 & Female BANC v888)
==============================================================================================
Organizes Kenyon-cell-to-MBON synapses across the 15 canonical Mushroom Body compartments
(gamma1..5, alpha'1..3, beta'1..2, alpha1..3, beta1..2; Aso et al., eLife 2014).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch


CANONICAL_COMPARTMENTS: List[Dict[str, str]] = [
    # Gamma lobe (Fast Short-Term Memory, STM)
    {"id": 0,  "name": "gamma1",   "tier": "STM", "lobe": "gamma",       "dan_family": "PPL1"},
    {"id": 1,  "name": "gamma2",   "tier": "STM", "lobe": "gamma",       "dan_family": "PPL1"},
    {"id": 2,  "name": "gamma3",   "tier": "STM", "lobe": "gamma",       "dan_family": "PAM"},
    {"id": 3,  "name": "gamma4",   "tier": "STM", "lobe": "gamma",       "dan_family": "PAM"},
    {"id": 4,  "name": "gamma5",   "tier": "STM", "lobe": "gamma",       "dan_family": "PAM"},
    # Alpha'/Beta' lobe (Intermediate-Term Memory, ITM)
    {"id": 5,  "name": "alpha_p1", "tier": "ITM", "lobe": "alpha_prime", "dan_family": "PPL1_PAM"},
    {"id": 6,  "name": "alpha_p2", "tier": "ITM", "lobe": "alpha_prime", "dan_family": "PPL1"},
    {"id": 7,  "name": "alpha_p3", "tier": "ITM", "lobe": "alpha_prime", "dan_family": "PPL1"},
    {"id": 8,  "name": "beta_p1",  "tier": "ITM", "lobe": "alpha_prime", "dan_family": "PAM"},
    {"id": 9,  "name": "beta_p2",  "tier": "ITM", "lobe": "alpha_prime", "dan_family": "PAM"},
    # Alpha/Beta lobe (Consolidated Long-Term Memory, LTM)
    {"id": 10, "name": "alpha1",   "tier": "LTM", "lobe": "alpha_beta",  "dan_family": "PAM"},
    {"id": 11, "name": "alpha2",   "tier": "LTM", "lobe": "alpha_beta",  "dan_family": "PPL1"},
    {"id": 12, "name": "alpha3",   "tier": "LTM", "lobe": "alpha_beta",  "dan_family": "PPL1"},
    {"id": 13, "name": "beta1",    "tier": "LTM", "lobe": "alpha_beta",  "dan_family": "PAM"},
    {"id": 14, "name": "beta2",    "tier": "LTM", "lobe": "alpha_beta",  "dan_family": "PAM"},
]

COMPARTMENT_NAME_TO_ID: Dict[str, int] = {c["name"]: int(c["id"]) for c in CANONICAL_COMPARTMENTS}

# Aso et al. (eLife 2014) canonical MBON01..MBON35 primary dendritic compartment mapping
MBON_TYPE_TO_COMPARTMENT: Dict[str, str] = {
    "MBON01": "gamma5",   # gamma5beta'2a (Glut, avoidance)
    "MBON02": "beta2",    # beta2beta'2a (Glut)
    "MBON03": "beta_p2",  # beta'2mp (Glut)
    "MBON04": "beta_p2",  # beta'2mp_bilateral (Glut)
    "MBON05": "gamma4",   # gamma4>gamma1gamma2 (Glut)
    "MBON06": "beta1",    # beta1>alpha (Glut)
    "MBON07": "alpha1",   # alpha1 (Glut/ACh LTM)
    "MBON08": "gamma3",   # sc/gamma3
    "MBON09": "gamma3",   # gamma3beta'1 (GABA)
    "MBON10": "beta_p1",  # beta'1 (GABA)
    "MBON11": "gamma1",   # gamma1pedc>alpha/beta (GABA master gate)
    "MBON12": "gamma2",   # gamma2alpha'1 (ACh approach)
    "MBON13": "alpha_p2", # alpha'2 (ACh approach)
    "MBON14": "alpha3",   # alpha3 (ACh LTM approach)
    "MBON15": "alpha_p1", # alpha'1 (ACh/Glut)
    "MBON16": "alpha_p1", # alpha'1 (ACh)
    "MBON17": "alpha_p3", # alpha'3 (ACh)
    "MBON18": "alpha2",   # alpha2sc (ACh LTM approach)
    "MBON19": "beta1",    # alpha2p3p / beta1 (ACh)
    "MBON20": "gamma1",   # gamma1gamma2 (ACh/GABA)
    "MBON21": "gamma4",   # gamma4gamma5 (ACh/Glut)
    "MBON22": "beta2",    # calyx/beta2 (ACh)
    "MBON23": "alpha2",   # alpha2sp (ACh LTM)
    "MBON24": "beta_p2",  # beta2gamma5 / beta'2 (ACh)
    "MBON25": "gamma2",   # gamma1gamma2 / gamma2 (ACh)
    "MBON26": "beta_p2",  # beta'2d (ACh/Glut)
    "MBON27": "gamma5",   # gamma5d (ACh/Glut)
    "MBON28": "beta2",    # alpha'3/beta2
    "MBON29": "gamma4",   # gamma4gamma5
    "MBON30": "gamma1",   # gamma1gamma2
    "MBON31": "gamma2",   # gamma2alpha'1
    "MBON32": "gamma2",   # gamma2alpha'1
    "MBON33": "beta2",    # beta2/gamma
    "MBON34": "alpha2",
    "MBON35": "gamma2",
}

# DAN cell-type prefix -> primary compartment (Aso et al., eLife 2014)
DAN_PREFIX_TO_COMPARTMENT: Dict[str, str] = {
    "PPL101": "gamma1",
    "PPL102": "gamma1",
    "PPL103": "gamma2",
    "PPL104": "alpha_p3",
    "PPL105": "alpha2",
    "PPL106": "alpha3",
    "PPL107": "alpha_p2",
    "PPL108": "alpha_p1",
    "PAM01":  "gamma5",
    "PAM02":  "beta_p2",
    "PAM03":  "beta2",
    "PAM04":  "beta2",
    "PAM05":  "beta_p2",
    "PAM06":  "beta_p2",
    "PAM07":  "alpha_p1",
    "PAM08":  "gamma4",
    "PAM09":  "beta1",
    "PAM10":  "beta_p1",
    "PAM11":  "alpha1",
    "PAM12":  "gamma3",
    "PAM13":  "beta1",
    "PAM14":  "beta1",
    "PAM15":  "gamma5",
}

# Functional task behavioral channels -> disjoint compartment sets
TASK_CHANNEL_COMPARTMENTS: Dict[str, Dict[str, List[str]]] = {
    "navigation_xor": {
        "stm": ["gamma2"],
        "itm": ["alpha_p1", "alpha_p2"],
        "ltm": ["alpha2", "alpha3", "alpha1"],
    },
    "social_courtship": {
        "stm": ["gamma5", "gamma1"],
        "itm": ["beta_p2", "beta_p1"],
        "ltm": ["beta1", "beta2"],
    },
    "foraging_odor": {
        "stm": ["gamma4", "gamma3"],
        "itm": ["alpha_p3"],
        "ltm": ["beta2"],
    },
}


@dataclass
class KCMBONAnatomy:
    kc_idx: np.ndarray              # (N_KC,) global neuron indices
    mbon_idx: np.ndarray            # (N_MBON,) global neuron indices
    dan_idx: np.ndarray             # (N_DAN,) global neuron indices
    kc_side_is_R: np.ndarray        # (N_KC,) bool
    kc_tier: np.ndarray             # (N_KC,) int: 0=STM(gamma), 1=ITM(alpha'beta'), 2=LTM(alphabeta)
    mbon_side_is_R: np.ndarray      # (N_MBON,) bool
    mbon_comp_id: np.ndarray        # (N_MBON,) int in 0..14
    mbon_tier: np.ndarray           # (N_MBON,) int: 0=STM, 1=ITM, 2=LTM
    mbon_valence: np.ndarray        # (N_MBON,) float32: +1.0 (ACh approach) or -1.0 (GABA/Glut avoidance)
    mbon_types: List[str]
    dan_comp_id: np.ndarray         # (N_DAN,) int in 0..14
    dan_is_ppl1: np.ndarray         # (N_DAN,) bool (True=aversive PPL1, False=appetitive PAM)
    W0_km: np.ndarray               # (N_KC, N_MBON) float32 exact EM signed weights from W_base_csr
    W_eff_km: np.ndarray            # (N_KC, N_MBON) float32 effective structural weights (EM + homeostatic calyx-lobe)
    struct_mask_km: np.ndarray      # (N_KC, N_MBON) float32 in {0, 1} anatomical connectivity mask
    C_dan_mbon: np.ndarray          # (N_DAN, N_MBON) float32 normalized compartmental DAN->MBON coupling
    em_kc_mbon_edges: int
    em_dan_mbon_edges: int

    def summary(self) -> Dict[str, int]:
        return {
            "n_kc": int(len(self.kc_idx)),
            "n_mbon": int(len(self.mbon_idx)),
            "n_dan": int(len(self.dan_idx)),
            "n_compartments": len(CANONICAL_COMPARTMENTS),
            "em_kc_mbon_edges": int(self.em_kc_mbon_edges),
            "em_dan_mbon_edges": int(self.em_dan_mbon_edges),
            "effective_kc_mbon_contacts": int(self.struct_mask_km.sum()),
        }


def extract_kc_mbon_anatomy(ds) -> KCMBONAnatomy:
    """Extracts exact KC, MBON, and DAN compartmental wiring directly from ConnectomeDataset."""
    meta = ds.meta_df
    c_type = meta["type"].fillna("").astype(str)
    c_class = meta["class"].fillna("").astype(str)
    sides = meta["somaSide"].fillna("").astype(str)
    nt = meta["nt_name"].fillna("").astype(str).str.lower()

    # 1. Kenyon Cells (ordered identically across KCg, KCa'b', KCab)
    is_kc = c_class.str.contains("Kenyon|KC", case=False, na=False) | c_type.str.startswith("KC")
    kc_idx = np.flatnonzero(is_kc.values).astype(np.int32)
    kc_types = c_type.iloc[kc_idx].values
    kc_side_is_R = (sides.iloc[kc_idx].values == "R")
    kc_tier = np.zeros(len(kc_idx), dtype=np.int32)
    for i, t_str in enumerate(kc_types):
        if t_str.startswith("KCa'b'") or "apbp" in t_str.lower():
            kc_tier[i] = 1
        elif t_str.startswith("KCab") or ("ab" in t_str.lower() and not t_str.startswith("KCg")):
            kc_tier[i] = 2
        else:
            kc_tier[i] = 0

    # 2. Mushroom Body Output Neurons (MBON01..MBON35)
    is_mbon = c_class.str.contains("MBON|mushroom_body_output", case=False, na=False) | c_type.str.startswith("MBON")
    mbon_idx = np.flatnonzero(is_mbon.values).astype(np.int32)
    mbon_types = c_type.iloc[mbon_idx].tolist()
    mbon_side_is_R = (sides.iloc[mbon_idx].values == "R")
    mbon_nt = nt.iloc[mbon_idx].values

    mbon_comp_id = np.zeros(len(mbon_idx), dtype=np.int32)
    mbon_tier = np.zeros(len(mbon_idx), dtype=np.int32)
    mbon_valence = np.ones(len(mbon_idx), dtype=np.float32)

    for m_i, raw_t in enumerate(mbon_types):
        base_t = raw_t.split("-")[0].split("_")[0][:6]
        c_name = MBON_TYPE_TO_COMPARTMENT.get(base_t, "gamma2" if mbon_nt[m_i] == "acetylcholine" else "gamma1")
        cid = COMPARTMENT_NAME_TO_ID[c_name]
        mbon_comp_id[m_i] = cid
        t_lbl = CANONICAL_COMPARTMENTS[cid]["tier"]
        mbon_tier[m_i] = 0 if t_lbl == "STM" else (1 if t_lbl == "ITM" else 2)
        mbon_valence[m_i] = +1.0 if mbon_nt[m_i] == "acetylcholine" else -1.0

    # 3. Dopaminergic Neurons (PPL1 & PAM)
    is_dan = c_class.str.contains("DAN", case=False, na=False) | c_type.str.startswith(("PPL1", "PAM"))
    dan_idx = np.flatnonzero(is_dan.values).astype(np.int32)
    dan_types = c_type.iloc[dan_idx].tolist()
    dan_is_ppl1 = np.array([t.startswith("PPL") for t in dan_types], dtype=bool)
    dan_comp_id = np.zeros(len(dan_idx), dtype=np.int32)
    for d_i, dt in enumerate(dan_types):
        matched = "gamma1" if dan_is_ppl1[d_i] else "gamma5"
        for pref, cname in DAN_PREFIX_TO_COMPARTMENT.items():
            if dt.startswith(pref):
                matched = cname
                break
        dan_comp_id[d_i] = COMPARTMENT_NAME_TO_ID[matched]

    # 4. Exact EM KC -> MBON and DAN -> MBON submatrices from W_base_csr
    W_km_sp = ds.W_base_csr[kc_idx][:, mbon_idx]
    em_kc_mbon_edges = int(W_km_sp.nnz)
    W0_km = W_km_sp.toarray().astype(np.float32)

    W_dm_sp = ds.W_base_csr[dan_idx][:, mbon_idx]
    em_dan_mbon_edges = int(W_dm_sp.nnz)
    raw_dm = np.abs(W_dm_sp.toarray().astype(np.float32))

    # Build normalized compartmental DAN -> MBON matrix combining EM edges + canonical Aso compartment match
    C_dan_mbon = raw_dm.copy()
    for m_i in range(len(mbon_idx)):
        col_sum = float(C_dan_mbon[:, m_i].sum())
        if col_sum < 1e-4:
            same_comp = (dan_comp_id == mbon_comp_id[m_i]).astype(np.float32)
            if same_comp.sum() > 0:
                C_dan_mbon[:, m_i] = same_comp
        col_sum = float(C_dan_mbon[:, m_i].sum())
        if col_sum > 1e-6:
            C_dan_mbon[:, m_i] /= col_sum

    # Effective structural weight matrix W_eff_km: exact EM weights where non-zero,
    # plus sparse anatomical ipsilateral lobe-matched contacts where cleft table is sparse (e.g. BANC v888)
    W_eff_km = W0_km.copy()
    rng_anat = np.random.default_rng(20261101)
    for m_i in range(len(mbon_idx)):
        nz_col = int((np.abs(W_eff_km[:, m_i]) > 0).sum())
        if nz_col < 45:
            # Anatomically consistent ipsilateral lobe KCs (Aso et al., 2014: ~25% calyx-lobe convergence)
            cand = np.flatnonzero((kc_side_is_R == mbon_side_is_R[m_i]) & (kc_tier == mbon_tier[m_i]))
            if len(cand) > 0:
                n_pick = min(len(cand), max(75, int(0.28 * len(cand))))
                picked = rng_anat.choice(cand, size=n_pick, replace=False)
                W_eff_km[picked, m_i] = np.maximum(W_eff_km[picked, m_i], 18.0)

    struct_mask_km = (np.abs(W_eff_km) > 1e-6).astype(np.float32)
    return KCMBONAnatomy(
        kc_idx=kc_idx,
        mbon_idx=mbon_idx,
        dan_idx=dan_idx,
        kc_side_is_R=kc_side_is_R,
        kc_tier=kc_tier,
        mbon_side_is_R=mbon_side_is_R,
        mbon_comp_id=mbon_comp_id,
        mbon_tier=mbon_tier,
        mbon_valence=mbon_valence,
        mbon_types=mbon_types,
        dan_comp_id=dan_comp_id,
        dan_is_ppl1=dan_is_ppl1,
        W0_km=W0_km,
        W_eff_km=W_eff_km,
        struct_mask_km=struct_mask_km,
        C_dan_mbon=C_dan_mbon,
        em_kc_mbon_edges=em_kc_mbon_edges,
        em_dan_mbon_edges=em_dan_mbon_edges,
    )


class CompartmentKCMBONMatrix:
    """
    GPU/CPU Compartment-Factored KC x MBON Synaptic Matrix (K, N_KC, N_MBON).
    Supports:
      - Zero-interference multi-compartment three-factor learning (Cohn et al., Nature 2015)
      - Real spiking DAN -> Compartment -> MBON neuromodulation
      - Cross-compartment gamma -> alpha'beta' -> alphabeta sleep consolidation
      - 80% STM/ITM compartment washout with 100% alpha/beta LTM preservation
    """

    MIN_GAIN = 0.05
    MAX_GAIN = 2.60

    def __init__(self, eng, use_fp16_on_cuda: bool = True):
        self.eng = eng
        self.ds = eng.ds
        self.K = eng.K
        self.device = eng.device

        if not hasattr(self.ds, "_kc_mbon_anatomy"):
            self.ds._kc_mbon_anatomy = extract_kc_mbon_anatomy(self.ds)
        self.anat: KCMBONAnatomy = self.ds._kc_mbon_anatomy

        self.N_kc = len(self.anat.kc_idx)
        self.N_mbon = len(self.anat.mbon_idx)
        self.N_dan = len(self.anat.dan_idx)

        # Map each KC to its plastic_slot in WholeCNSEngine (or -1 if not in plastic_neurons)
        self.kc_plastic_slot = self.ds.plastic_slot[self.anat.kc_idx].astype(np.int64)

        # Caste learning-rate multipliers across K brains
        self.lr_mult = self.eng.lr_mult.astype(np.float32)

        if self.device == "cuda":
            self.dtype_gpu = torch.float16 if use_fp16_on_cuda else torch.float32
            self.kc_idx_gpu = torch.from_numpy(self.anat.kc_idx).long().to("cuda")
            self.mbon_idx_gpu = torch.from_numpy(self.anat.mbon_idx).long().to("cuda")
            self.dan_idx_gpu = torch.from_numpy(self.anat.dan_idx).long().to("cuda")
            self.W0_gpu = torch.from_numpy(self.anat.W0_km).to(device="cuda", dtype=torch.float32)
            self.Weff_gpu = torch.from_numpy(self.anat.W_eff_km).to(device="cuda", dtype=torch.float32)
            self.mask_gpu = torch.from_numpy(self.anat.struct_mask_km).to(device="cuda", dtype=self.dtype_gpu)
            self.C_dm_gpu = torch.from_numpy(self.anat.C_dan_mbon).to(device="cuda", dtype=torch.float32)
            self.valence_gpu = torch.from_numpy(self.anat.mbon_valence).to(device="cuda", dtype=torch.float32)
            self.lr_gpu = torch.from_numpy(self.lr_mult).to(device="cuda", dtype=torch.float32)[:, None, None]

            # Core (K, N_KC, N_MBON) synaptic gain tensor
            self.M_gpu = torch.ones((self.K, self.N_kc, self.N_mbon), device="cuda", dtype=self.dtype_gpu)
            self.elig_kc_gpu = torch.zeros((self.K, self.N_kc), device="cuda", dtype=torch.float32)
            self.calyx_trace_L_gpu = {
                ch: torch.zeros((self.K, self.N_kc), device="cuda", dtype=torch.float32)
                for ch in TASK_CHANNEL_COMPARTMENTS
            }
            self.calyx_trace_R_gpu = {
                ch: torch.zeros((self.K, self.N_kc), device="cuda", dtype=torch.float32)
                for ch in TASK_CHANNEL_COMPARTMENTS
            }
            self._has_deviation = True
        else:
            self.M_cpu = np.ones((self.K, self.N_kc, self.N_mbon), dtype=np.float32)
            self.elig_kc_cpu = np.zeros((self.K, self.N_kc), dtype=np.float32)
            self.calyx_trace_L_cpu = {
                ch: np.zeros((self.K, self.N_kc), dtype=np.float32)
                for ch in TASK_CHANNEL_COMPARTMENTS
            }
            self.calyx_trace_R_cpu = {
                ch: np.zeros((self.K, self.N_kc), dtype=np.float32)
                for ch in TASK_CHANNEL_COMPARTMENTS
            }
            self._has_deviation = True

    def vram_mb(self) -> float:
        elem_bytes = 2 if (self.device == "cuda" and self.dtype_gpu == torch.float16) else 4
        return round((self.K * self.N_kc * self.N_mbon * elem_bytes) / (1024.0 ** 2), 2)

    def reset(self, base_gain: float = 1.0):
        if self.device == "cuda":
            self.M_gpu.fill_(float(base_gain))
            self.elig_kc_gpu.zero_()
            for ch in TASK_CHANNEL_COMPARTMENTS:
                self.calyx_trace_L_gpu[ch].zero_()
                self.calyx_trace_R_gpu[ch].zero_()
        else:
            self.M_cpu.fill(float(base_gain))
            self.elig_kc_cpu.fill(0.0)
            for ch in TASK_CHANNEL_COMPARTMENTS:
                self.calyx_trace_L_cpu[ch].fill(0.0)
                self.calyx_trace_R_cpu[ch].fill(0.0)
        self._has_deviation = True

    def reset_eligibility(self):
        if self.device == "cuda":
            self.elig_kc_gpu.zero_()
        else:
            self.elig_kc_cpu.fill(0.0)

    def clone_state(self) -> np.ndarray:
        if self.device == "cuda":
            return self.M_gpu.float().cpu().numpy().copy()
        return self.M_cpu.copy()

    def load_state(self, state_np: np.ndarray):
        if self.device == "cuda":
            self.M_gpu.copy_(torch.from_numpy(state_np).to(device="cuda", dtype=self.dtype_gpu))
        else:
            self.M_cpu[:] = state_np
        self._has_deviation = True

    def inject_step_cuda(self, arriving: torch.Tensor, scale_k: torch.Tensor, g_t: torch.Tensor, syn_gain_t: torch.Tensor):
        """
        Event-driven active-KC row gather-multiply into postsynaptic MBON conductances.
        Net conductance into MBON_m equals exact M_km[k, i, m] * W_eff[i, m] * scale_k.
        """
        s_kc = arriving.index_select(1, self.kc_idx_gpu).float()  # (K, N_KC)
        self.elig_kc_gpu.mul_(0.988).add_(s_kc)

        if not self._has_deviation:
            return

        act_local = torch.nonzero(s_kc.any(dim=0), as_tuple=False).squeeze(-1)
        if act_local.numel() == 0:
            return

        s_act = s_kc.index_select(1, act_local)  # (K, M_act)
        m_act = self.M_gpu.index_select(1, act_local).float()  # (K, M_act, N_MBON)
        node_g = syn_gain_t.index_select(1, self.kc_idx_gpu.index_select(0, act_local))[:, :, None]  # (K, M_act, 1)

        weff_act = self.Weff_gpu.index_select(0, act_local)[None, :, :]  # (1, M_act, N_MBON)
        w0_act = self.W0_gpu.index_select(0, act_local)[None, :, :]      # (1, M_act, N_MBON)

        # Exact differential conductance: target (M_act * Weff) minus static CSR contribution (node_g * W0)
        eff_delta = m_act * weff_act - node_g * w0_act  # (K, M_act, N_MBON)
        delta_g = torch.bmm(s_act.unsqueeze(1), eff_delta).squeeze(1) * scale_k  # (K, N_MBON)
        g_t[:, self.mbon_idx_gpu] += delta_g

    def inject_step_cpu(self, arriving_list: List[np.ndarray], scale_k: np.ndarray, g: np.ndarray, syn_gain: np.ndarray):
        self.elig_kc_cpu *= 0.988
        s_kc = np.zeros((self.K, self.N_kc), dtype=np.float32)
        global_to_local = getattr(self, "_cpu_kc_lookup", None)
        if global_to_local is None:
            lut = np.full(self.ds.N, -1, dtype=np.int32)
            lut[self.anat.kc_idx] = np.arange(self.N_kc, dtype=np.int32)
            self._cpu_kc_lookup = lut
            global_to_local = lut

        any_kc = False
        for k in range(self.K):
            arr = arriving_list[k]
            if len(arr) == 0:
                continue
            loc = global_to_local[arr]
            loc = loc[loc >= 0]
            if len(loc) > 0:
                s_kc[k, loc] = 1.0
                any_kc = True

        if any_kc:
            self.elig_kc_cpu += s_kc

        if not self._has_deviation or not any_kc:
            return

        act_local = np.flatnonzero(s_kc.any(axis=0))
        s_act = s_kc[:, act_local]  # (K, M_act)
        m_act = self.M_cpu[:, act_local, :]  # (K, M_act, N_MBON)
        node_g = syn_gain[:, self.anat.kc_idx[act_local]][:, :, None]
        eff_delta = m_act * self.anat.W_eff_km[None, act_local, :] - node_g * self.anat.W0_km[None, act_local, :]
        delta_g = np.einsum("ka,kam->km", s_act, eff_delta) * scale_k
        g[:, self.anat.mbon_idx] += delta_g

    def mbon_column_mask(
        self,
        compartments: Sequence[str],
        side: Optional[str] = None,
        valence_sign: Optional[int] = None,
    ) -> np.ndarray:
        """Boolean mask over N_MBON columns matching requested compartment names, hemisphere, and NT valence."""
        c_ids = [COMPARTMENT_NAME_TO_ID[c] for c in compartments if c in COMPARTMENT_NAME_TO_ID]
        mask = np.isin(self.anat.mbon_comp_id, c_ids)
        if side == "L":
            mask = mask & (~self.anat.mbon_side_is_R)
        elif side == "R":
            mask = mask & self.anat.mbon_side_is_R
        if valence_sign is not None:
            if valence_sign > 0:
                mask = mask & (self.anat.mbon_valence > 0)
            elif valence_sign < 0:
                mask = mask & (self.anat.mbon_valence < 0)
        return mask

    def apply_compartment_dopamine(
        self,
        compartments: Sequence[str],
        da_signal: float,
        side: Optional[str] = None,
        lr: float = 0.22,
        extinction_rate: float = 0.08,
        elig_threshold: float = 0.12,
    ) -> Dict[str, float]:
        """
        Cohn et al. (Nature 2015) compartmental three-factor plasticity rule:
          - da_signal < 0 (PPL1 punishment): depresses Approach (ACh+) MBON columns in `compartments`
            where KC eligibility > threshold, and potentiates/extinguishes silent-KC synapses.
          - da_signal > 0 (PAM reward): potentiates Approach (ACh+) MBON columns & depresses
            Avoidance (Glut/GABA-) MBON columns in `compartments` where KC eligibility > threshold.
        Crucially, ONLY columns m in `compartments` are modified; all other MBON compartments
        on the exact same Kenyon Cell axons experience ZERO weight drift!
        """
        col_mask = self.mbon_column_mask(compartments, side=side)
        cols = np.flatnonzero(col_mask)
        if len(cols) == 0 or abs(da_signal) < 1e-6:
            return {"updated_columns": 0, "mean_gain": 1.0}

        self._has_deviation = True
        matched_channels = [
            ch for ch, trs in TASK_CHANNEL_COMPARTMENTS.items()
            if any(c in (trs["stm"] + trs["itm"] + trs["ltm"]) for c in compartments)
        ]
        if self.device == "cuda":
            cols_t = torch.from_numpy(cols).long().to("cuda")
            val_sign = self.valence_gpu.index_select(0, cols_t)[None, None, :]  # (1, 1, C_sub)
            el = torch.clamp(self.elig_kc_gpu, 0.0, 3.0)[:, :, None]            # (K, N_KC, 1)
            active_kc = (el > elig_threshold).float()
            silent_kc = 1.0 - active_kc
            tr_delta = (self.lr_gpu.squeeze(-1) * float(lr) * float(da_signal)) * active_kc.squeeze(-1)
            for ch in matched_channels:
                if side in (None, "L"):
                    self.calyx_trace_L_gpu[ch] = torch.clamp(self.calyx_trace_L_gpu[ch] + tr_delta, -1.35, 1.35)
                if side in (None, "R"):
                    self.calyx_trace_R_gpu[ch] = torch.clamp(self.calyx_trace_R_gpu[ch] + tr_delta, -1.35, 1.35)

            # Signed target direction per MBON valence:
            # da_signal > 0 (reward) -> +1 on ACh approach, -1 on GABA/Glut avoidance
            # da_signal < 0 (punish) -> -1 on ACh approach, +1 on GABA/Glut avoidance
            signed_drive = float(da_signal) * val_sign
            cur_sub = self.M_gpu.index_select(2, cols_t).float()
            mask_sub = self.mask_gpu.index_select(1, cols_t).float()[None, :, :]

            hebb_delta = self.lr_gpu * float(lr) * el * signed_drive * active_kc
            ext_delta = float(extinction_rate) * abs(float(da_signal)) * (1.0 - cur_sub) * silent_kc
            updated = torch.clamp(cur_sub + (hebb_delta + ext_delta) * mask_sub, self.MIN_GAIN, self.MAX_GAIN)
            self.M_gpu.index_copy_(2, cols_t, updated.to(self.dtype_gpu))
            mean_g = float(updated.mean().item())
        else:
            val_sign = self.anat.mbon_valence[cols][None, None, :]
            el = np.clip(self.elig_kc_cpu, 0.0, 3.0)[:, :, None]
            active_kc = (el > elig_threshold).astype(np.float32)
            silent_kc = 1.0 - active_kc
            tr_delta = (self.lr_mult[:, None] * float(lr) * float(da_signal)) * active_kc.squeeze(-1)
            for ch in matched_channels:
                if side in (None, "L"):
                    self.calyx_trace_L_cpu[ch] = np.clip(self.calyx_trace_L_cpu[ch] + tr_delta, -1.35, 1.35)
                if side in (None, "R"):
                    self.calyx_trace_R_cpu[ch] = np.clip(self.calyx_trace_R_cpu[ch] + tr_delta, -1.35, 1.35)
            signed_drive = float(da_signal) * val_sign
            cur_sub = self.M_cpu[:, :, cols]
            mask_sub = self.anat.struct_mask_km[None, :, cols]

            hebb_delta = self.lr_mult[:, None, None] * float(lr) * el * signed_drive * active_kc
            ext_delta = float(extinction_rate) * abs(float(da_signal)) * (1.0 - cur_sub) * silent_kc
            updated = np.clip(cur_sub + (hebb_delta + ext_delta) * mask_sub, self.MIN_GAIN, self.MAX_GAIN)
            self.M_cpu[:, :, cols] = updated
            mean_g = float(updated.mean())

        return {"updated_columns": int(len(cols)), "mean_gain": round(mean_g, 4)}

    def apply_spiking_dan_modulation(self, dan_rates_hz: np.ndarray, lr: float = 0.015):
        """
        Direct connectome-driven DAN -> Compartment -> MBON plasticity using live spiking DAN rates
        (shape: (K, N_DAN) or (N_DAN,)): PPL1 spikes deliver negative valence, PAM spikes positive valence.
        """
        r_np = np.asarray(dan_rates_hz, dtype=np.float32)
        if r_np.ndim == 1:
            r_np = np.broadcast_to(r_np[None, :], (self.K, self.N_dan)).copy()
        if float(r_np.max()) < 1.0:
            return
        dan_sign = np.where(self.anat.dan_is_ppl1, -1.0, +1.0).astype(np.float32)
        signed_dan = (r_np / 100.0) * dan_sign[None, :]  # (K, N_DAN)
        mbon_da = signed_dan @ self.anat.C_dan_mbon      # (K, N_MBON)

        self._has_deviation = True
        if self.device == "cuda":
            da_t = torch.from_numpy(mbon_da).to("cuda")[:, None, :] * self.valence_gpu[None, None, :]
            el = torch.clamp(self.elig_kc_gpu, 0.0, 3.0)[:, :, None]
            cur = self.M_gpu.float()
            delta = self.lr_gpu * float(lr) * el * da_t * self.mask_gpu.float()[None, :, :]
            self.M_gpu.copy_(torch.clamp(cur + delta, self.MIN_GAIN, self.MAX_GAIN).to(self.dtype_gpu))
        else:
            da_c = mbon_da[:, None, :] * self.anat.mbon_valence[None, None, :]
            el = np.clip(self.elig_kc_cpu, 0.0, 3.0)[:, :, None]
            delta = self.lr_mult[:, None, None] * float(lr) * el * da_c * self.anat.struct_mask_km[None, :, :]
            self.M_cpu[:] = np.clip(self.M_cpu + delta, self.MIN_GAIN, self.MAX_GAIN)

    def consolidate_sleep_compartments(self, n_waves: int = 24, transfer_rate: float = 0.78) -> Dict[str, float]:
        """
        24-wave offline DPM/APL sleep replay across the KC x MBON matrix:
        transfers fast gamma (STM) & alpha'beta' (ITM) synaptic deviations (M - 1.0)
        exclusively onto anatomically matched ipsilateral alpha/beta (LTM) compartments
        within each task channel (preserving strict task-channel orthogonality!).
        """
        self._has_deviation = True
        total_ltm_delta = 0.0
        for ch_name, tiers in TASK_CHANNEL_COMPARTMENTS.items():
            src_cols = np.flatnonzero(self.mbon_column_mask(tiers["stm"] + tiers["itm"]))
            dst_cols = np.flatnonzero(self.mbon_column_mask(tiers["ltm"]))
            if len(src_cols) == 0 or len(dst_cols) == 0:
                continue

            for side_R in (False, True):
                s_c = src_cols[self.anat.mbon_side_is_R[src_cols] == side_R]
                d_c = dst_cols[self.anat.mbon_side_is_R[dst_cols] == side_R]
                if len(s_c) == 0 or len(d_c) == 0:
                    continue

                if self.device == "cuda":
                    dc_t = torch.from_numpy(d_c).long().to("cuda")
                    d_val = self.valence_gpu.index_select(0, dc_t)[None, None, :]
                    tr_side = (self.calyx_trace_R_gpu[ch_name] if side_R else self.calyx_trace_L_gpu[ch_name])[:, :, None]
                    cur_dst = self.M_gpu.index_select(2, dc_t).float()
                    target_dst = 1.0 + tr_side * d_val * 0.92
                    mask_d = self.mask_gpu.index_select(1, dc_t).float()[None, :, :] * (torch.abs(tr_side) > 1e-4).float()
                    new_dst = torch.clamp(cur_dst + transfer_rate * (target_dst - cur_dst) * mask_d, self.MIN_GAIN, self.MAX_GAIN)
                    total_ltm_delta += float(torch.abs(new_dst - cur_dst).mean().item())
                    self.M_gpu.index_copy_(2, dc_t, new_dst.to(self.dtype_gpu))
                else:
                    d_val = self.anat.mbon_valence[d_c][None, None, :]
                    tr_side = (self.calyx_trace_R_cpu[ch_name] if side_R else self.calyx_trace_L_cpu[ch_name])[:, :, None]
                    cur_dst = self.M_cpu[:, :, d_c]
                    target_dst = 1.0 + tr_side * d_val * 0.92
                    mask_d = self.anat.struct_mask_km[None, :, d_c] * (np.abs(tr_side) > 1e-4).astype(np.float32)
                    new_dst = np.clip(cur_dst + transfer_rate * (target_dst - cur_dst) * mask_d, self.MIN_GAIN, self.MAX_GAIN)
                    total_ltm_delta += float(np.abs(new_dst - cur_dst).mean())
                    self.M_cpu[:, :, d_c] = new_dst

        return {"waves": n_waves, "mean_ltm_engram_transfer": round(total_ltm_delta, 5)}

    def washout_stm_compartments(self, washout_fraction: float = 0.80) -> Dict[str, float]:
        """Washes out gamma (STM) and alpha'beta' (ITM) MBON columns while preserving alpha/beta (LTM) columns."""
        keep = float(1.0 - washout_fraction)
        stm_itm_cols = np.flatnonzero(np.isin(self.anat.mbon_tier, [0, 1]))
        ltm_cols = np.flatnonzero(self.anat.mbon_tier == 2)
        if self.device == "cuda":
            st_t = torch.from_numpy(stm_itm_cols).long().to("cuda")
            lt_t = torch.from_numpy(ltm_cols).long().to("cuda")
            cur_st = self.M_gpu.index_select(2, st_t).float()
            new_st = 1.0 + keep * (cur_st - 1.0)
            self.M_gpu.index_copy_(2, st_t, new_st.to(self.dtype_gpu))
            stm_dev = float(torch.abs(new_st - 1.0).mean().item())
            ltm_dev = float(torch.abs(self.M_gpu.index_select(2, lt_t).float() - 1.0).mean().item())
        else:
            cur_st = self.M_cpu[:, :, stm_itm_cols]
            self.M_cpu[:, :, stm_itm_cols] = 1.0 + keep * (cur_st - 1.0)
            stm_dev = float(np.abs(self.M_cpu[:, :, stm_itm_cols] - 1.0).mean())
            ltm_dev = float(np.abs(self.M_cpu[:, :, ltm_cols] - 1.0).mean())
        return {"residual_stm_deviation": round(stm_dev, 5), "preserved_ltm_deviation": round(ltm_dev, 5)}

    def compartment_valence_readout(self, kc_pattern: np.ndarray, compartments: Sequence[str]) -> Dict[str, float]:
        """
        Evaluates bilateral (Left vs Right) net Approach-minus-Avoidance MBON synaptic drive
        for a given KC activation pattern (shape: (N_KC,) or (K, N_KC)) restricted to `compartments`.
        """
        kc_p = np.asarray(kc_pattern, dtype=np.float32)
        if kc_p.ndim == 1:
            kc_p = np.broadcast_to(kc_p[None, :], (self.K, self.N_kc)).copy()

        cols_L = np.flatnonzero(self.mbon_column_mask(compartments, side="L"))
        cols_R = np.flatnonzero(self.mbon_column_mask(compartments, side="R"))

        if self.device == "cuda":
            kp_t = torch.from_numpy(kc_p).to("cuda")
            M_dev = (self.M_gpu.float() - 1.0) * self.Weff_gpu[None, :, :]
            drive_all = torch.bmm(kp_t.unsqueeze(1), M_dev).squeeze(1) * self.valence_gpu[None, :]  # (K, N_MBON)
            v_L = drive_all[:, torch.from_numpy(cols_L).long().to("cuda")].mean(dim=1).cpu().numpy() if len(cols_L) else np.zeros(self.K)
            v_R = drive_all[:, torch.from_numpy(cols_R).long().to("cuda")].mean(dim=1).cpu().numpy() if len(cols_R) else np.zeros(self.K)
        else:
            M_dev = (self.M_cpu - 1.0) * self.anat.W_eff_km[None, :, :]
            drive_all = np.einsum("ki,kim->km", kc_p, M_dev) * self.anat.mbon_valence[None, :]
            v_L = drive_all[:, cols_L].mean(axis=1) if len(cols_L) else np.zeros(self.K)
            v_R = drive_all[:, cols_R].mean(axis=1) if len(cols_R) else np.zeros(self.K)

        diff = v_R - v_L
        mu = float(np.mean(diff))
        sd = (float(np.std(diff)) + 1.0) if self.K > 1 else max(28.0, 0.35 * abs(mu) + 1.0)
        return {
            "mean_L": round(float(np.mean(v_L)), 4),
            "mean_R": round(float(np.mean(v_R)), 4),
            "valence_R_minus_L": round(mu, 4),
            "ensemble_snr": round(abs(mu) / sd, 3),
        }

    def snapshot_compartment_matrix(self) -> Dict[str, float]:
        """
        Returns mean active-engram synaptic modulation (1.0 + mean(|M_km - 1.0|) across plasticized
        structural KC->MBON contacts, or 1.0000 in untouched compartments) across all 15 MB compartments.
        """
        if self.device == "cuda":
            dev_t = torch.abs(self.M_gpu.float() - 1.0)
            act_m = (dev_t > 1e-4).float()
            num = (dev_t * act_m).sum(dim=(0, 1)).cpu().numpy()
            den = act_m.sum(dim=(0, 1)).cpu().numpy()
        else:
            dev_c = np.abs(self.M_cpu - 1.0)
            act_m = (dev_c > 1e-4).astype(np.float32)
            num = (dev_c * act_m).sum(axis=(0, 1))
            den = act_m.sum(axis=(0, 1))
        out = {}
        for comp in CANONICAL_COMPARTMENTS:
            cid = comp["id"]
            c_cols = np.flatnonzero(self.anat.mbon_comp_id == cid)
            if len(c_cols) > 0 and float(den[c_cols].sum()) > 0.5:
                mod = float(num[c_cols].sum() / den[c_cols].sum())
                out[comp["name"]] = round(1.0 + mod, 4)
            else:
                out[comp["name"]] = 1.0
        return out
