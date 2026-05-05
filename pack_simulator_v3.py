"""
3×3 Li-ion Battery Pack Simulator — v3: configurable pack size
===============================================================
Key change from v2: pack layout (n_rows × n_cols) is now a parameter
that can be varied at episode reset, enabling generalization across
pack sizes within a single RL policy.

Physics justification for why size-generalization works:
  - Per-cell SOC dynamics are identical regardless of pack size
    (same surrogate, same I_cell range 0–5A)
  - State vector statistics (max, min, std) are size-independent by design
  - Only thermal coupling topology changes (avg neighbors: 3.0 for 2×2 → 5.25 for 4×4)
  - The 9th state element n_cells_norm = n_cells / N_CELLS_MAX informs the
    policy of the current topology without changing the state dimension

Action convention (CHANGED from v2):
  - v2: action → I_pack; I_cell = I_pack / 9  (broken for n_cells ≠ 9)
  - v3: action → I_cell; I_pack = I_cell × n_cells  (size-invariant)
  - The simulator clip is now: np.clip(I_cell, 0.0, 5.0) before scaling

Supported pack layouts (configurable at runtime):
  2×2  (4 cells)   3×2  (6 cells)   3×3  (9 cells)
  4×3  (12 cells)  4×4  (16 cells)

Pack layout (row-major, 0-indexed):
  [0][1]...[n_cols-1]
  [n_cols]...
"""

import os, sys, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn

# ══════════════════════════════════════════════════════════════════
# MLP ARCHITECTURES (unchanged)
# ══════════════════════════════════════════════════════════════════

class ElectroThermalMLP(nn.Module):
    """All three surrogates: 3->64->64->32->2"""
    def __init__(self, input_dim=3, hidden=64, output_dim=2, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, output_dim),
        )
    def forward(self, x): return self.net(x)


class DegradationMLP(nn.Module):
    """NMC degradation: 2->128->128->64->32->1"""
    def __init__(self, input_dim=2, dropout=0.02):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 128),       nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64,  32),        nn.ReLU(),
            nn.Linear(32,  1),         nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x)


class DegradationMLPSmall(nn.Module):
    """LFP/LCO degradation: 2->16->8->1"""
    def __init__(self, input_dim=2, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, 8),         nn.ReLU(),
            nn.Linear(8,  1),         nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x)


# ══════════════════════════════════════════════════════════════════
# CHEMISTRY PARAMETERS
# ══════════════════════════════════════════════════════════════════

CHEMISTRY_PARAMS = {
    'NMC': {'R_int': 0.050, 'C_nom': 3.0,  'C_nom_mAh': 3000, 'T_max': 45.0,
            'SOC_min': 0.05, 'SOC_max': 0.95, 'theta': 0,
            'surrogate_cls': ElectroThermalMLP,
            'degradation_cls': DegradationMLP},
    'LFP': {'R_int': 0.025, 'C_nom': 2.5,  'C_nom_mAh': 2500, 'T_max': 45.0,
            'SOC_min': 0.05, 'SOC_max': 0.95, 'theta': 2,
            'surrogate_cls': ElectroThermalMLP,
            'degradation_cls': DegradationMLPSmall},
    'LCO': {'R_int': 0.060, 'C_nom': 1.1,  'C_nom_mAh': 1100, 'T_max': 40.0,
            'SOC_min': 0.05, 'SOC_max': 0.95, 'theta': 1,
            'surrogate_cls': ElectroThermalMLP,
            'degradation_cls': DegradationMLPSmall},
}

# Thermal parameters (per-cell, physics-based)
C_TH    = 50.0   # J/K  thermal capacitance
K_IJ    = 0.5    # W/K  cell-to-cell coupling coefficient
H_BASE  = 5.0    # W/(m²K)  base convective coefficient
H_COEFF = 200.0  # W/(m²K per kg/s)
A_CELL  = 0.002  # m²  cell surface area exposed to coolant
DT      = 1.0    # s   timestep

# Pack size normalization constant (max expected pack size)
N_CELLS_MAX = 16  # 4×4 grid; used to normalize n_cells context variable


def _build_adjacency_and_kmat(n_rows: int, n_cols: int):
    """
    Build adjacency dict and symmetric conduction matrix for an
    n_rows × n_cols rectangular cell grid.

    Cell index: i = row * n_cols + col  (row-major)
    Neighbours: all 8-connected cells within bounds (Moore neighbourhood)
    """
    n = n_rows * n_cols
    adj = {}
    for i in range(n):
        r, c = divmod(i, n_cols)
        nb = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < n_rows and 0 <= c2 < n_cols:
                    nb.append(r2 * n_cols + c2)
        adj[i] = nb

    K = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in adj[i]:
            K[i, j]  = K_IJ
            K[i, i] -= K_IJ
    return adj, K


def _load_model(cls, wpath, spath):
    m = cls()
    m.load_state_dict(torch.load(wpath, map_location='cpu'))
    m.eval()
    with open(spath, 'rb') as f:
        stats = pickle.load(f)
    return m, stats


# ══════════════════════════════════════════════════════════════════
# PACK SIMULATOR — configurable size
# ══════════════════════════════════════════════════════════════════

class PackSimulator:
    """
    Battery pack simulator with configurable layout (n_rows × n_cols).

    Action interface (v3):
        I_cell  [A] — per-cell charging current, clipped to [0, 5] A
        I_pack  = I_cell × n_cells  (computed internally, not exposed)
        mdot    [kg/s] — coolant mass flow rate, clipped to [0, 0.05]

    State vector (9-dimensional, size-independent):
        [SOC_max, SOC_min, T_max, T_min, SOH_avg,
         sigma_SOC, sigma_T, theta_norm, n_cells_norm]
    """

    # Supported rectangular layouts sampled during training
    SUPPORTED_LAYOUTS = [(2, 2), (2, 3), (3, 3), (3, 4), (4, 4)]

    def __init__(self, chemistry, surrogate_dir, degradation_dir,
                 safety_dir, T_ambient=25.0, SOC_init=0.2,
                 SOH_init=1.0, cell_variation=0.01,
                 n_rows=3, n_cols=3):

        assert chemistry in CHEMISTRY_PARAMS
        self.chemistry = chemistry
        self.params    = CHEMISTRY_PARAMS[chemistry]
        self.T_ambient = T_ambient
        self.T_cool    = T_ambient
        self.cell_var  = cell_variation
        cl             = chemistry.lower()

        # ── Load surrogate ──────────────────────────────────────
        surr_cls = self.params['surrogate_cls']
        self.surrogate, self.surr_stats = _load_model(
            surr_cls,
            os.path.join(surrogate_dir, f'{cl}_surrogate_model.pt'),
            os.path.join(surrogate_dir, f'{cl}_surrogate_stats.pkl'))

        self._Xmean = torch.tensor(
            self.surr_stats['X_mean'], dtype=torch.float32)
        self._Xstd  = torch.tensor(
            self.surr_stats['X_std'],  dtype=torch.float32)
        self._Ymean = self.surr_stats['Y_mean']
        self._Ystd  = self.surr_stats['Y_std']

        # ── Load degradation ────────────────────────────────────
        deg_cls = self.params['degradation_cls']
        self.deg_model, self.deg_stats = _load_model(
            deg_cls,
            os.path.join(degradation_dir, f'{cl}_degradation_model.pt'),
            os.path.join(degradation_dir, f'{cl}_degradation_stats.pkl'))

        # ── Load safety classifier ──────────────────────────────
        self.safety_clf = self.safety_scaler = None
        try:
            import joblib
            clf_data = joblib.load(
                os.path.join(safety_dir, 'safety_classifier_model.pkl'))
            if isinstance(clf_data, dict):
                self.safety_clf    = clf_data.get('model', clf_data)
                self.safety_scaler = clf_data.get('scaler', None)
            else:
                self.safety_clf = clf_data
        except Exception:
            pass

        # ── Pack layout (configurable) ──────────────────────────
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_cells = n_rows * n_cols
        self.ADJACENCY, self.K_MAT = _build_adjacency_and_kmat(n_rows, n_cols)

        self._init_state(SOC_init, SOH_init)
        self.step_count = 0

    # ── Layout management ─────────────────────────────────────────

    def set_layout(self, n_rows: int, n_cols: int):
        """Reconfigure pack layout and rebuild thermal matrix.
        Call before reset() when changing pack size mid-training.
        """
        self.n_rows  = n_rows
        self.n_cols  = n_cols
        self.n_cells = n_rows * n_cols
        self.ADJACENCY, self.K_MAT = _build_adjacency_and_kmat(n_rows, n_cols)

    # ── State initialisation ──────────────────────────────────────

    def _init_state(self, SOC_init, SOH_init):
        rng = np.random.default_rng()
        self.SOC = np.clip(
            SOC_init + rng.normal(0, self.cell_var, self.n_cells),
            self.params['SOC_min'], self.params['SOC_max']
        ).astype(np.float32)
        self.T   = np.full(self.n_cells, self.T_ambient, dtype=np.float32)
        self.SOH = np.full(self.n_cells, SOH_init,       dtype=np.float32)

    def reset(self, SOC_init=0.2, SOH_init=1.0, T_ambient=None,
              n_rows=None, n_cols=None):
        """
        Reset the simulator, optionally changing pack layout.

        Parameters
        ----------
        n_rows, n_cols : int, optional
            New pack dimensions.  If provided, the thermal matrix is
            rebuilt before initialising cell states.
        """
        if T_ambient is not None:
            self.T_ambient = T_ambient
            self.T_cool    = T_ambient

        if n_rows is not None and n_cols is not None:
            self.set_layout(n_rows, n_cols)

        self._init_state(SOC_init, SOH_init)
        self.step_count = 0
        return self._state()

    # ── Simulation step ───────────────────────────────────────────

    def step(self, I_cell, mdot):
        """
        Advance simulation by one timestep.

        Parameters
        ----------
        I_cell : float
            Per-cell charging current in amperes.
            Clipped to [0, 5] A; I_pack = I_cell × n_cells.
        mdot : float
            Coolant mass flow rate [kg/s], clipped to [0, 0.05].

        Returns
        -------
        state : np.ndarray (9,)
        done  : bool
        info  : dict
        """
        I_cell = float(np.clip(I_cell, 0.0, 5.0))
        mdot   = float(np.clip(mdot,   0.0, 0.05))
        I_pack = I_cell * self.n_cells  # total pack current

        # ── Batched surrogate: all n_cells in one forward pass ────
        # LFP convention: negative current = charging (Arbin/A123 dataset)
        I_signed = -I_cell if self.chemistry == 'LFP' else I_cell
        X = np.column_stack([
            self.SOC,
            self.T,
            np.full(self.n_cells, I_signed)
        ]).astype(np.float32)

        Xt = (torch.from_numpy(X) - self._Xmean) / self._Xstd
        with torch.no_grad():
            Yn = self.surrogate(Xt).numpy()

        deltas  = Yn * self._Ystd + self._Ymean
        new_SOC = np.clip(
            self.SOC + deltas[:, 0],
            self.params['SOC_min'], self.params['SOC_max']
        ).astype(np.float32)
        new_T   = (self.T + deltas[:, 1]).astype(np.float32)

        # ── Pack-level thermal effects (joule heat is already in surrogate) ─
        # Add only conduction between cells and coolant removal.
        dT_cond = self.K_MAT @ new_T               # (n_cells,)
        h_conv  = H_BASE + H_COEFF * mdot
        Q_conv  = h_conv * A_CELL * (new_T - self.T_cool)
        new_T   = (new_T + (dT_cond - Q_conv) * DT / C_TH).astype(np.float32)

        # ── SOH update every 100 steps ────────────────────────────
        new_SOH = self.SOH.copy()
        if self.step_count % 100 == 0 and self.step_count > 0:
            cf  = min(self.step_count / 600.0, 1.0)
            x   = np.array([[cf, float(new_T.mean())]], dtype=np.float32)
            xn  = (x - self.deg_stats['X_mean']) / self.deg_stats['X_std']
            with torch.no_grad():
                soh_new = self.deg_model(torch.tensor(xn)).item()
            delta   = max(0.0, float(self.SOH.mean()) - soh_new)
            new_SOH = np.clip(self.SOH - delta, 0.5, 1.0).astype(np.float32)

        # ── Safety (soft signal, not hard termination) ────────────
        T_max = float(new_T.max())
        P_run = 0.0
        if self.safety_clf is not None:
            try:
                soc_n   = float(new_SOC.mean())
                cap_n   = self.params['C_nom_mAh'] / 15000.0
                chem_id = self.params['theta']
                is_lco  = 1 if self.chemistry == 'LCO' else 0
                is_lfp  = 1 if self.chemistry == 'LFP' else 0
                is_nmc  = 1 if self.chemistry == 'NMC' else 0
                feats   = [[soc_n, cap_n, chem_id, is_lco, is_lfp, is_nmc,
                            soc_n**2, soc_n * cap_n]]
                if self.safety_scaler is not None:
                    feats = self.safety_scaler.transform(feats)
                P_run = float(self.safety_clf.predict_proba(feats)[0, 1])
            except Exception:
                P_run = float(T_max > self.params['T_max'] * 0.9)
        else:
            P_run = float(T_max > self.params['T_max'] * 0.9)

        self.SOC = new_SOC
        self.T   = new_T
        self.SOH = new_SOH
        self.step_count += 1

        # ── Termination ───────────────────────────────────────────
        done = False; done_reason = ''
        if float(self.SOC.min()) >= 0.80:
            done = True; done_reason = 'success'
        elif T_max >= self.params['T_max']:
            done = True; done_reason = 'thermal_abort'
        elif self.step_count >= 2400:
            done = True; done_reason = 'timeout'
        # P_run: available for soft reward penalty; not used as hard cutoff

        info = {
            'SOC': self.SOC.copy(), 'T': self.T.copy(),
            'SOH': self.SOH.copy(), 'P_run': P_run,
            'I_cell': I_cell, 'I_pack': I_pack, 'mdot': mdot,
            'n_cells': self.n_cells, 'n_rows': self.n_rows,
            'n_cols': self.n_cols, 'done_reason': done_reason,
            'step': self.step_count,
        }
        return self._state(), done, info

    # ── Observation ───────────────────────────────────────────────

    def _state(self):
        """
        9-dimensional state vector, size-independent.

        Elements:
          0  SOC_max        — max cell SOC
          1  SOC_min        — min cell SOC (charging target: ≥ 0.80)
          2  T_max          — max cell temperature
          3  T_min          — min cell temperature
          4  SOH_avg        — mean state of health
          5  sigma_SOC      — SOC spread across cells
          6  sigma_T        — temperature spread across cells
          7  theta_norm     — chemistry context (0/1/2 → normalised)
          8  n_cells_norm   — pack size context (n_cells / N_CELLS_MAX)
        """
        theta_norm = float(self.params['theta']) / 2.0      # ∈ {0, 0.5, 1}
        n_norm     = float(self.n_cells) / float(N_CELLS_MAX)  # ∈ [0.25, 1]
        return np.array([
            self.SOC.max(), self.SOC.min(),
            self.T.max(),   self.T.min(),
            self.SOH.mean(),
            self.SOC.std(), self.T.std(),
            theta_norm,
            n_norm,
        ], dtype=np.float32)

    def get_pack_summary(self):
        return {
            'SOC_mean': float(self.SOC.mean()),
            'SOC_min':  float(self.SOC.min()),
            'SOC_max':  float(self.SOC.max()),
            'SOC_std':  float(self.SOC.std()),
            'T_mean':   float(self.T.mean()),
            'T_min':    float(self.T.min()),
            'T_max':    float(self.T.max()),
            'T_std':    float(self.T.std()),
            'SOH_mean': float(self.SOH.mean()),
            'step':     self.step_count,
            'chemistry': self.chemistry,
            'layout':   f'{self.n_rows}×{self.n_cols}',
            'n_cells':  self.n_cells,
        }

    def __repr__(self):
        s = self.get_pack_summary()
        return (f'PackSim({self.chemistry} {s["layout"]}) '
                f'step={s["step"]} '
                f'SOC={s["SOC_min"]:.3f}-{s["SOC_max"]:.3f} '
                f'T={s["T_min"]:.1f}-{s["T_max"]:.1f}C')
