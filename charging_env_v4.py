"""
Charging Environment — v3: multi-chemistry + multi-pack-size
============================================================
Extends v2 by randomizing pack layout (n_rows × n_cols) each
episode in addition to chemistry θ.

Context variable extension:
  v2: θ ∈ {NMC, LCO, LFP}
  v3: (θ, n_cells) ∈ {NMC, LCO, LFP} × {4, 6, 9, 12, 16}

Action interface (CHANGED):
  BatteryPackEnv._scale() now expects 2D action [I_cell_norm, mdot_norm]
  where I_cell_norm ∈ [-1,1] → I_cell ∈ [I_CELL_MIN, I_CELL_MAX]
  I_pack is derived internally: I_pack = I_cell × n_cells

State dimension: 9 (was 8)
  [SOC_max, SOC_min, T_max, T_min, SOH_avg, σ_SOC, σ_T, θ_norm, n_cells_norm]
"""

import os, sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from pack_simulator_v4 import PackSimulator, CHEMISTRY_PARAMS, N_CELLS_MAX

BASE = SCRIPT_DIR

DEFAULT_SURROGATE_DIRS = {
    'NMC': os.path.join(BASE, 'nmc_surrogate_outputs'),
    'LFP': os.path.join(BASE, 'lfp_surrogate_outputs'),
    'LCO': os.path.join(BASE, 'lco_surrogate_outputs'),
}
DEFAULT_DEGRADATION_DIRS = {
    'NMC': os.path.join(BASE, 'nmc_degradation_outputs'),
    'LFP': os.path.join(BASE, 'lfp_degradation_outputs'),
    'LCO': os.path.join(BASE, 'lco_degradation_outputs'),
}
DEFAULT_SAFETY_DIR = BASE

# ── Pack sizes to train on ────────────────────────────────────────
# Each tuple is (n_rows, n_cols).  All are supported by the simulator.
# The 3×3 layout is included to maintain continuity with v2 training.
PACK_LAYOUTS = [
    (2, 2),   #  4 cells
    (2, 3),   #  6 cells
    (3, 3),   #  9 cells  ← original
    (3, 4),   # 12 cells
    (4, 4),   # 16 cells
]

# ── Per-cell current limits (shared across all pack sizes) ────────
I_CELL_MIN = 0.05   # A  (minimum, keeps some current flowing)
I_CELL_MAX = 5.00   # A  (~2C for LFP, ~1.67C for NMC, ~4.5C for LCO)

MDOT_MIN = 0.0
MDOT_MAX = 0.05

SOC_INIT  = 0.2
MAX_STEPS = 2400
CHEMISTRIES = ['NMC', 'LFP', 'LCO']
OBS_DIM   = 11   # +2 vs v3: soc_cc and dsoc_cc_dt for LFP observability
ACT_DIM   = 2


class BatteryPackEnv:
    """
    Environment wrapper that randomizes both chemistry and pack layout
    at each episode reset.

    Action convention (v3):
        a[0]: I_cell_norm ∈ [-1, 1]  →  I_cell ∈ [I_CELL_MIN, I_CELL_MAX]
        a[1]: mdot_norm  ∈ [-1, 1]   →  mdot   ∈ [MDOT_MIN,   MDOT_MAX]
    The simulator receives I_cell (not I_pack) and scales internally.
    """

    def __init__(self,
                 surrogate_dirs=None,
                 degradation_dirs=None,
                 safety_dir=None,
                 T_ambient=25.0,
                 chemistry=None,
                 pack_layout=None,
                 cell_variation=0.01):

        self.surrogate_dirs   = surrogate_dirs   or DEFAULT_SURROGATE_DIRS
        self.degradation_dirs = degradation_dirs or DEFAULT_DEGRADATION_DIRS
        self.safety_dir       = safety_dir       or DEFAULT_SAFETY_DIR
        self.T_ambient        = T_ambient
        self.fixed_chemistry  = chemistry
        self.fixed_layout     = pack_layout      # (n_rows, n_cols) or None
        self.cell_variation   = cell_variation

        self.obs_dim           = OBS_DIM
        self.act_dim           = ACT_DIM
        self.observation_space = _Space(OBS_DIM)
        self.action_space      = _Box(ACT_DIM)

        # One simulator per chemistry (layout is changed at reset via set_layout)
        self._simulators = {}
        for chem in CHEMISTRIES:
            try:
                self._simulators[chem] = PackSimulator(
                    chemistry       = chem,
                    surrogate_dir   = self.surrogate_dirs[chem],
                    degradation_dir = self.degradation_dirs[chem],
                    safety_dir      = self.safety_dir,
                    T_ambient       = T_ambient,
                    cell_variation  = cell_variation,
                    n_rows=3, n_cols=3,   # default; overridden at reset
                )
                print(f'  [ENV] Loaded {chem}')
            except Exception as e:
                print(f'  [ENV] WARNING: {chem} failed: {e}')

        self._available = list(self._simulators.keys())
        if not self._available:
            raise RuntimeError('No simulators loaded.')

        print(f'  [ENV] Available chemistries : {self._available}')
        print(f'  [ENV] Pack layouts          : {PACK_LAYOUTS}')
        print(f'  [ENV] I_cell range          : [{I_CELL_MIN}, {I_CELL_MAX}] A (per cell)')
        print(f'  [ENV] Obs dim               : {OBS_DIM}  '
              f'(+1 for n_cells_norm vs v2)')

        self.sim        = None
        self.chemistry  = None
        self.layout     = None
        self.step_count = 0
        self._rng       = np.random.default_rng(42)

    # ── Action scaling ────────────────────────────────────────────

    def _scale(self, a):
        """Map action ∈ [-1,1]² to (I_cell [A], mdot [kg/s])."""
        I_cell = (float(a[0]) + 1) / 2 * (I_CELL_MAX - I_CELL_MIN) + I_CELL_MIN
        mdot   = (float(a[1]) + 1) / 2 * (MDOT_MAX   - MDOT_MIN)   + MDOT_MIN
        return float(I_cell), float(mdot)

    # ── Episode management ────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Sample chemistry — LFP oversampled 50% to compensate for its
        # harder observability (flat OCV) and shorter episode length.
        # More LFP gradient updates close the gap with NMC/LCO faster.
        if self.fixed_chemistry in self._simulators:
            self.chemistry = self.fixed_chemistry
        else:
            avail = self._available
            if len(avail) == 3:
                # NMC=25%, LFP=50%, LCO=25%
                probs = [0.25 if c != 'LFP' else 0.50 for c in avail]
            else:
                probs = [1.0/len(avail)] * len(avail)
            self.chemistry = self._rng.choice(avail, p=probs)
        self.sim = self._simulators[self.chemistry]

        # Sample pack layout
        if self.fixed_layout is not None:
            n_rows, n_cols = self.fixed_layout
        else:
            n_rows, n_cols = PACK_LAYOUTS[
                self._rng.integers(0, len(PACK_LAYOUTS))
            ]
        self.layout = (n_rows, n_cols)

        # LFP is more sensitive to ambient temperature variance:
        # its shorter charging window (~1080 vs ~1520 steps) means less
        # recovery margin when starting warm. Use tighter ±2°C for LFP.
        t_range = 2.0 if self.chemistry == 'LFP' else 5.0
        T_amb = self.T_ambient + float(self._rng.uniform(-t_range, t_range))
        obs   = self.sim.reset(
            SOC_init   = SOC_INIT,
            SOH_init   = 1.0,
            T_ambient  = T_amb,
            n_rows     = n_rows,
            n_cols     = n_cols,
        )
        self.step_count = 0
        return obs, {
            'chemistry': self.chemistry,
            'layout':    self.layout,
            'n_cells':   self.sim.n_cells,
        }

    def step(self, action):
        I_cell, mdot       = self._scale(action)
        curr, done, info   = self.sim.step(I_cell, mdot)

        # Reward: SOC progress (per-step dense) + terminal bonuses
        # Using SOC_min change — same formula as v2 for continuity
        soc_avg  = (curr[0] + curr[1]) / 2
        reward   = 2.0 * max(0.0, soc_avg - getattr(self, '_prev_soc_avg', soc_avg))
        reward  -= 0.1 * float(curr[5])   # SOC imbalance penalty

        reason = info['done_reason']
        if reason == 'success':
            reward += 50.0
        elif reason == 'thermal_abort':
            reward -= 20.0

        self._prev_soc_avg = soc_avg
        self.step_count   += 1
        truncated = self.step_count >= MAX_STEPS and not done
        return curr, reward, done, truncated, info

    def render(self):
        if self.sim:
            print(self.sim)

    def close(self):
        pass


class _Space:
    def __init__(self, dim): self.shape = (dim,)

class _Box(_Space):
    def __init__(self, dim):
        super().__init__(dim)
        self.low  = -np.ones(dim, dtype=np.float32)
        self.high =  np.ones(dim, dtype=np.float32)
    def sample(self):
        return np.random.uniform(-1, 1, self.shape[0]).astype(np.float32)
