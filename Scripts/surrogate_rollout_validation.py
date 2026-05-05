"""
Multi-Step Rollout Validation for All Three Electro-Thermal Surrogates
=======================================================================
Validates NMC, LFP, and LCO surrogates in autoregressive mode:
  - Feed model predictions back as inputs for N steps
  - Compare rolled-out SOC and T trajectories against ground truth
  - Report cumulative SOC error and temperature drift over the episode

Why rollout validation matters:
  One-step MAE looks at individual predictions in isolation.
  In the RL simulator the surrogate is called thousands of times
  in sequence — small per-step errors can accumulate and cause
  the simulated trajectory to diverge from reality.
  This script tests exactly that scenario.

Metrics reported:
  - MAE at step 100, 300, 500 (cumulative error growth)
  - Max absolute SOC error over the full rollout
  - Max absolute T error over the full rollout
  - Drift ratio: final_error / one_step_MAE (ideally close to 1)

Usage:
  Set the three OUTPUT_DIR paths below (same as used in training scripts).
  Set DATA paths to point to one held-out test file per chemistry.
  Run: python surrogate_rollout_validation.py
"""

import os
import re
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scipy.io as sio
import openpyxl
import matplotlib
matplotlib.use('Agg')   # non-interactive backend
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# ══════════════════════════════════════════════════════════════════
# !! SET THESE PATHS !!
# ══════════════════════════════════════════════════════════════════

# Output dirs from each surrogate training script
NMC_MODEL_DIR = r"nmc_surrogate_outputs"
LFP_MODEL_DIR = r"lfp_surrogate_outputs"
LCO_MODEL_DIR = r"lco_surrogate_outputs"

# One held-out test file per chemistry
# Use a file NOT seen during training for a fair test
NMC_TEST_FILE = r"C:\Users\saumy\OneDrive\Desktop\IN Pallavi ma'am\Datasets extracted\NMC electro thermal\LG 18650HG2.zip\LG 18650HG2 Li-ion Battery Data a_015790\LG_HG2_Original_Dataset_McMasterU_e3a009\25degC\10-29-18_10.08 551_US06_25degC_LGHG2"

LFP_TEST_FILE = r"C:\Users\saumy\OneDrive\Desktop\IN Pallavi ma'am\Datasets extracted\LFP electro thermal\p8kf893yv3-1\Miscellaneous\A002_UDDS_P35.mat"

LCO_TEST_FILE = r"C:\Users\saumy\OneDrive\Desktop\IN Pallavi ma'am\Datasets extracted\LCO electro thermal\A123_DST-US06-FUDS-25\DST-US06-FUDS-25\A1-007-DST-US06-FUDS-25-20120827.xlsx"

ROLLOUT_STEPS = 500   # number of autoregressive steps to simulate
OUTPUT_DIR    = r"rollout_validation_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Chemistry-specific nominal capacities ─────────────────────────
C_NOM = {'NMC': 3.0, 'LFP': 2.5, 'LCO': 1.1}


# ══════════════════════════════════════════════════════════════════
# 1. MODEL ARCHITECTURE (must match each training script)
# ══════════════════════════════════════════════════════════════════

class ElectroThermalMLP(nn.Module):
    """NMC and LFP architecture: 3->64->64->32->2"""
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
    def forward(self, x):
        return self.net(x)


def load_surrogate(model_dir, model_file='nmc_surrogate_model.pt'):
    """Load a trained surrogate model and its normalisation stats."""
    model = ElectroThermalMLP()
    # Detect which chemistry from filename
    if 'lco' in model_file:
        # LCO uses same architecture in v1
        model = ElectroThermalMLP()
    model.load_state_dict(
        torch.load(os.path.join(model_dir, model_file), map_location='cpu'))
    model.eval()

    stats_file = model_file.replace('_model.pt', '_stats.pkl')
    with open(os.path.join(model_dir, stats_file), 'rb') as f:
        stats = pickle.load(f)
    return model, stats


# ══════════════════════════════════════════════════════════════════
# 2. DATA LOADERS (one per chemistry)
# ══════════════════════════════════════════════════════════════════

def load_nmc_test_sequence(fpath):
    """
    Load NMC test sequence from LG HG2 .mat file.
    Returns arrays: time_s, SOC, T, I at 1 Hz.
    """
    mat  = sio.loadmat(fpath)
    meas = mat['meas'][0, 0]
    t    = meas['Time'].flatten().astype(float)
    I    = meas['Current'].flatten().astype(float)
    T    = meas['Battery_Temp_degC'].flatten().astype(float)
    Ah   = meas['Ah'].flatten().astype(float)

    # Resample to 1 Hz
    t_grid = np.arange(t[0], t[-1], 1.0)
    I_r = np.interp(t_grid, t, I)
    T_r = np.interp(t_grid, t, T)
    Ah_r = np.interp(t_grid, t, Ah)

    # SOC from Ah (starts at 1.0 for discharge files)
    soc_0 = 1.0 if I_r[:5].mean() <= 0 else 0.1
    SOC_r = (soc_0 + Ah_r / C_NOM['NMC']).clip(0, 1)

    return t_grid - t_grid[0], SOC_r, T_r, I_r


def load_lfp_test_sequence(fpath):
    """
    Load LFP test sequence from A002_UDDS .mat file (flat Data struct).
    Returns arrays: time_s, SOC, T, I at 1 Hz.
    """
    mat = sio.loadmat(fpath)
    d   = mat['Data'][0, 0]
    t   = d['time'].flatten().astype(float)
    I   = d['current'].flatten().astype(float)
    T_f = 'Ts1' if 'Ts1' in d.dtype.names else 'Ts'
    T   = d[T_f].flatten().astype(float)
    chg = d['chgAh'].flatten().astype(float)
    dis = d['disAh'].flatten().astype(float)

    t_grid = np.arange(t[0], t[-1], 1.0)
    I_r  = np.interp(t_grid, t, I)
    T_r  = np.interp(t_grid, t, T)
    chg_r = np.interp(t_grid, t, chg)
    dis_r = np.interp(t_grid, t, dis)

    net_ah = chg_r - dis_r
    SOC_r  = (1.0 + net_ah / C_NOM['LFP']).clip(0, 1)

    return t_grid - t_grid[0], SOC_r, T_r, I_r


def load_lco_test_sequence(fpath):
    """
    Load LCO test sequence from CALCE xlsx file.
    Extracts DST segment (step 8) as the test trajectory.
    Returns arrays: time_s, SOC, T, I at 1 Hz.
    """
    wb = openpyxl.load_workbook(fpath, read_only=True)
    sheet = None
    for s in wb.sheetnames:
        if s.startswith('Channel_') and s != 'Channel_Chart':
            sheet = s
            break
    if sheet is None:
        raise ValueError(f'No data sheet in {fpath}')

    df = pd.read_excel(fpath, sheet_name=sheet)
    seg = df[df['Step_Index'] == 8].copy().reset_index(drop=True)

    t   = seg['Test_Time(s)'].values.astype(float)
    I   = seg['Current(A)'].values.astype(float)
    T   = seg['Temperature (C)_1'].values.astype(float)
    dis = seg['Discharge_Capacity(Ah)'].values.astype(float)
    chg = seg['Charge_Capacity(Ah)'].values.astype(float)

    t_grid = np.arange(t[0], t[-1], 1.0)
    I_r  = np.interp(t_grid, t, I)
    T_r  = np.interp(t_grid, t, T)
    dis_r = np.interp(t_grid, t, dis)
    chg_r = np.interp(t_grid, t, chg)

    net_dis = (dis_r - dis_r[0]) - (chg_r - chg_r[0])
    SOC_r   = (1.0 - net_dis / C_NOM['LCO']).clip(0, 1)

    return t_grid - t_grid[0], SOC_r, T_r, I_r


# ══════════════════════════════════════════════════════════════════
# 3. ROLLOUT ENGINE
# ══════════════════════════════════════════════════════════════════

def predict_one_step(model, stats, soc_t, T_t, I_t):
    """Single-step prediction, returns (soc_next, T_next)."""
    x  = np.array([[soc_t, T_t, I_t]], dtype=np.float32)
    xn = (x - stats['X_mean']) / stats['X_std']
    with torch.no_grad():
        yn = model(torch.tensor(xn)).numpy()
    delta    = yn * stats['Y_std'] + stats['Y_mean']
    soc_next = float(np.clip(soc_t + delta[0, 0], 0.0, 1.0))
    T_next   = float(T_t + delta[0, 1])
    return soc_next, T_next


def run_rollout(model, stats, SOC_true, T_true, I_true, n_steps):
    """
    Autoregressive rollout for n_steps.
    At each step: use predicted (SOC, T) as next input.
    Current I is taken from ground truth (RL agent controls this).

    Returns:
      SOC_pred, T_pred  : predicted trajectories (length n_steps+1)
      SOC_true_clip     : ground truth clipped to same length
      T_true_clip       : ground truth clipped to same length
    """
    n_steps = min(n_steps, len(SOC_true) - 1)

    SOC_pred = np.zeros(n_steps + 1)
    T_pred   = np.zeros(n_steps + 1)

    # Start from ground truth initial conditions
    SOC_pred[0] = SOC_true[0]
    T_pred[0]   = T_true[0]

    for k in range(n_steps):
        soc_next, T_next = predict_one_step(
            model, stats,
            soc_t=SOC_pred[k],
            T_t  =T_pred[k],
            I_t  =I_true[k]    # ground truth current (agent action)
        )
        SOC_pred[k+1] = soc_next
        T_pred[k+1]   = T_next

    return (SOC_pred,
            T_pred,
            SOC_true[:n_steps+1],
            T_true[:n_steps+1])


def compute_rollout_metrics(SOC_pred, T_pred, SOC_true, T_true):
    """Compute rollout accuracy metrics at multiple horizons."""
    n = len(SOC_pred) - 1
    horizons = [h for h in [100, 300, 500] if h <= n]

    metrics = {}
    for h in horizons:
        soc_err = np.abs(SOC_pred[:h+1] - SOC_true[:h+1])
        T_err   = np.abs(T_pred[:h+1]   - T_true[:h+1])
        metrics[h] = {
            'soc_mae': soc_err.mean(),
            'soc_max': soc_err.max(),
            'T_mae':   T_err.mean(),
            'T_max':   T_err.max(),
        }

    # Full rollout
    soc_err_all = np.abs(SOC_pred - SOC_true)
    T_err_all   = np.abs(T_pred   - T_true)
    metrics['full'] = {
        'soc_mae': soc_err_all.mean(),
        'soc_max': soc_err_all.max(),
        'T_mae':   T_err_all.mean(),
        'T_max':   T_err_all.max(),
    }

    return metrics


# ══════════════════════════════════════════════════════════════════
# 4. PLOTTING
# ══════════════════════════════════════════════════════════════════

def plot_rollout(chemistry, SOC_pred, T_pred, SOC_true, T_true,
                 save_path):
    """Plot predicted vs true SOC and T trajectories."""
    steps = np.arange(len(SOC_pred))

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f'{chemistry} Surrogate — {len(SOC_pred)-1}-Step Rollout Validation',
                 fontsize=13, fontweight='bold')

    # SOC
    ax = axes[0]
    ax.plot(steps, SOC_true, 'k-',  lw=1.5, label='Ground truth', alpha=0.8)
    ax.plot(steps, SOC_pred, 'r--', lw=1.5, label='Model rollout')
    ax.fill_between(steps,
                    SOC_true - np.abs(SOC_pred - SOC_true),
                    SOC_true + np.abs(SOC_pred - SOC_true),
                    alpha=0.15, color='red', label='Error envelope')
    ax.set_ylabel('SOC', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Temperature
    ax = axes[1]
    ax.plot(steps, T_true, 'k-',  lw=1.5, label='Ground truth', alpha=0.8)
    ax.plot(steps, T_pred, 'b--', lw=1.5, label='Model rollout')
    ax.fill_between(steps,
                    T_true - np.abs(T_pred - T_true),
                    T_true + np.abs(T_pred - T_true),
                    alpha=0.15, color='blue', label='Error envelope')
    ax.set_ylabel('Temperature (°C)', fontsize=11)
    ax.set_xlabel('Timestep (s)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Plot saved: {save_path}')


# ══════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════

def validate_one_chemistry(chemistry, model_dir, model_file,
                            load_fn, test_file):
    print(f'\n{"="*60}')
    print(f'  {chemistry} SURROGATE — ROLLOUT VALIDATION')
    print(f'{"="*60}')

    # Load model
    model, stats = load_surrogate(model_dir, model_file)
    model.eval()
    print(f'  Model loaded from: {model_dir}')

    # Load test sequence
    try:
        t, SOC_true, T_true, I_true = load_fn(test_file)
    except Exception as e:
        print(f'  ERROR loading test file: {e}')
        print(f'  Test file: {test_file}')
        return None

    print(f'  Test file: {os.path.basename(test_file)}')
    print(f'  Sequence length: {len(t)} steps')
    print(f'  SOC range: {SOC_true.min():.3f} - {SOC_true.max():.3f}')
    print(f'  T range  : {T_true.min():.1f} - {T_true.max():.1f} C')
    print(f'  I range  : {I_true.min():.3f} - {I_true.max():.3f} A')

    n_steps = min(ROLLOUT_STEPS, len(t) - 1)
    print(f'  Rolling out {n_steps} steps...')

    # Run rollout
    SOC_pred, T_pred, SOC_gt, T_gt = run_rollout(
        model, stats, SOC_true, T_true, I_true, n_steps)

    # Metrics
    metrics = compute_rollout_metrics(SOC_pred, T_pred, SOC_gt, T_gt)

    print(f'\n  Rollout accuracy (cumulative error over steps):')
    print(f'  {"Horizon":<10} {"SOC MAE":<14} {"SOC Max":<14} '
          f'{"T MAE (C)":<14} {"T Max (C)":<12}')
    print(f'  {"-"*62}')
    for h, m in metrics.items():
        label = f'{h} steps' if isinstance(h, int) else 'full rollout'
        print(f'  {label:<10} {m["soc_mae"]:.6f}      {m["soc_max"]:.6f}      '
              f'{m["T_mae"]:.4f}         {m["T_max"]:.4f}')

    # Plot
    plot_path = os.path.join(OUTPUT_DIR,
                             f'{chemistry.lower()}_rollout.png')
    plot_rollout(chemistry, SOC_pred, T_pred, SOC_gt, T_gt, plot_path)

    return metrics


def main():
    print('Multi-Step Rollout Validation — All Three Surrogates')
    print(f'Rollout length: {ROLLOUT_STEPS} steps')

    results = {}

    results['NMC'] = validate_one_chemistry(
        chemistry  = 'NMC',
        model_dir  = NMC_MODEL_DIR,
        model_file = 'nmc_surrogate_model.pt',
        load_fn    = load_nmc_test_sequence,
        test_file  = NMC_TEST_FILE,
    )

    results['LFP'] = validate_one_chemistry(
        chemistry  = 'LFP',
        model_dir  = LFP_MODEL_DIR,
        model_file = 'lfp_surrogate_model.pt',
        load_fn    = load_lfp_test_sequence,
        test_file  = LFP_TEST_FILE,
    )

    results['LCO'] = validate_one_chemistry(
        chemistry  = 'LCO',
        model_dir  = LCO_MODEL_DIR,
        model_file = 'lco_surrogate_model.pt',
        load_fn    = load_lco_test_sequence,
        test_file  = LCO_TEST_FILE,
    )

    # Summary table
    print(f'\n{"="*60}')
    print('ROLLOUT VALIDATION SUMMARY (full rollout metrics)')
    print(f'{"="*60}')
    print(f'{"Chemistry":<12} {"SOC MAE":<16} {"SOC Max":<16} '
          f'{"T MAE (C)":<14} {"T Max (C)"}')
    print('-' * 62)
    for chem, m in results.items():
        if m is None:
            print(f'{chem:<12} ERROR - check test file path')
            continue
        fm = m['full']
        print(f'{chem:<12} {fm["soc_mae"]:.6f}         '
              f'{fm["soc_max"]:.6f}         '
              f'{fm["T_mae"]:.4f}          '
              f'{fm["T_max"]:.4f}')

    print(f'\nPlots saved to: {OUTPUT_DIR}/')
    print()
    print('NOTE FOR PAPER:')
    print('  If SOC Max error < 0.05 over 500 steps -> surrogate is')
    print('  suitable for RL simulation (episode length ~600 steps).')
    print('  If T Max error < 2.0 C -> thermal safety signal is reliable.')

    # Save results
    with open(os.path.join(OUTPUT_DIR, 'rollout_metrics.pkl'), 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main()