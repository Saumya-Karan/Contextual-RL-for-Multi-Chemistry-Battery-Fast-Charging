"""
LCO Electro-Thermal Surrogate Model
=====================================
Trains the one-step dynamics model for LCO chemistry:
    [SOC_t, T_t, I_t] --> [SOC_{t+1}, T_{t+1}]

Dataset : CALCE A123 18650 LCO cell (calce.umd.edu -> A123)
Cell IDs: A1-007, A1-008 (two cells per temperature)
Temps   : 0, 10, 25, 40 degC
Profiles: DST (Dynamic Stress Test), US06, FUDS — all in one xlsx per cell/temp

Folder structure:
  LCO electro thermal/
    A123_DST-US06-FUDS-0/   DST-US06-FUDS-0/   A1-007-DST-US06-FUDS-0-YYYYMMDD.xlsx
                                                 A1-008-DST-US06-FUDS-0-YYYYMMDD.xlsx
    A123_DST-US06-FUDS-10/  DST-US06-FUDS-10/  A1-007-...xlsx
    A123_DST-US06-FUDS-25/  DST-US06-FUDS-25/  A1-007-...xlsx
    A123_DST-US06-FUDS-40/  DST-US06-FUDS-40/  A1-007-...xlsx

File naming: A1-007-DST-US06-FUDS-[temp]-YYYYMMDD.xlsx
  [temp] is the ambient temperature in degC

Protocol structure within each xlsx (Step_Index):
  Steps 4+5   : CC-CV charge
  Step 8      : DST  drive cycle discharge  -> extract this
  Steps 12+13 : CC-CV charge
  Step 16     : US06 drive cycle discharge  -> extract this
  Steps 20+21 : CC-CV charge
  Step 24     : FUDS drive cycle discharge  -> extract this

Cell specs:
  C_nom = 1.1 Ah  (CALCE A123 18650 LCO)
  Voltage range: 2.0 - 3.7 V

Temperature acquisition:
  Temperature (C)_1 column is directly in the xlsx — no separate file needed.

SOC derivation:
  Cumulative Ah is reset per step in this dataset.
  SOC is derived per discharge segment:
    SOC(t) = 1.0 - (DisAh(t) - DisAh(t0) - ChgAh(t) + ChgAh(t0)) / C_nom
  where t0 is the start of each discharge step.
"""

import os
import re
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import openpyxl
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ══════════════════════════════════════════════════════════════════
# !! SET THIS PATH !!
# The root folder containing the four temperature subfolders
# ══════════════════════════════════════════════════════════════════
DATA_ROOT = r"C:\Users\saumy\OneDrive\Desktop\IN Pallavi ma'am\Datasets extracted\LCO electro thermal"

OUTPUT_DIR = r"lco_surrogate_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────
C_NOM        = 1.1          # Ah — CALCE A123 18650 LCO
TEMPS_USE    = [0, 10, 25, 40]   # all four available temperatures
DRIVE_STEPS  = {8: 'DST', 16: 'US06', 24: 'FUDS'}  # step -> profile name

TRAIN_SPLIT  = 0.70
VAL_SPLIT    = 0.15
TEST_SPLIT   = 0.15

BATCH_SIZE   = 4096
EPOCHS       = 200
LR           = 2e-3
PATIENCE     = 20

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True


# ══════════════════════════════════════════════════════════════════
# 1. HELPERS
# ══════════════════════════════════════════════════════════════════

def parse_temp_from_filename(fname):
    """
    Extract temperature from CALCE filename.
    A1-007-DST-US06-FUDS-10-20120815.xlsx  -> 10
    A1-007-DST-US06-FUDS-0-20120813.xlsx   -> 0
    A1-007-DST-US06-FUDS-25-20120827.xlsx  -> 25
    """
    fname = os.path.basename(fname)
    # Pattern: FUDS-[temp]-YYYYMMDD
    m = re.search(r'FUDS-(\d+)-\d{8}', fname)
    if m:
        return int(m.group(1))
    # Fallback: extract from parent folder name
    return None


def parse_temp_from_folder(folder):
    """Extract temperature from folder name like DST-US06-FUDS-10."""
    m = re.search(r'FUDS-(\d+)$', os.path.basename(folder))
    return int(m.group(1)) if m else None


def get_data_sheet(fpath):
    """Auto-detect the Arbin data sheet (Channel_X-XXX)."""
    try:
        wb = openpyxl.load_workbook(fpath, read_only=True)
        for sheet in wb.sheetnames:
            if sheet.startswith('Channel_') and sheet != 'Channel_Chart':
                return sheet
    except:
        pass
    return None


# ══════════════════════════════════════════════════════════════════
# 2. DATA LOADING
# ══════════════════════════════════════════════════════════════════

def load_calce_xlsx(fpath, temp_nominal):
    """
    Load one CALCE A123 xlsx file and extract all three drive cycle
    discharge segments (DST, US06, FUDS).

    Returns list of DataFrames, one per drive cycle segment.
    Each DataFrame has columns: [time_s, I, V, T_C, SOC, source]
    """
    sheet = get_data_sheet(fpath)
    if sheet is None:
        print(f'    WARNING: no data sheet in {os.path.basename(fpath)}')
        return []

    try:
        df = pd.read_excel(fpath, sheet_name=sheet)
    except Exception as e:
        print(f'    WARNING: cannot read {os.path.basename(fpath)}: {e}')
        return []

    # Required columns
    required = ['Test_Time(s)', 'Step_Index', 'Current(A)',
                 'Voltage(V)', 'Charge_Capacity(Ah)',
                 'Discharge_Capacity(Ah)', 'Temperature (C)_1']
    if not all(c in df.columns for c in required):
        print(f'    WARNING: missing columns in {os.path.basename(fpath)}')
        return []

    segments = []
    fname_base = os.path.basename(fpath).replace('.xlsx', '')

    for step_idx, profile_name in DRIVE_STEPS.items():
        seg = df[df['Step_Index'] == step_idx].copy().reset_index(drop=True)

        if len(seg) < 50:
            continue

        # Time from zero for this segment
        seg['time_s'] = seg['Test_Time(s)'] - seg['Test_Time(s)'].iloc[0]

        # SOC derivation within this segment
        # Each segment starts with cell at full charge (SOC=1.0)
        # SOC decreases as net discharge accumulates
        dis_start = seg['Discharge_Capacity(Ah)'].iloc[0]
        chg_start = seg['Charge_Capacity(Ah)'].iloc[0]
        dis_delta  = seg['Discharge_Capacity(Ah)'] - dis_start
        chg_delta  = seg['Charge_Capacity(Ah)'] - chg_start
        net_dis    = dis_delta - chg_delta   # net Ah discharged
        seg['SOC'] = (1.0 - net_dis / C_NOM).clip(0.0, 1.0)

        # Temperature: use measured column, fall back to nominal
        T = pd.to_numeric(seg['Temperature (C)_1'], errors='coerce')
        if T.isna().sum() > len(T) * 0.5:
            T = pd.Series(float(temp_nominal), index=seg.index)
        seg['T_C'] = T.fillna(float(temp_nominal))

        out = seg[['time_s', 'Current(A)', 'Voltage(V)',
                   'T_C', 'SOC']].copy()
        out.columns = ['time_s', 'I', 'V', 'T_C', 'SOC']
        out['source'] = f'{fname_base}_{profile_name}'

        segments.append(out)

    return segments


def resample_to_1hz(df):
    """Resample to uniform 1 Hz grid via linear interpolation."""
    if df is None or len(df) < 20:
        return None
    t_grid = np.arange(df['time_s'].iloc[0], df['time_s'].iloc[-1], 1.0)
    if len(t_grid) < 20:
        return None
    out = pd.DataFrame({'time_s': t_grid})
    for col in ['I', 'V', 'T_C', 'SOC']:
        out[col] = np.interp(t_grid, df['time_s'].values, df[col].values)
    out['source'] = df['source'].iloc[0]
    return out


def load_all_data(data_root):
    """
    Walk the data root, find all temperature subfolders,
    load all xlsx files from each.
    """
    print(f'  Data root: {data_root}')
    all_dfs = []

    # Find all subfolders containing xlsx files
    # Structure: data_root / A123_DST-US06-FUDS-10 / DST-US06-FUDS-10 / *.xlsx
    # OR:        data_root / A123_DST-US06-FUDS-10 / *.xlsx
    xlsx_files_found = []

    for root, dirs, files in os.walk(data_root):
        for f in files:
            if f.endswith('.xlsx') and 'FUDS' in f and not f.startswith('~'):
                xlsx_files_found.append(os.path.join(root, f))

    if not xlsx_files_found:
        print('  ERROR: No xlsx files found. Check DATA_ROOT.')
        return []

    # Group by temperature
    by_temp = {}
    for fpath in xlsx_files_found:
        # Try filename first
        temp = parse_temp_from_filename(fpath)
        if temp is None:
            # Try parent folder
            temp = parse_temp_from_folder(os.path.dirname(fpath))
        if temp is None:
            print(f'    SKIP (cannot parse temp): {os.path.basename(fpath)}')
            continue
        if temp not in TEMPS_USE:
            continue
        by_temp.setdefault(temp, []).append(fpath)

    for temp in sorted(by_temp.keys()):
        files = sorted(by_temp[temp])
        print(f'\n  Temperature: {temp} degC  ({len(files)} files)')
        for fpath in files:
            segs = load_calce_xlsx(fpath, temp_nominal=temp)
            for seg in segs:
                seg_1hz = resample_to_1hz(seg)
                if seg_1hz is None:
                    continue
                # Only keep dynamic segments (I must vary)
                if seg_1hz['I'].std() < 0.1:
                    continue
                all_dfs.append(seg_1hz)
                print(f'    + {seg_1hz["source"].iloc[0]}: '
                      f'n={len(seg_1hz):,}, '
                      f'SOC={seg_1hz["SOC"].min():.2f}-{seg_1hz["SOC"].max():.2f}, '
                      f'T={seg_1hz["T_C"].min():.1f}-{seg_1hz["T_C"].max():.1f}C, '
                      f'I={seg_1hz["I"].min():.2f}-{seg_1hz["I"].max():.2f}A')

    total = sum(len(d) for d in all_dfs)
    print(f'\n  --> {len(all_dfs)} segments, {total:,} total timesteps')
    return all_dfs


# ══════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════

def build_xy_pairs(dfs):
    """
    X: [SOC_t, T_t, I_t]
    Y: [dSOC, dT]  (one-step residuals)
    """
    X_list, Y_list = [], []
    for df in dfs:
        soc = df['SOC'].values
        T   = df['T_C'].values
        I   = df['I'].values

        d_soc = soc[1:] - soc[:-1]
        d_T   = T[1:]   - T[:-1]

        X = np.column_stack([soc[:-1], T[:-1], I[:-1]])
        Y = np.column_stack([d_soc, d_T])

        mask = (
            (np.abs(d_soc) < 0.05) &
            (np.abs(d_T)   < 2.0)  &
            (soc[:-1] >= 0.0) & (soc[:-1] <= 1.0)
        )
        X_list.append(X[mask])
        Y_list.append(Y[mask])

    return (np.vstack(X_list).astype(np.float32),
            np.vstack(Y_list).astype(np.float32))


def compute_normalisation(X, Y):
    return {
        'X_mean': X.mean(axis=0),
        'X_std' : X.std(axis=0)  + 1e-8,
        'Y_mean': Y.mean(axis=0),
        'Y_std' : Y.std(axis=0)  + 1e-8,
    }


def normalise(X, Y, stats):
    Xn = ((X - stats['X_mean']) / stats['X_std']).astype(np.float32)
    Yn = ((Y - stats['Y_mean']) / stats['Y_std']).astype(np.float32)
    return Xn, Yn


# ══════════════════════════════════════════════════════════════════
# 4. MODEL  (identical architecture to NMC and LFP surrogates)
# ══════════════════════════════════════════════════════════════════

class ElectroThermalMLP(nn.Module):
    def __init__(self, input_dim=3, hidden=64, output_dim=2, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),

            nn.Linear(hidden // 2, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════
# 5. TRAINING
# ══════════════════════════════════════════════════════════════════

def train_model(X_train, Y_train, X_val, Y_val, device):
    model     = ElectroThermalMLP().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimiser, patience=10, factor=0.5, verbose=False)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    val_ds   = TensorDataset(torch.tensor(X_val),   torch.tensor(Y_val))

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)

    best_val_loss = np.inf
    patience_ctr  = 0
    best_state    = None
    history       = {'train': [], 'val': []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(val_ds)

        history['train'].append(train_loss)
        history['val'].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1

        if epoch % 10 == 0:
            print(f'    Epoch {epoch:3d}: train={train_loss:.6f}  '
                  f'val={val_loss:.6f}  '
                  f'lr={optimiser.param_groups[0]["lr"]:.2e}')

        if patience_ctr >= PATIENCE:
            print(f'    Early stopping at epoch {epoch}')
            break

    model.load_state_dict(best_state)
    return model, history


# ══════════════════════════════════════════════════════════════════
# 6. EVALUATION
# ══════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, Y_test, stats, device):
    model.eval()
    with torch.no_grad():
        Y_pred_norm = model(torch.tensor(X_test).to(device)).cpu().numpy()

    Y_pred = Y_pred_norm * stats['Y_std'] + stats['Y_mean']
    Y_true = Y_test      * stats['Y_std'] + stats['Y_mean']

    soc_mae  = np.mean(np.abs(Y_pred[:, 0] - Y_true[:, 0]))
    soc_rmse = np.sqrt(np.mean((Y_pred[:, 0] - Y_true[:, 0]) ** 2))
    T_mae    = np.mean(np.abs(Y_pred[:, 1] - Y_true[:, 1]))
    T_rmse   = np.sqrt(np.mean((Y_pred[:, 1] - Y_true[:, 1]) ** 2))

    print(f'\n  Test set results ({len(X_test):,} samples):')
    print(f'  dSOC  -- MAE: {soc_mae:.6f}   RMSE: {soc_rmse:.6f}')
    print(f'  dT    -- MAE: {T_mae:.4f} C   RMSE: {T_rmse:.4f} C')
    print(f'\n  Absolute SOC accuracy : +/-{soc_mae*100:.4f}% per step')
    print(f'  Temperature accuracy  : +/-{T_mae:.4f} C per step')

    return {'soc_mae': soc_mae, 'soc_rmse': soc_rmse,
            'T_mae': T_mae, 'T_rmse': T_rmse, 'n_test': len(X_test)}


# ══════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('high')
    print(f'Device     : {device}')
    if device.type == 'cuda':
        print(f'GPU        : {torch.cuda.get_device_name(0)}')
        print(f'VRAM       : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
    print(f'C_nom      : {C_NOM} Ah')
    print(f'Temps      : {TEMPS_USE} degC')
    print(f'Profiles   : {list(DRIVE_STEPS.values())}')
    print()

    print('=' * 60)
    print('STEP 1: Loading data')
    print('=' * 60)

    all_dfs = load_all_data(DATA_ROOT)
    if not all_dfs:
        print('\nERROR: No data loaded. Check DATA_ROOT path.')
        return

    total = sum(len(d) for d in all_dfs)
    print(f'\nTotal: {len(all_dfs)} segments, {total:,} timesteps')

    print('\n' + '=' * 60)
    print('STEP 2: Building (X, Y) pairs')
    print('=' * 60)

    X, Y = build_xy_pairs(all_dfs)
    print(f'X shape : {X.shape}   (SOC_t, T_t, I_t)')
    print(f'Y shape : {Y.shape}   (dSOC, dT)')
    print(f'SOC range : {X[:,0].min():.3f} - {X[:,0].max():.3f}')
    print(f'T range   : {X[:,1].min():.1f} - {X[:,1].max():.1f} C')
    print(f'I range   : {X[:,2].min():.2f} - {X[:,2].max():.2f} A')

    n     = len(X)
    n_tr  = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)
    rng   = np.random.default_rng(SEED)
    idx   = rng.permutation(n)

    X_train, Y_train = X[idx[:n_tr]],            Y[idx[:n_tr]]
    X_val,   Y_val   = X[idx[n_tr:n_tr+n_val]],  Y[idx[n_tr:n_tr+n_val]]
    X_test,  Y_test  = X[idx[n_tr+n_val:]],       Y[idx[n_tr+n_val:]]
    print(f'\nSplit: train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}')

    stats          = compute_normalisation(X_train, Y_train)
    X_tr_n, Y_tr_n = normalise(X_train, Y_train, stats)
    X_v_n,  Y_v_n  = normalise(X_val,   Y_val,   stats)
    X_te_n, Y_te_n = normalise(X_test,  Y_test,  stats)

    print(f'\nNormalisation (SOC, T, I):')
    print(f'  Mean : {stats["X_mean"]}')
    print(f'  Std  : {stats["X_std"]}')

    print('\n' + '=' * 60)
    print('STEP 3: Training')
    print('=' * 60)
    print(f'Batch={BATCH_SIZE}  LR={LR}  MaxEpoch={EPOCHS}  Patience={PATIENCE}')
    print()

    model, history = train_model(X_tr_n, Y_tr_n, X_v_n, Y_v_n, device)

    print('\n' + '=' * 60)
    print('STEP 4: Evaluation')
    print('=' * 60)

    metrics = evaluate_model(model, X_te_n, Y_te_n, stats, device)

    torch.save(model.state_dict(),
               os.path.join(OUTPUT_DIR, 'lco_surrogate_model.pt'))
    with open(os.path.join(OUTPUT_DIR, 'lco_surrogate_stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)
    with open(os.path.join(OUTPUT_DIR, 'lco_surrogate_history.pkl'), 'wb') as f:
        pickle.dump({'history': history, 'metrics': metrics}, f)

    print(f'\nSaved to: {OUTPUT_DIR}/')

    print('\n' + '=' * 60)
    print('SUMMARY  -- fill into Table II of paper')
    print('=' * 60)
    print(f'  Chemistry        : LCO (CALCE A123 18650)')
    print(f'  Profiles         : DST, US06, FUDS')
    print(f'  Temperatures     : {TEMPS_USE} degC')
    print(f'  Training samples : {len(X_train):,}')
    print(f'  Test samples     : {len(X_test):,}')
    print(f'  dSOC MAE         : {metrics["soc_mae"]:.6f}  ({metrics["soc_mae"]*100:.4f}%)')
    print(f'  dSOC RMSE        : {metrics["soc_rmse"]:.6f}')
    print(f'  dT   MAE         : {metrics["T_mae"]:.4f} C')
    print(f'  dT   RMSE        : {metrics["T_rmse"]:.4f} C')


# ── Inference helper (used by RL pack simulator) ──────────────────

def load_lco_surrogate(output_dir):
    model = ElectroThermalMLP()
    model.load_state_dict(
        torch.load(os.path.join(output_dir, 'lco_surrogate_model.pt'),
                   map_location='cpu'))
    model.eval()
    with open(os.path.join(output_dir, 'lco_surrogate_stats.pkl'), 'rb') as f:
        stats = pickle.load(f)
    return model, stats


def predict_step(model, stats, soc_t, T_t, I_t):
    x  = np.array([[soc_t, T_t, I_t]], dtype=np.float32)
    xn = (x - stats['X_mean']) / stats['X_std']
    with torch.no_grad():
        yn = model(torch.tensor(xn)).numpy()
    delta    = yn * stats['Y_std'] + stats['Y_mean']
    soc_next = float(np.clip(soc_t + delta[0, 0], 0.0, 1.0))
    T_next   = float(T_t + delta[0, 1])
    return soc_next, T_next


if __name__ == '__main__':
    main()