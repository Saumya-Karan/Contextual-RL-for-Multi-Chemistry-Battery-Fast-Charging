"""
LFP Electro-Thermal Surrogate Model
=====================================
Trains the one-step dynamics model for LFP chemistry:
    [SOC_t, T_t, I_t] --> [SOC_{t+1}, T_{t+1}]

Dataset : A123 26650 LFP cell (Mendeley p8kf893yv3/1)
Sources :
  DYNdata/A002_DYN/ -- dynamic drive cycles, 6 temperatures
                       struct: DYNData.script1/2/3
                       temperature: from matching _S1.xlsx (Aux_Temperature_1)

  CCCV/             -- CC-CV charge profiles 1C/2C/3C/4C at ~25 degC
                       struct: Data (flat) with fields time/current/voltage/chgAh/disAh/Ts
                       temperature: Ts field directly in .mat

  Miscellaneous/    -- UDDS drive cycles at P25 and P35
                       struct: Data (flat) with fields time/current/voltage/chgAh/disAh/Ts or Ts1
                       temperature: Ts/Ts1 field directly in .mat
                       NOTE: A002_CCCV_1C_2 uses DYNData struct (same as DYN folder)

C_nom = 2.5 Ah (A123 26650 LFP)
"""

import os
import re
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scipy.io as sio
import pickle
import openpyxl

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ══════════════════════════════════════════════════════════════════
# !! SET THIS PATH !!
# ══════════════════════════════════════════════════════════════════
BASE = r"C:\Users\saumy\OneDrive\Desktop\IN Pallavi ma'am\Datasets extracted\LFP electro thermal\p8kf893yv3-1"

DATA_DYN   = os.path.join(BASE, r"DYNdata\A002_DYN")
DATA_CCCV  = os.path.join(BASE, r"CCCV")
DATA_MISC  = os.path.join(BASE, r"Miscellaneous")
OUTPUT_DIR = r"lfp_surrogate_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────
C_NOM       = 2.5
TEMPS_USE   = [-5, 5, 15, 25, 35, 45]
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15

BATCH_SIZE  = 4096
EPOCHS      = 200
LR          = 2e-3
PATIENCE    = 20

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True


# ══════════════════════════════════════════════════════════════════
# 1. HELPERS
# ══════════════════════════════════════════════════════════════════

def parse_temp_from_filename(fname):
    """Parse temperature from A002_DYN_10_P25 -> +25, A002_UDDS_N05 -> -5."""
    m = re.search(r'_([PN])(\d+)(?:\.mat|_S\d|$)', os.path.basename(fname))
    if not m:
        return None
    return (1.0 if m.group(1) == 'P' else -1.0) * float(m.group(2))


def get_xlsx_sheet(fpath):
    """Auto-detect Arbin data sheet (Channel_X-XXX, not Channel_Chart)."""
    try:
        wb = openpyxl.load_workbook(fpath, read_only=True)
        for sheet in wb.sheetnames:
            if sheet.startswith('Channel_') and sheet != 'Channel_Chart':
                return sheet
    except:
        pass
    return None


def resample_to_1hz(df, time_col='time_s'):
    """Resample DataFrame to uniform 1 Hz via linear interpolation."""
    if df is None or len(df) < 20:
        return None
    t_grid = np.arange(df[time_col].iloc[0], df[time_col].iloc[-1], 1.0)
    if len(t_grid) < 20:
        return None
    out = pd.DataFrame({time_col: t_grid})
    for col in df.columns:
        if col != time_col:
            out[col] = np.interp(t_grid,
                                 df[time_col].values,
                                 df[col].values)
    return out


def compute_soc(df, soc_initial):
    """SOC(t) = soc_initial + (chgAh - disAh)/C_nom, clipped [0,1]."""
    df = df.copy()
    df['SOC'] = (soc_initial + (df['chgAh'] - df['disAh']) / C_NOM).clip(0.0, 1.0)
    return df


# ══════════════════════════════════════════════════════════════════
# 2. LOADERS FOR EACH STRUCT TYPE
# ══════════════════════════════════════════════════════════════════

def load_flat_data_struct(mat_path):
    """
    Load a .mat file with a flat 'Data' struct:
      Data.time, Data.current, Data.voltage, Data.chgAh, Data.disAh, Data.Ts (or Ts1)
    Used by: CCCV/*.mat and Miscellaneous/A002_UDDS_*.mat
    Returns DataFrame [time_s, I, V, chgAh, disAh, T_C] at original sampling.
    """
    try:
        mat = sio.loadmat(mat_path)
        if 'Data' not in mat:
            return None
        d = mat['Data'][0, 0]

        t   = d['time'].flatten().astype(float)
        I   = d['current'].flatten().astype(float)
        V   = d['voltage'].flatten().astype(float)
        chg = d['chgAh'].flatten().astype(float)
        dis = d['disAh'].flatten().astype(float)

        # Temperature field: 'Ts1' in some files, 'Ts' in others
        T_field = 'Ts1' if 'Ts1' in d.dtype.names else 'Ts'
        T = d[T_field].flatten().astype(float)

        df = pd.DataFrame({'time_s': t, 'I': I, 'V': V,
                           'chgAh': chg, 'disAh': dis, 'T_C': T})
        df['time_s'] -= df['time_s'].iloc[0]
        return df
    except Exception as e:
        print(f'    WARNING flat-struct {os.path.basename(mat_path)}: {e}')
        return None


def load_dyndata_script(mat_path, script_name):
    """
    Load one script from a DYNData struct .mat file.
    Used by: DYNdata/*.mat and CCCV/A002_CCCV_1C_2.mat
    Returns DataFrame [time_s, I, V, chgAh, disAh] — no temperature.
    """
    try:
        mat = sio.loadmat(mat_path)
        d   = mat['DYNData'][0, 0]
        s   = d[script_name][0, 0]
        df  = pd.DataFrame({
            'time_s': s['time'].flatten().astype(float),
            'I':      s['current'].flatten().astype(float),
            'V':      s['voltage'].flatten().astype(float),
            'chgAh':  s['chgAh'].flatten().astype(float),
            'disAh':  s['disAh'].flatten().astype(float),
        })
        df['time_s'] -= df['time_s'].iloc[0]
        return df
    except Exception as e:
        print(f'    WARNING DYNData {os.path.basename(mat_path)}[{script_name}]: {e}')
        return None


def load_xlsx_temperature(xlsx_path):
    """
    Load temperature from Arbin xlsx file (Aux_Temperature_1 column).
    Returns DataFrame [time_s, T_C] with time starting at 0.
    """
    sheet = get_xlsx_sheet(xlsx_path)
    if sheet is None:
        return None
    try:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        if 'Aux_Temperature_1(C)' not in df.columns:
            return None
        out = df[['Test_Time(s)', 'Aux_Temperature_1(C)']].copy()
        out.columns = ['time_s', 'T_C']
        out['time_s'] = pd.to_numeric(out['time_s'], errors='coerce')
        out['T_C']    = pd.to_numeric(out['T_C'],    errors='coerce')
        out = out.dropna()
        out['time_s'] -= out['time_s'].iloc[0]
        return out.reset_index(drop=True)
    except Exception as e:
        print(f'    WARNING xlsx {os.path.basename(xlsx_path)}: {e}')
        return None


def merge_dyndata_with_xlsx(mat_df, xlsx_df, t_nominal=None):
    """
    Merge DYNData dynamics (no temperature) with xlsx temperature.
    Falls back to t_nominal if xlsx not available.
    Returns 1Hz merged DataFrame with T_C column.
    """
    if mat_df is None:
        return None

    mat_1hz = resample_to_1hz(mat_df)
    if mat_1hz is None:
        return None

    if xlsx_df is not None and len(xlsx_df) >= 5:
        t0 = max(mat_1hz['time_s'].iloc[0],  xlsx_df['time_s'].iloc[0])
        t1 = min(mat_1hz['time_s'].iloc[-1], xlsx_df['time_s'].iloc[-1])
        if t1 - t0 < 60:
            # Time windows don't overlap well — use full mat range
            t0, t1 = mat_1hz['time_s'].iloc[0], mat_1hz['time_s'].iloc[-1]
        mask = (mat_1hz['time_s'] >= t0) & (mat_1hz['time_s'] <= t1)
        out  = mat_1hz[mask].copy().reset_index(drop=True)
        out['T_C'] = np.interp(out['time_s'].values,
                               xlsx_df['time_s'].values,
                               xlsx_df['T_C'].values)
    elif t_nominal is not None:
        out = mat_1hz.copy()
        out['T_C'] = float(t_nominal)
    else:
        return None

    return out


# ══════════════════════════════════════════════════════════════════
# 3. FOLDER LOADERS
# ══════════════════════════════════════════════════════════════════

def load_dyn_folder(folder):
    """
    DYN folder: DYNData struct + temperature from _S1.xlsx.
    script1 = dynamic discharge (SOC_0=1.0)
    script3 = slow charge back  (SOC_0=0.0)
    script2 = OCV characterisation — skipped (I~0)
    """
    print(f'\n  [DYN] {folder}')
    dfs = []
    n_skip = 0

    all_mat = sorted(glob.glob(os.path.join(folder, '*.mat')))
    all_mat += sorted([f for f in glob.glob(os.path.join(folder, '*'))
                       if os.path.isfile(f) and '.' not in os.path.basename(f)])

    for mat_path in all_mat:
        temp = parse_temp_from_filename(mat_path)
        if temp is None or temp not in TEMPS_USE:
            n_skip += 1
            continue

        base    = re.sub(r'\.mat$', '', mat_path)
        xlsx_s1 = base + '_S1.xlsx'

        for sname, soc_0 in [('script1', 1.0), ('script3', 0.0)]:
            mat_df  = load_dyndata_script(mat_path, sname)
            xlsx_df = load_xlsx_temperature(xlsx_s1) if os.path.exists(xlsx_s1) else None
            merged  = merge_dyndata_with_xlsx(mat_df, xlsx_df, t_nominal=temp)
            if merged is None:
                continue
            merged = compute_soc(merged, soc_initial=soc_0)
            if merged['I'].std() < 0.1:
                continue
            merged['source'] = f'{os.path.basename(mat_path)}_{sname}'
            dfs.append(merged)
            print(f'    + {merged["source"].iloc[0]}: '
                  f'n={len(merged):,}, '
                  f'SOC={merged["SOC"].min():.2f}-{merged["SOC"].max():.2f}, '
                  f'T={merged["T_C"].min():.1f}-{merged["T_C"].max():.1f}C')

    print(f'    --> {len(dfs)} files ({n_skip} skipped — temp out of range)')
    return dfs


def load_cccv_folder(folder):
    """
    CCCV folder: two struct types coexist.
      - A002_CCCV_1C/2C/3C/4C   : flat 'Data' struct WITH temperature (Ts field)
      - A002_CCCV_1C_2           : DYNData struct, NO temperature
    All tests done at ~25 degC (chamber controlled).
    CCCV charges from empty: SOC_0 = 0.05
    """
    print(f'\n  [CCCV] {folder}')
    dfs = []

    all_files = sorted([f for f in glob.glob(os.path.join(folder, '*'))
                        if os.path.isfile(f)])

    for fpath in all_files:
        fname = os.path.basename(fpath)

        try:
            mat  = sio.loadmat(fpath)
            keys = [k for k in mat.keys() if not k.startswith('__')]
            if not keys:
                continue

            if 'Data' in keys:
                # Flat struct — has real temperature in Ts field
                df = load_flat_data_struct(fpath)
                if df is None:
                    continue
                df_1hz = resample_to_1hz(df)
                if df_1hz is None:
                    continue
                df_1hz = compute_soc(df_1hz, soc_initial=0.05)
                if df_1hz['I'].std() < 0.05:
                    continue
                df_1hz['source'] = fname
                dfs.append(df_1hz)
                print(f'    + {fname}: '
                      f'n={len(df_1hz):,}, '
                      f'SOC={df_1hz["SOC"].min():.2f}-{df_1hz["SOC"].max():.2f}, '
                      f'I={df_1hz["I"].min():.2f}-{df_1hz["I"].max():.2f}A, '
                      f'T={df_1hz["T_C"].min():.1f}-{df_1hz["T_C"].max():.1f}C')

            elif 'DYNData' in keys:
                # DYNData struct — no temperature, use nominal 25C
                for sname, soc_0 in [('script1', 0.05), ('script3', 0.0)]:
                    mat_df = load_dyndata_script(fpath, sname)
                    merged = merge_dyndata_with_xlsx(mat_df, None, t_nominal=25.0)
                    if merged is None:
                        continue
                    merged = compute_soc(merged, soc_initial=soc_0)
                    if merged['I'].std() < 0.05:
                        continue
                    merged['source'] = f'{fname}_{sname}'
                    dfs.append(merged)
                    print(f'    + {merged["source"].iloc[0]}: '
                          f'n={len(merged):,}, '
                          f'SOC={merged["SOC"].min():.2f}-{merged["SOC"].max():.2f}, '
                          f'T=25.0C (nominal)')

        except Exception as e:
            print(f'    SKIP {fname}: {e}')

    print(f'    --> {len(dfs)} files from CCCV')
    return dfs


def load_misc_folder(folder):
    """
    Miscellaneous folder:
      A002_UDDS_P25, A002_UDDS_P35 : flat 'Data' struct WITH temperature (Ts/Ts1)
      A002_PeriodicPulseData        : SKIP
    """
    print(f'\n  [Misc] {folder}')
    dfs = []

    all_files = sorted([f for f in glob.glob(os.path.join(folder, '*'))
                        if os.path.isfile(f)])

    for fpath in all_files:
        fname = os.path.basename(fpath)

        if 'Periodic' in fname or 'Pulse' in fname:
            print(f'    SKIP {fname} (periodic pulse — not useful for surrogate)')
            continue

        temp = parse_temp_from_filename(fpath)
        if temp is None or temp not in TEMPS_USE:
            print(f'    SKIP {fname} (temp {temp} not in TEMPS_USE)')
            continue

        try:
            mat  = sio.loadmat(fpath)
            keys = [k for k in mat.keys() if not k.startswith('__')]

            if 'Data' not in keys:
                print(f'    SKIP {fname} (unknown struct: {keys})')
                continue

            # Flat Data struct with measured temperature
            df = load_flat_data_struct(fpath)
            if df is None:
                continue
            df_1hz = resample_to_1hz(df)
            if df_1hz is None:
                continue
            df_1hz = compute_soc(df_1hz, soc_initial=1.0)
            if df_1hz['I'].std() < 0.1:
                continue
            df_1hz['source'] = fname
            dfs.append(df_1hz)
            print(f'    + {fname}: '
                  f'n={len(df_1hz):,}, '
                  f'SOC={df_1hz["SOC"].min():.2f}-{df_1hz["SOC"].max():.2f}, '
                  f'I={df_1hz["I"].min():.2f}-{df_1hz["I"].max():.2f}A, '
                  f'T={df_1hz["T_C"].min():.1f}-{df_1hz["T_C"].max():.1f}C')

        except Exception as e:
            print(f'    SKIP {fname}: {e}')

    print(f'    --> {len(dfs)} files from Misc')
    return dfs


# ══════════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING
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
# 5. MODEL  (identical architecture to NMC surrogate)
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
# 6. TRAINING
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
# 7. EVALUATION
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
# 8. MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('high')
    print(f'Device     : {device}')
    if device.type == 'cuda':
        print(f'GPU        : {torch.cuda.get_device_name(0)}')
        print(f'VRAM       : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
    print(f'C_nom      : {C_NOM} Ah  |  Temps: {TEMPS_USE} degC')
    print()

    print('=' * 60)
    print('STEP 1: Loading data')
    print('=' * 60)

    dfs_dyn  = load_dyn_folder(DATA_DYN)
    dfs_cccv = load_cccv_folder(DATA_CCCV)
    dfs_misc = load_misc_folder(DATA_MISC)
    all_dfs  = dfs_dyn + dfs_cccv + dfs_misc

    if not all_dfs:
        print('\nERROR: No data loaded. Check the three paths at top of script.')
        return

    total = sum(len(d) for d in all_dfs)
    print(f'\n{"="*60}')
    print(f'Total: {len(all_dfs)} files, {total:,} timesteps')
    print(f'  DYN  : {len(dfs_dyn)} files,  {sum(len(d) for d in dfs_dyn):,} steps')
    print(f'  CCCV : {len(dfs_cccv)} files,  {sum(len(d) for d in dfs_cccv):,} steps')
    print(f'  Misc : {len(dfs_misc)} files,  {sum(len(d) for d in dfs_misc):,} steps')

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
               os.path.join(OUTPUT_DIR, 'lfp_surrogate_model.pt'))
    with open(os.path.join(OUTPUT_DIR, 'lfp_surrogate_stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)
    with open(os.path.join(OUTPUT_DIR, 'lfp_surrogate_history.pkl'), 'wb') as f:
        pickle.dump({'history': history, 'metrics': metrics}, f)

    print(f'\nSaved to: {OUTPUT_DIR}/')

    print('\n' + '=' * 60)
    print('SUMMARY  -- fill into Table II of paper')
    print('=' * 60)
    print(f'  Chemistry        : LFP (A123 26650)')
    print(f'  Sources          : DYN + CCCV + UDDS')
    print(f'  Temperatures     : {TEMPS_USE} degC')
    print(f'  Training samples : {len(X_train):,}')
    print(f'  Test samples     : {len(X_test):,}')
    print(f'  dSOC MAE         : {metrics["soc_mae"]:.6f}  ({metrics["soc_mae"]*100:.4f}%)')
    print(f'  dSOC RMSE        : {metrics["soc_rmse"]:.6f}')
    print(f'  dT   MAE         : {metrics["T_mae"]:.4f} C')
    print(f'  dT   RMSE        : {metrics["T_rmse"]:.4f} C')


# ── Inference helper ─────────────────────────────────────────────

def load_lfp_surrogate(output_dir):
    model = ElectroThermalMLP()
    model.load_state_dict(
        torch.load(os.path.join(output_dir, 'lfp_surrogate_model.pt'),
                   map_location='cpu'))
    model.eval()
    with open(os.path.join(output_dir, 'lfp_surrogate_stats.pkl'), 'rb') as f:
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