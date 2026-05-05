"""
NMC Electro-Thermal Surrogate Model
=====================================
Trains the one-step dynamics model for NMC chemistry:
    [SOC_t, T_t, I_t] --> [SOC_{t+1}, T_{t+1}]

Dataset: LG 18650HG2 (NMC), McMaster University
Source : Mendeley cp3473x7xv/3
Temps  : 0degC, 10degC, 25degC, 40degC

Architecture: MLP (3 -> 64 -> 64 -> 32 -> 2)
    Predicts DELTA_SOC and DELTA_T (residual targets).

Optimised for: 16GB VRAM GPU (e.g. RTX 4080 / 3090)

Usage:
    Set DATA_ROOT to the LG_HG2_Original_Dataset_McMasterU folder.
    Run: python nmc_surrogate_train.py
"""

import os
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
import pickle
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ══════════════════════════════════════════════════════════════════
# !! SET THIS PATH !!
# ══════════════════════════════════════════════════════════════════
DATA_ROOT = r"C:\Users\saumy\OneDrive\Desktop\IN Pallavi ma'am\Datasets extracted\NMC electro thermal\LG 18650HG2.zip\LG 18650HG2 Li-ion Battery Data a_015790\LG_HG2_Original_Dataset_McMasterU_e3a009"

OUTPUT_DIR = r"nmc_surrogate_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────
C_NOM       = 3.0
TEMPS_USE   = ['0degC', '10degC', '25degC', '40degC']
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15

BATCH_SIZE  = 4096       # 16GB VRAM — safe headroom
EPOCHS      = 200
LR          = 2e-3       # scaled with batch size
PATIENCE    = 20

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True   # auto-tune CUDA kernels


# ══════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════

def load_mat_file(fpath):
    """Load a .mat file. Returns DataFrame: [time_s, V, I, T, Ah]"""
    mat  = sio.loadmat(fpath)
    meas = mat['meas'][0, 0]
    t    = meas['Time'].flatten().astype(float)
    V    = meas['Voltage'].flatten().astype(float)
    I    = meas['Current'].flatten().astype(float)
    T    = meas['Battery_Temp_degC'].flatten().astype(float)
    Ah   = meas['Ah'].flatten().astype(float)
    return pd.DataFrame({'time_s': t, 'V': V, 'I': I, 'T': T, 'Ah': Ah})


def load_csv_file(fpath):
    """Load a .csv file with 28-row metadata header. Returns DataFrame: [time_s, V, I, T, Ah]"""
    try:
        raw = pd.read_csv(fpath, skiprows=28, low_memory=False, encoding='latin1')
        raw = raw.iloc[1:]   # drop unit row
        raw.columns = ['TimeStamp', 'Step', 'Status', 'ProgTime', 'StepTime',
                       'Cycle', 'CycleLevel', 'Procedure',
                       'V', 'I', 'T', 'Ah', 'Wh', 'Cnt', '_drop']
        for col in ['V', 'I', 'T', 'Ah']:
            raw[col] = pd.to_numeric(raw[col], errors='coerce')
        raw = raw.dropna(subset=['V', 'I', 'T'])

        def time_to_sec(s):
            try:
                parts = str(s).split(':')
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            except:
                return np.nan

        raw['time_s'] = raw['ProgTime'].apply(time_to_sec)
        raw = raw.dropna(subset=['time_s'])
        return raw[['time_s', 'V', 'I', 'T', 'Ah']].reset_index(drop=True)
    except Exception as e:
        print(f'    WARNING: Could not parse {fpath}: {e}')
        return None


def resample_to_1hz(df):
    """Resample time series to uniform 1 Hz via linear interpolation."""
    if df is None or len(df) < 10:
        return None
    t_uniform = np.arange(df['time_s'].iloc[0], df['time_s'].iloc[-1], 1.0)
    if len(t_uniform) < 10:
        return None
    df_out = pd.DataFrame({'time_s': t_uniform})
    for col in ['V', 'I', 'T', 'Ah']:
        df_out[col] = np.interp(t_uniform, df['time_s'].values, df[col].values)
    return df_out


def compute_soc(df, soc_initial=None):
    """
    Derive SOC from cumulative Ah column.
    SOC(t) = SOC_0 + Ah(t) / C_nom
    """
    if soc_initial is None:
        soc_initial = 0.1 if df['I'].iloc[:5].mean() >= 0 else 1.0
    df = df.copy()
    df['SOC'] = (soc_initial + df['Ah'] / C_NOM).clip(0.0, 1.0)
    return df


def load_temperature_folder(temp_folder, temp_label):
    """
    Load all .mat and .csv files from one temperature subfolder.
    Returns list of DataFrames resampled to 1 Hz with SOC column.
    """
    print(f'  Loading {temp_label}...')
    dfs = []

    mat_files = sorted(glob.glob(os.path.join(temp_folder, '*.mat')))
    csv_files = sorted(glob.glob(os.path.join(temp_folder, '*.csv')))

    # Files with no extension (MATLAB v5 .mat without .mat suffix)
    no_ext = [f for f in glob.glob(os.path.join(temp_folder, '*'))
              if not f.endswith('.mat') and not f.endswith('.csv')
              and os.path.isfile(f)]

    for fpath in mat_files + no_ext:
        try:
            df = load_mat_file(fpath)
            df = resample_to_1hz(df)
            if df is None:
                continue
            df = compute_soc(df)
            df['temp_label'] = temp_label
            df['source']     = os.path.basename(fpath)
            if df['I'].std() > 0.3:   # skip near-static charge files
                dfs.append(df)
                print(f'    + {os.path.basename(fpath)[:55]}: '
                      f'n={len(df):,}, '
                      f'SOC={df["SOC"].min():.2f}-{df["SOC"].max():.2f}, '
                      f'T={df["T"].min():.1f}-{df["T"].max():.1f}C')
        except Exception as e:
            print(f'    SKIP {os.path.basename(fpath)}: {e}')

    for fpath in csv_files:
        try:
            df = load_csv_file(fpath)
            df = resample_to_1hz(df)
            if df is None:
                continue
            df = compute_soc(df)
            df['temp_label'] = temp_label
            df['source']     = os.path.basename(fpath)
            if df['I'].std() > 0.3:
                dfs.append(df)
                print(f'    + {os.path.basename(fpath)[:55]}: '
                      f'n={len(df):,}, '
                      f'SOC={df["SOC"].min():.2f}-{df["SOC"].max():.2f}, '
                      f'T={df["T"].min():.1f}-{df["T"].max():.1f}C')
        except Exception as e:
            print(f'    SKIP {os.path.basename(fpath)}: {e}')

    print(f'  --> {len(dfs)} usable files, '
          f'{sum(len(d) for d in dfs):,} total timesteps')
    return dfs


# ══════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════

def build_xy_pairs(dfs):
    """
    Build one-step prediction pairs.
    X: [SOC_t, T_t, I_t]
    Y: [dSOC,  dT]   (residual targets)
    """
    X_list, Y_list = [], []
    for df in dfs:
        soc = df['SOC'].values
        T   = df['T'].values
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
# 3. MODEL
# ══════════════════════════════════════════════════════════════════

class ElectroThermalMLP(nn.Module):
    """
    3 -> 64 -> 64 -> 32 -> 2  MLP with BatchNorm and Dropout.
    Predicts normalised [dSOC, dT].
    """
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
# 4. TRAINING
# ══════════════════════════════════════════════════════════════════

def train_model(X_train, Y_train, X_val, Y_val, device):
    model     = ElectroThermalMLP().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimiser, patience=10, factor=0.5, verbose=False)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    val_ds   = TensorDataset(torch.tensor(X_val),   torch.tensor(Y_val))

    # pin_memory + num_workers: overlap CPU data loading with GPU compute
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

    best_val_loss = np.inf
    patience_ctr  = 0
    best_state    = None
    history       = {'train': [], 'val': []}

    for epoch in range(1, EPOCHS + 1):

        # Train
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

        # Validate
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
# 5. EVALUATION
# ══════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, Y_test, stats, device):
    """Evaluate on held-out test set in physical units."""
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
    print(f'\n  Absolute SOC accuracy : +/-{soc_mae * 100:.4f}% per step')
    print(f'  Temperature accuracy  : +/-{T_mae:.4f} C per step')

    return {
        'soc_mae':  soc_mae,
        'soc_rmse': soc_rmse,
        'T_mae':    T_mae,
        'T_rmse':   T_rmse,
        'n_test':   len(X_test),
    }


# ══════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('high')   # TF32: ~8x matmul speedup on Ada/Ampere
    print(f'Device     : {device}')
    if device.type == 'cuda':
        print(f'GPU        : {torch.cuda.get_device_name(0)}')
        print(f'VRAM       : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'Data root  : {DATA_ROOT}')
    print()

    # Step 1: Load data
    print('=' * 60)
    print('STEP 1: Loading data')
    print('=' * 60)

    all_dfs = []
    for temp in TEMPS_USE:
        folder = os.path.join(DATA_ROOT, temp)
        if not os.path.isdir(folder):
            print(f'  WARNING: folder not found: {folder}')
            continue
        dfs = load_temperature_folder(folder, temp)
        all_dfs.extend(dfs)

    if len(all_dfs) == 0:
        print('ERROR: No data loaded. Check DATA_ROOT path.')
        return

    total_pts = sum(len(d) for d in all_dfs)
    print(f'\nTotal: {len(all_dfs)} files, {total_pts:,} timesteps')

    # Step 2: Build (X, Y) pairs
    print('\n' + '=' * 60)
    print('STEP 2: Building (X, Y) pairs')
    print('=' * 60)

    X, Y = build_xy_pairs(all_dfs)
    print(f'X shape : {X.shape}   (SOC_t, T_t, I_t)')
    print(f'Y shape : {Y.shape}   (dSOC, dT)')
    print(f'SOC range : {X[:,0].min():.3f} - {X[:,0].max():.3f}')
    print(f'T range   : {X[:,1].min():.1f} - {X[:,1].max():.1f} C')
    print(f'I range   : {X[:,2].min():.2f} - {X[:,2].max():.2f} A')

    # Step 3: Split
    n     = len(X)
    n_tr  = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)

    rng     = np.random.default_rng(SEED)
    idx     = rng.permutation(n)
    tr_idx  = idx[:n_tr]
    val_idx = idx[n_tr : n_tr + n_val]
    te_idx  = idx[n_tr + n_val:]

    X_train, Y_train = X[tr_idx],  Y[tr_idx]
    X_val,   Y_val   = X[val_idx], Y[val_idx]
    X_test,  Y_test  = X[te_idx],  Y[te_idx]

    print(f'\nSplit: train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}')

    # Step 4: Normalise
    stats          = compute_normalisation(X_train, Y_train)
    X_tr_n, Y_tr_n = normalise(X_train, Y_train, stats)
    X_v_n,  Y_v_n  = normalise(X_val,   Y_val,   stats)
    X_te_n, Y_te_n = normalise(X_test,  Y_test,  stats)

    print(f'\nNormalisation (X: SOC, T, I):')
    print(f'  Mean : {stats["X_mean"]}')
    print(f'  Std  : {stats["X_std"]}')

    # Step 5: Train
    print('\n' + '=' * 60)
    print('STEP 3: Training MLP surrogate')
    print('=' * 60)
    print(f'Batch size : {BATCH_SIZE}')
    print(f'LR         : {LR}')
    print(f'Max epochs : {EPOCHS}')
    print(f'Early stop : patience={PATIENCE}')
    print()

    model, history = train_model(X_tr_n, Y_tr_n, X_v_n, Y_v_n, device)

    # Step 6: Evaluate
    print('\n' + '=' * 60)
    print('STEP 4: Evaluation on held-out test set')
    print('=' * 60)

    metrics = evaluate_model(model, X_te_n, Y_te_n, stats, device)

    # Step 7: Save
    model_path = os.path.join(OUTPUT_DIR, 'nmc_surrogate_model.pt')
    stats_path = os.path.join(OUTPUT_DIR, 'nmc_surrogate_stats.pkl')
    hist_path  = os.path.join(OUTPUT_DIR, 'nmc_surrogate_history.pkl')

    torch.save(model.state_dict(), model_path)
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    with open(hist_path, 'wb') as f:
        pickle.dump({'history': history, 'metrics': metrics}, f)

    print(f'\nSaved:')
    print(f'  Model   : {model_path}')
    print(f'  Stats   : {stats_path}')
    print(f'  History : {hist_path}')

    # Paper summary
    print('\n' + '=' * 60)
    print('SUMMARY  -- fill into Table II of paper')
    print('=' * 60)
    print(f'  Chemistry        : NMC (LG HG2 18650)')
    print(f'  Temperatures     : {TEMPS_USE}')
    print(f'  Training samples : {len(X_train):,}')
    print(f'  Test samples     : {len(X_test):,}')
    print(f'  dSOC MAE         : {metrics["soc_mae"]:.6f}  ({metrics["soc_mae"]*100:.4f}%)')
    print(f'  dSOC RMSE        : {metrics["soc_rmse"]:.6f}')
    print(f'  dT   MAE         : {metrics["T_mae"]:.4f} C')
    print(f'  dT   RMSE        : {metrics["T_rmse"]:.4f} C')


# ══════════════════════════════════════════════════════════════════
# INFERENCE HELPER  (used by the RL pack simulator)
# ══════════════════════════════════════════════════════════════════

def load_nmc_surrogate(output_dir):
    """
    Load trained NMC surrogate for RL simulator.

    Example:
        model, stats = load_nmc_surrogate('nmc_surrogate_outputs')
        soc_next, T_next = predict_step(model, stats, soc_t=0.6, T_t=25.0, I_t=3.0)
    """
    model = ElectroThermalMLP()
    model.load_state_dict(
        torch.load(os.path.join(output_dir, 'nmc_surrogate_model.pt'),
                   map_location='cpu'))
    model.eval()
    with open(os.path.join(output_dir, 'nmc_surrogate_stats.pkl'), 'rb') as f:
        stats = pickle.load(f)
    return model, stats


def predict_step(model, stats, soc_t, T_t, I_t):
    """
    One-step prediction: returns (soc_next, T_next).
    """
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