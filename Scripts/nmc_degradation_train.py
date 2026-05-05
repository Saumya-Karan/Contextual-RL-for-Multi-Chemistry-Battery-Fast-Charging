"""
NMC Degradation Model  (v5 — fixed Sandia loader)
====================================================
Combines CALCE CS2 (cycle-level detail at 25C) with
Sandia NMC_1-6 (reference checks at 5/15/25/35/45C).

Bug fixed from v4:
  - Script now scans all NMC_X_TC and NMC_X_TC1-TC2 folders
  - Only _Reg files loaded (not _Mod)
  - xlrd correctly used for .xls files
"""

import os, sys, re, glob, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import openpyxl, pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ── Paths ─────────────────────────────────────────────────────────
CALCE_ROOT  = r"C:\Users\saumy\OneDrive\Desktop\IN Pallavi ma'am\Datasets extracted\NMC degradation"
SANDIA_ROOT = r"C:\Users\saumy\OneDrive\Desktop\IN Pallavi ma'am\Datasets extracted\Sandia degradation all\Sandia Cell Cycle Testing Data"
OUTPUT_DIR  = r"nmc_degradation_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────
T_NOM_CALCE  = 25.0
CALCE_TRAIN  = ['CS2_33','CS2_34','CS2_35','CS2_36']
CALCE_VAL    = ['CS2_37']
CALCE_TEST   = ['CS2_38']

BATCH_SIZE = 64
EPOCHS     = 3000
LR         = 5e-4
PATIENCE   = 300
SEED       = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# ── Import shared Sandia loader ────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Scripts.sandia_degradation_loader import load_sandia_chemistry

# ══════════════════════════════════════════════════════════════════
# 1. CALCE LOADER
# ══════════════════════════════════════════════════════════════════

def get_data_sheet(fpath):
    try:
        eng = 'xlrd' if fpath.endswith('.xls') else 'openpyxl'
        xl  = pd.ExcelFile(fpath, engine=eng)
        for s in xl.sheet_names:
            if 'Channel' in s and 'Chart' not in s and 'Stat' not in s:
                return s
    except: pass
    return None


def load_calce_cell(cell_folder):
    cell_id  = os.path.basename(cell_folder)
    all_xlsx = (sorted(glob.glob(os.path.join(cell_folder,'*.xlsx')))
              + sorted([f for f in glob.glob(os.path.join(cell_folder,'*'))
                        if os.path.isfile(f)
                        and '.' not in os.path.basename(f)]))
    records = []; global_cycle = 0
    for fpath in all_xlsx:
        sheet = get_data_sheet(fpath)
        try:
            df = pd.read_excel(fpath,
                               sheet_name=sheet if sheet else 0,
                               header=0, engine='openpyxl')
        except: continue
        if 'Cycle_Index' not in df.columns: continue
        cyc_cap   = df.groupby('Cycle_Index')['Discharge_Capacity(Ah)'].max()
        per_cycle = cyc_cap.diff(); per_cycle.iloc[0] = cyc_cap.iloc[0]
        valid = per_cycle[per_cycle > 0.5]
        for _, cap in valid.items():
            global_cycle += 1
            records.append({'cycle': global_cycle, 'cap_Ah': float(cap),
                            'T_C': T_NOM_CALCE, 'cell': cell_id,
                            'source': 'CALCE'})
    if not records: return pd.DataFrame()
    r = pd.DataFrame(records)
    c_init = r['cap_Ah'].iloc[:5].median()
    r['SOH']        = (r['cap_Ah'] / c_init).clip(0.0, 1.0)
    r['cycle_frac'] = r['cycle'] / r['cycle'].max()
    r['SOH_smooth'] = r['SOH'].rolling(7, center=True, min_periods=1).median()
    return r


def load_calce_all():
    records = []
    for top in sorted(glob.glob(os.path.join(CALCE_ROOT,'CS2_*'))):
        if not os.path.isdir(top): continue
        name = os.path.basename(top)
        m = re.search(r'CS2_(\d+)', name)
        if not m or int(m.group(1)) in [8,21]: continue
        inner = os.path.join(top, name)
        df = load_calce_cell(inner if os.path.isdir(inner) else top)
        if len(df) == 0: continue
        split = ('TRAIN' if name in CALCE_TRAIN else
                 'VAL'   if name in CALCE_VAL   else
                 'TEST'  if name in CALCE_TEST  else '?')
        records.append(df)
        print(f'    [CALCE/{split}] {name}: {len(df)} cycles, '
              f'SOH {df["SOH"].min():.3f}-{df["SOH"].max():.3f}')
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()

# ══════════════════════════════════════════════════════════════════
# 2. FEATURES / MODEL / TRAINING  (same as v3/v4)
# ══════════════════════════════════════════════════════════════════

def build_xy(df):
    return (df[['cycle_frac','T_C']].values.astype(np.float32),
            df[['SOH_smooth']].values.astype(np.float32))

def compute_normalisation(X):
    return {'X_mean': X.mean(0), 'X_std': X.std(0)+1e-8}

def normalise_X(X, stats):
    return ((X - stats['X_mean']) / stats['X_std']).astype(np.float32)


class DegradationMLP(nn.Module):
    def __init__(self, input_dim=2, dropout=0.02):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128,128),       nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128,64),        nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64,32),         nn.ReLU(),
            nn.Linear(32,1),          nn.Sigmoid(),
        )
    def forward(self,x): return self.net(x)


def train_model(X_tr, Y_tr, X_v, Y_v, device):
    model = DegradationMLP().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sch   = ReduceLROnPlateau(opt, patience=50, factor=0.5,
                               verbose=False, min_lr=1e-7)
    crit  = nn.MSELoss()
    t_ds  = TensorDataset(torch.tensor(X_tr), torch.tensor(Y_tr))
    v_ds  = TensorDataset(torch.tensor(X_v),  torch.tensor(Y_v))
    t_dl  = DataLoader(t_ds, BATCH_SIZE, shuffle=True,  num_workers=0)
    v_dl  = DataLoader(v_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    best_v = np.inf; pat = 0; best_state = None
    history = {'train':[], 'val':[]}

    for ep in range(1, EPOCHS+1):
        model.train(); tl=0
        for xb,yb in t_dl:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            l = crit(model(xb),yb); l.backward(); opt.step()
            tl += l.item()*len(xb)
        tl /= len(t_ds)

        model.eval(); vl=0
        with torch.no_grad():
            for xb,yb in v_dl:
                xb,yb = xb.to(device),yb.to(device)
                vl += crit(model(xb),yb).item()*len(xb)
        vl /= len(v_ds)

        history['train'].append(tl); history['val'].append(vl)
        sch.step(vl)
        if vl < best_v:
            best_v = vl
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            pat = 0
        else: pat += 1

        if ep % 300 == 0:
            print(f'    Epoch {ep:5d}: train={tl:.6f}  val={vl:.6f}  '
                  f'lr={opt.param_groups[0]["lr"]:.2e}')
        if pat >= PATIENCE:
            print(f'    Early stopping at epoch {ep}'); break

    model.load_state_dict(best_state)
    return model, history


def evaluate(model, X, Y, device, label=''):
    model.eval()
    with torch.no_grad():
        Yp = model(torch.tensor(X).to(device)).cpu().numpy()
    mae  = np.mean(np.abs(Yp-Y))
    rmse = np.sqrt(np.mean((Yp-Y)**2))
    r2   = 1 - np.sum((Yp-Y)**2) / (np.sum((Y-Y.mean())**2)+1e-12)
    print(f'  [{label}] n={len(X):,}  MAE={mae:.4f} ({mae*100:.2f}%)  '
          f'RMSE={rmse:.4f}  R²={r2:.4f}')
    return {'mae':mae,'rmse':rmse,'r2':r2}

# ══════════════════════════════════════════════════════════════════
# 3. MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    print('='*60); print('STEP 1a: CALCE CS2'); print('='*60)
    df_calce = load_calce_all()
    print(f'  CALCE: {len(df_calce)} records')

    print('\n'+'='*60); print('STEP 1b: Sandia NMC'); print('='*60)
    df_sandia = load_sandia_chemistry(SANDIA_ROOT, 'NMC')
    n_san = len(df_sandia) if len(df_sandia)>0 else 0

    df_all = pd.concat([df_calce]+([df_sandia] if n_san>0 else []),
                       ignore_index=True)
    print(f'\n  Combined: {len(df_all)} records, '
          f'T={df_all["T_C"].min():.0f}-{df_all["T_C"].max():.0f}C')

    print('\n'+'='*60); print('STEP 2: Split'); print('='*60)
    calce_tr = df_calce[df_calce['cell'].isin(CALCE_TRAIN)]
    calce_v  = df_calce[df_calce['cell'].isin(CALCE_VAL)]
    calce_te = df_calce[df_calce['cell'].isin(CALCE_TEST)]

    train_df = pd.concat([calce_tr]+([df_sandia] if n_san>0 else []),
                         ignore_index=True)
    val_df   = calce_v
    test_df  = calce_te

    X_tr, Y_tr = build_xy(train_df)
    X_v,  Y_v  = build_xy(val_df)
    X_te, Y_te = build_xy(test_df)
    print(f'  Train={len(X_tr)} (CALCE {len(calce_tr)} + Sandia {n_san})')
    print(f'  Val={len(X_v)} (CS2_37)   Test={len(X_te)} (CS2_38)')

    stats = compute_normalisation(X_tr)
    X_tr_n = normalise_X(X_tr, stats)
    X_v_n  = normalise_X(X_v,  stats)
    X_te_n = normalise_X(X_te, stats)

    print('\n'+'='*60); print('STEP 3: Training'); print('='*60)
    print(f'Batch={BATCH_SIZE}  LR={LR}  Epochs={EPOCHS}  Patience={PATIENCE}\n')
    model, history = train_model(X_tr_n, Y_tr, X_v_n, Y_v, device)

    print('\n'+'='*60); print('STEP 4: Evaluation'); print('='*60)
    m_tr = evaluate(model, X_tr_n, Y_tr, device, 'Train')
    m_v  = evaluate(model, X_v_n,  Y_v,  device, 'Val  ')
    m_te = evaluate(model, X_te_n, Y_te, device, 'Test ')

    torch.save(model.state_dict(),
               os.path.join(OUTPUT_DIR,'nmc_degradation_model.pt'))
    with open(os.path.join(OUTPUT_DIR,'nmc_degradation_stats.pkl'),'wb') as f:
        pickle.dump(stats,f)
    with open(os.path.join(OUTPUT_DIR,'nmc_degradation_history.pkl'),'wb') as f:
        pickle.dump({'history':history,'metrics':{'train':m_tr,'val':m_v,'test':m_te}},f)

    print(f'\nSaved: {OUTPUT_DIR}/')
    print('\n'+'='*60); print('SUMMARY'); print('='*60)
    print(f'  Chemistry   : NMC (CALCE CS2 + Sandia NMC_1-6)')
    print(f'  CALCE       : {len(df_calce)} cycles at 25C')
    print(f'  Sandia      : {n_san} ref points at 5/15/25/35/45C')
    print(f'  MAE  (test) : {m_te["mae"]:.6f} ({m_te["mae"]*100:.4f}%)')
    print(f'  RMSE (test) : {m_te["rmse"]:.6f}')
    print(f'  R²   (test) : {m_te["r2"]:.4f}')


def load_nmc_degradation(output_dir):
    model = DegradationMLP()
    model.load_state_dict(torch.load(
        os.path.join(output_dir,'nmc_degradation_model.pt'), map_location='cpu'))
    model.eval()
    with open(os.path.join(output_dir,'nmc_degradation_stats.pkl'),'rb') as f:
        stats = pickle.load(f)
    return model, stats

def predict_soh(model, stats, cycle_frac, T_C=25.0):
    x  = np.array([[cycle_frac, T_C]], dtype=np.float32)
    xn = normalise_X(x, stats)
    with torch.no_grad():
        return float(np.clip(model(torch.tensor(xn)).item(), 0.0, 1.0))

if __name__ == '__main__':
    main()