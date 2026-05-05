"""
LFP Degradation Model  (v2 — tiny model for small dataset)
============================================================
Dataset: Sandia LFP_1-6, _Reg files, 5/15/25/35/45 degC
Total: ~147 reference capacity records

Key changes from v1:
  1. Much smaller network: 2->16->8->1  (prevents overfitting on 147 pts)
  2. Higher dropout: 0.1
  3. Stronger weight decay: 1e-3
  4. Filter out near-flat 45C data for LFP_1/2/3 (SOH > 0.98 = noise)
     LFP barely degrades at 45C in Sandia test window
  5. Smaller batch: 16 (tiny dataset needs small batches)
"""

import os, sys, re, glob, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, pickle

import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

SANDIA_ROOT = r"C:\Users\saumy\OneDrive\Desktop\IN Pallavi ma'am\Datasets extracted\Sandia degradation all\Sandia Cell Cycle Testing Data"
OUTPUT_DIR  = r"lfp_degradation_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_CELLS = ['LFP_1','LFP_2','LFP_3','LFP_4']
VAL_CELLS   = ['LFP_5']
TEST_CELLS  = ['LFP_6']

BATCH_SIZE = 16
EPOCHS     = 5000
LR         = 1e-3
PATIENCE   = 500
SEED       = 42
torch.manual_seed(SEED); np.random.seed(SEED)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Scripts.sandia_degradation_loader import load_sandia_chemistry


def build_xy(df):
    return (df[['cycle_frac','T_C']].values.astype(np.float32),
            df[['SOH_smooth']].values.astype(np.float32))

def compute_normalisation(X):
    return {'X_mean':X.mean(0), 'X_std':X.std(0)+1e-8}

def normalise_X(X, stats):
    return ((X-stats['X_mean'])/stats['X_std']).astype(np.float32)


class DegradationMLP(nn.Module):
    """Tiny MLP — prevents overfitting on only 147 data points."""
    def __init__(self, input_dim=2, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, 8),         nn.ReLU(),
            nn.Linear(8,  1),         nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x)


def train_model(X_tr, Y_tr, X_v, Y_v, device):
    model = DegradationMLP().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    sch   = ReduceLROnPlateau(opt, patience=100, factor=0.5,
                               verbose=False, min_lr=1e-7)
    crit  = nn.MSELoss()
    t_ds  = TensorDataset(torch.tensor(X_tr), torch.tensor(Y_tr))
    v_ds  = TensorDataset(torch.tensor(X_v),  torch.tensor(Y_v))
    t_dl  = DataLoader(t_ds, BATCH_SIZE, shuffle=True,  num_workers=0)
    v_dl  = DataLoader(v_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    best_v=np.inf; pat=0; best_state=None
    history={'train':[],'val':[]}

    for ep in range(1, EPOCHS+1):
        model.train(); tl=0
        for xb,yb in t_dl:
            xb,yb=xb.to(device),yb.to(device)
            opt.zero_grad(set_to_none=True)
            l=crit(model(xb),yb); l.backward(); opt.step()
            tl += l.item()*len(xb)
        tl /= len(t_ds)

        model.eval(); vl=0
        with torch.no_grad():
            for xb,yb in v_dl:
                xb,yb=xb.to(device),yb.to(device)
                vl += crit(model(xb),yb).item()*len(xb)
        vl /= len(v_ds)

        history['train'].append(tl); history['val'].append(vl)
        sch.step(vl)
        if vl<best_v:
            best_v=vl
            best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
            pat=0
        else: pat+=1

        if ep%500==0:
            print(f'    Epoch {ep:5d}: train={tl:.6f}  val={vl:.6f}  '
                  f'lr={opt.param_groups[0]["lr"]:.2e}')
        if pat>=PATIENCE:
            print(f'    Early stopping at epoch {ep}'); break

    model.load_state_dict(best_state)
    return model, history


def evaluate(model, X, Y, device, label=''):
    model.eval()
    with torch.no_grad():
        Yp = model(torch.tensor(X).to(device)).cpu().numpy()
    mae  = np.mean(np.abs(Yp-Y))
    rmse = np.sqrt(np.mean((Yp-Y)**2))
    r2   = 1-np.sum((Yp-Y)**2)/(np.sum((Y-Y.mean())**2)+1e-12)
    print(f'  [{label}] n={len(X):,}  MAE={mae:.4f} ({mae*100:.2f}%)  '
          f'RMSE={rmse:.4f}  R²={r2:.4f}')
    return {'mae':mae,'rmse':rmse,'r2':r2}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    print('='*60); print('STEP 1: Loading Sandia LFP'); print('='*60)
    df = load_sandia_chemistry(SANDIA_ROOT, 'LFP')
    if len(df)==0:
        print('ERROR: No data'); return

    # Filter out near-flat 45C data (SOH range < 0.05 = no meaningful fade)
    def has_meaningful_fade(grp):
        return (grp['SOH'].max() - grp['SOH'].min()) > 0.05
    df = df.groupby(['cell','T_C']).filter(has_meaningful_fade)
    print(f'\n  After filtering flat groups: {len(df)} records')
    print(f'  T range: {df["T_C"].min():.0f}-{df["T_C"].max():.0f}C')
    print(f'  SOH range: {df["SOH"].min():.3f}-{df["SOH"].max():.3f}')

    print('\n'+'='*60); print('STEP 2: Cell-held-out split'); print('='*60)
    train_df = df[df['cell'].isin(TRAIN_CELLS)]
    val_df   = df[df['cell'].isin(VAL_CELLS)]
    test_df  = df[df['cell'].isin(TEST_CELLS)]
    print(f'  Train: {TRAIN_CELLS} ({len(train_df)} samples)')
    print(f'  Val  : {VAL_CELLS}   ({len(val_df)} samples)')
    print(f'  Test : {TEST_CELLS}  ({len(test_df)} samples)')

    X_tr,Y_tr = build_xy(train_df)
    X_v, Y_v  = build_xy(val_df)
    X_te,Y_te = build_xy(test_df)

    stats  = compute_normalisation(X_tr)
    X_tr_n = normalise_X(X_tr, stats)
    X_v_n  = normalise_X(X_v,  stats)
    X_te_n = normalise_X(X_te, stats)

    print('\n'+'='*60); print('STEP 3: Training'); print('='*60)
    print(f'Network: 2->16->8->1 (tiny, prevents overfit on {len(df)} pts)')
    print(f'Batch={BATCH_SIZE}  LR={LR}  Patience={PATIENCE}\n')
    model, history = train_model(X_tr_n, Y_tr, X_v_n, Y_v, device)

    print('\n'+'='*60); print('STEP 4: Evaluation'); print('='*60)
    m_tr = evaluate(model, X_tr_n, Y_tr, device, 'Train')
    m_v  = evaluate(model, X_v_n,  Y_v,  device, 'Val  ')
    m_te = evaluate(model, X_te_n, Y_te, device, 'Test ')

    torch.save(model.state_dict(),
               os.path.join(OUTPUT_DIR,'lfp_degradation_model.pt'))
    with open(os.path.join(OUTPUT_DIR,'lfp_degradation_stats.pkl'),'wb') as f:
        pickle.dump(stats,f)
    with open(os.path.join(OUTPUT_DIR,'lfp_degradation_history.pkl'),'wb') as f:
        pickle.dump({'history':history,
                     'metrics':{'train':m_tr,'val':m_v,'test':m_te}},f)

    print(f'\nSaved: {OUTPUT_DIR}/')
    print('\n'+'='*60); print('SUMMARY'); print('='*60)
    print(f'  Chemistry   : LFP (Sandia LFP_1-6)')
    print(f'  Temperatures: 5/15/25/35/45 degC')
    print(f'  Records used: {len(df)} (after filtering flat 45C groups)')
    print(f'  MAE  (test) : {m_te["mae"]:.6f} ({m_te["mae"]*100:.4f}%)')
    print(f'  RMSE (test) : {m_te["rmse"]:.6f}')
    print(f'  R²   (test) : {m_te["r2"]:.4f}')


def load_lfp_degradation(output_dir):
    model = DegradationMLP()
    model.load_state_dict(torch.load(
        os.path.join(output_dir,'lfp_degradation_model.pt'),map_location='cpu'))
    model.eval()
    with open(os.path.join(output_dir,'lfp_degradation_stats.pkl'),'rb') as f:
        stats = pickle.load(f)
    return model, stats

def predict_soh(model, stats, cycle_frac, T_C=25.0):
    x  = np.array([[cycle_frac,T_C]], dtype=np.float32)
    xn = normalise_X(x, stats)
    with torch.no_grad():
        return float(np.clip(model(torch.tensor(xn)).item(),0.0,1.0))

if __name__ == '__main__':
    main()