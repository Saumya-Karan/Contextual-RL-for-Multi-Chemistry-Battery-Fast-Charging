"""
Quick diagnostic: what does the surrogate actually predict?
Run this before any RL training to verify surrogate outputs make sense.
"""
import sys, os, pickle
import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = r"C:\Users\saumy\OneDrive\Desktop\IN Pallavi ma'am\Scripts"

class ElectroThermalMLP(nn.Module):
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

print("="*60)
print("Surrogate sanity check")
print("="*60)

for chem, folder in [('NMC','nmc_surrogate_outputs'),
                      ('LFP','lfp_surrogate_outputs'),
                      ('LCO','lco_surrogate_outputs')]:
    wpath = os.path.join(SCRIPT_DIR, folder, f'{chem.lower()}_surrogate_model.pt')
    spath = os.path.join(SCRIPT_DIR, folder, f'{chem.lower()}_surrogate_stats.pkl')

    model = ElectroThermalMLP()
    model.load_state_dict(torch.load(wpath, map_location='cpu'))
    model.eval()
    with open(spath,'rb') as f: stats = pickle.load(f)

    print(f"\n{chem} surrogate:")
    print(f"  X_mean={stats['X_mean']}  X_std={stats['X_std']}")
    print(f"  Y_mean={stats['Y_mean']}  Y_std={stats['Y_std']}")

    # Convention per chemistry
    sign = -1 if chem == 'LFP' else +1
    convention = "negative=charge (LFP/Arbin)" if chem == 'LFP' else "positive=charge (NMC/LCO/CALCE)"
    print(f"  Convention: {convention}")
    for I_mag in [0.5, 1.0, 2.0, 5.0]:
        I_cell = sign * I_mag
        x = np.array([[0.5, 25.0, I_cell]], dtype=np.float32)
        xn = (x - stats['X_mean']) / stats['X_std']
        with torch.no_grad():
            yn = model(torch.tensor(xn)).numpy()
        delta = yn * stats['Y_std'] + stats['Y_mean']
        dSOC = float(delta[0,0])
        dT   = float(delta[0,1])
        steps_needed = int(0.6 / dSOC) if dSOC > 0 else 999999
        print(f"  I_mag={I_mag:.1f}A (fed as {I_cell:+.1f}A) -> "
              f"dSOC={dSOC:+.6f}  steps={steps_needed:,}  "
              f"{'✓' if dSOC > 0 else '✗'}")