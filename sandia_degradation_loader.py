"""
Shared Sandia Degradation Data Loader
=======================================
Reads _Reg files for NMC, LFP, and LCO chemistries.

IMPORTANT: .xls files are read with xlrd directly (bypassing pandas)
because pandas >= 2.0 dropped xlrd 1.2.0 support, and xlrd 2.x
dropped .xls support. The two are incompatible when used together
through pandas. Direct xlrd reading sidesteps this entirely.

.xlsx files are read normally with openpyxl through pandas.
"""

import os, re, glob, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════
# XLS READER — bypasses pandas, uses xlrd directly
# ══════════════════════════════════════════════════════════════════

def read_xls_direct(fpath):
    """
    Read an Arbin .xls file directly with xlrd, bypassing pandas.
    Returns a DataFrame with standard Arbin columns, or None on failure.
    """
    try:
        import xlrd
    except ImportError:
        print(f'    ERROR: xlrd not installed. Run: python3.11.exe -m pip install xlrd==1.2.0')
        return None

    try:
        wb    = xlrd.open_workbook(fpath)
        # Find the Channel data sheet
        sheet = None
        for name in wb.sheet_names():
            if 'Channel' in name and 'Chart' not in name and 'Stat' not in name:
                sheet = wb.sheet_by_name(name)
                break
        if sheet is None:
            return None

        # First row = column headers
        headers = [str(sheet.cell_value(0, c)).strip()
                   for c in range(sheet.ncols)]

        # Read all rows into a list of dicts
        rows = []
        for r in range(1, sheet.nrows):
            row = {}
            for c, h in enumerate(headers):
                val = sheet.cell_value(r, c)
                row[h] = val
            rows.append(row)

        if not rows:
            return None

        df = pd.DataFrame(rows)

        # Convert numeric columns
        for col in ['Cycle_Index', 'Step_Index', 'Current(A)',
                    'Voltage(V)', 'Charge_Capacity(Ah)',
                    'Discharge_Capacity(Ah)']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    except Exception as e:
        return None


def read_xlsx_pandas(fpath):
    """Read .xlsx file normally through pandas + openpyxl."""
    try:
        xl = pd.ExcelFile(fpath, engine='openpyxl')
        sheet = None
        for s in xl.sheet_names:
            if 'Channel' in s and 'Chart' not in s and 'Stat' not in s:
                sheet = s; break
        return pd.read_excel(fpath,
                             sheet_name=sheet if sheet else 0,
                             header=0, engine='openpyxl')
    except:
        return None


def read_any_excel(fpath):
    """Read .xls or .xlsx, choosing the right method."""
    if fpath.lower().endswith('.xls'):
        return read_xls_direct(fpath)
    else:
        return read_xlsx_pandas(fpath)


# ══════════════════════════════════════════════════════════════════
# CAPACITY EXTRACTION
# ══════════════════════════════════════════════════════════════════

def per_cycle_capacity(df, min_cap=0.1):
    """Extract per-cycle discharge capacity from cumulative DisAh."""
    if df is None: return pd.Series(dtype=float)
    if 'Cycle_Index' not in df.columns or \
       'Discharge_Capacity(Ah)' not in df.columns:
        return pd.Series(dtype=float)
    cyc_cap   = df.groupby('Cycle_Index')['Discharge_Capacity(Ah)'].max()
    per_cycle = cyc_cap.diff()
    per_cycle.iloc[0] = cyc_cap.iloc[0]
    return per_cycle[per_cycle > min_cap]


def parse_temp_from_filename(fname):
    """NMC_1_25C_Reg.xls -> 25"""
    m = re.search(r'_(\d+)C_', os.path.basename(fname))
    return int(m.group(1)) if m else None


# ══════════════════════════════════════════════════════════════════
# MAIN LOADER
# ══════════════════════════════════════════════════════════════════

def load_sandia_chemistry(sandia_root, chemistry):
    """
    Load all _Reg files for one chemistry (NMC, LFP, or LCO).
    Returns DataFrame with columns:
      [cell, T_C, cycle_frac, SOH, SOH_smooth, source]
    """
    all_records = []

    # Find all folders matching chemistry_X_*
    folders = sorted([
        f for f in glob.glob(os.path.join(sandia_root, f'{chemistry}_*'))
        if os.path.isdir(f)
    ])

    if not folders:
        print(f'    WARNING: No {chemistry} folders in {sandia_root}')
        return pd.DataFrame()

    # Group folders by cell number
    by_cell = {}
    for folder in folders:
        m = re.search(rf'{chemistry}_(\d+)_', os.path.basename(folder))
        if not m: continue
        cell_num = int(m.group(1))
        by_cell.setdefault(cell_num, []).append(folder)

    for cell_num in sorted(by_cell.keys()):
        cell_id   = f'{chemistry}_{cell_num}'
        cell_recs = []

        for folder in by_cell[cell_num]:
            # Only _Reg files — skip _Mod
            reg_files = (
                sorted(glob.glob(os.path.join(folder, '*_Reg.xls')))
              + sorted(glob.glob(os.path.join(folder, '*_Reg.xlsx')))
            )

            for fpath in reg_files:
                temp_C = parse_temp_from_filename(fpath)
                if temp_C is None:
                    continue

                df = read_any_excel(fpath)
                valid = per_cycle_capacity(df, min_cap=0.1)
                if len(valid) == 0:
                    continue

                for idx, cap in valid.items():
                    cell_recs.append({
                        'cap_Ah':     float(cap),
                        'cyc_within': int(idx),
                        'T_C':        float(temp_C),
                        'cell':       cell_id,
                        'source':     'Sandia',
                    })

        if not cell_recs:
            continue

        df_cell = pd.DataFrame(cell_recs)

        # Process per temperature
        temp_dfs = []
        for temp_val in sorted(df_cell['T_C'].unique()):
            sub = df_cell[df_cell['T_C'] == temp_val].copy().reset_index(drop=True)
            if len(sub) < 2: continue
            c_init = sub['cap_Ah'].iloc[0]
            if c_init < 0.05: continue
            sub['SOH']        = (sub['cap_Ah'] / c_init).clip(0.0, 1.0)
            max_cyc           = max(sub['cyc_within'].max(), 1)
            sub['cycle_frac'] = sub['cyc_within'] / max_cyc
            sub['SOH_smooth'] = sub['SOH']
            temp_dfs.append(sub)
            print(f'    [{chemistry}/Sandia] {cell_id} T={int(temp_val)}C: '
                  f'{len(sub)} pts, '
                  f'SOH {sub["SOH"].min():.3f}-{sub["SOH"].max():.3f}')

        if temp_dfs:
            all_records.append(pd.concat(temp_dfs, ignore_index=True))

    if not all_records:
        return pd.DataFrame()

    result = pd.concat(all_records, ignore_index=True)
    n_cells = result['cell'].nunique()
    print(f'    [{chemistry}/Sandia] Total: {len(result)} records '
          f'from {n_cells} cells, '
          f'T={result["T_C"].min():.0f}-{result["T_C"].max():.0f}C')
    return result