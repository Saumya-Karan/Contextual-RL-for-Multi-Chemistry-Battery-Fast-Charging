"""
Safety Classifier Training Script
===================================
Trains P(runaway | SOC, chemistry, capacity) using the CHS-labelled
dataset produced by safety_classifier_preprocessing.py.

Class imbalance strategy (two-pronged):
  1. SMOTE (manual implementation) — oversamples minority class
     per chemistry group so LFP runaway cases (only 4/30) are not
     drowned out by majority class.
  2. class_weight='balanced' in all sklearn estimators — further
     penalises misclassification of the minority class.

Models trained:
  A. Logistic Regression        (interpretable baseline)
  B. Random Forest              (handles non-linearity well)
  C. Gradient Boosting (GBDT)   (best expected performance)

Evaluation:
  - Stratified K-Fold (k=5) to preserve class ratios
  - Metrics: ROC-AUC, F1, Precision, Recall, Brier score
  - Calibration check (Brier score) important since output feeds
    directly into RL reward as P(runaway)

Output:
  safety_classifier_results.txt   — metrics summary
  safety_classifier_model.pkl     — best model (for RL deployment)

Reference for CHS label:
  Lin L. et al. J. Energy Storage 61 (2023). DOI:10.1016/j.est.2022.106798
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, brier_score_loss,
                             classification_report, confusion_matrix)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────
# !! SET YOUR PATHS HERE !!
# ──────────────────────────────────────────────────────────────────
INPUT_CSV   = r"safety_classifier_dataset.csv"
OUTPUT_DIR  = os.path.dirname(INPUT_CSV)
RESULTS_TXT = os.path.join(OUTPUT_DIR, "safety_classifier_results.txt")
MODEL_PKL   = os.path.join(OUTPUT_DIR, "safety_classifier_model.pkl")

# ──────────────────────────────────────────────────────────────────
# MANUAL SMOTE
# ──────────────────────────────────────────────────────────────────
def smote_oversample(X, y, k=3, random_state=42):
    """
    Minimal SMOTE implementation (no external library needed).
    For each minority sample, generates synthetic points by
    interpolating with one of its k nearest neighbours.

    Applied per-class so majority class is never touched.
    """
    rng = np.random.default_rng(random_state)
    X = np.array(X, dtype=float)
    y = np.array(y)

    classes, counts = np.unique(y, return_counts=True)
    majority_count  = counts.max()
    majority_class  = classes[counts.argmax()]

    X_aug, y_aug = [X], [y]

    for cls in classes:
        if cls == majority_class:
            continue
        X_min  = X[y == cls]
        n_need = majority_count - len(X_min)
        if n_need <= 0:
            continue

        synthetic = []
        for _ in range(n_need):
            idx   = rng.integers(0, len(X_min))
            sample = X_min[idx]

            # k nearest neighbours within minority class
            dists  = np.linalg.norm(X_min - sample, axis=1)
            dists[idx] = np.inf                       # exclude self
            nn_idx = np.argsort(dists)[:k]
            nn     = X_min[rng.choice(nn_idx)]

            lam    = rng.uniform(0, 1)
            synthetic.append(sample + lam * (nn - sample))

        X_aug.append(np.array(synthetic))
        y_aug.append(np.full(n_need, cls))

    return np.vstack(X_aug), np.concatenate(y_aug)


# ──────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────
def build_features(df):
    """
    Features used:
      - soc            : state of charge at test time (0-100)
      - chemistry_id   : 0=LCO, 1=LFP, 2=NMC  (contextual variable θ)
      - capacity_mAh   : cell capacity
      - is_LCO / is_LFP / is_NMC : one-hot (helps tree models)
      - soc_sq         : SOC² — captures nonlinear runaway threshold
      - soc_x_cap      : interaction term

    Note: t_max, t_rate, v_diff are excluded — 72/89 values are null.
    The CHS score (our label) already encodes those signals.
    At RL deployment time, T and V are available as runtime inputs
    and should be added as features in a future version once a larger
    labelled dataset with complete signals is available.
    """
    feat = pd.DataFrame()
    feat['soc']        = df['soc'] / 100.0          # normalise to [0,1]
    feat['capacity']   = df['capacity_mAh'] / 15000.0  # normalise to [0,1]
    feat['chem_id']    = df['chemistry_id']
    feat['is_LCO']     = (df['chemistry'] == 'LCO').astype(int)
    feat['is_LFP']     = (df['chemistry'] == 'LFP').astype(int)
    feat['is_NMC']     = (df['chemistry'] == 'NMC').astype(int)
    feat['soc_sq']     = feat['soc'] ** 2
    feat['soc_x_cap']  = feat['soc'] * feat['capacity']
    return feat


# ──────────────────────────────────────────────────────────────────
# EVALUATION HELPER
# ──────────────────────────────────────────────────────────────────
def evaluate_model(name, model, X, y, cv, log):
    """
    Runs stratified cross-validation and reports key metrics.
    Uses predict_proba for AUC and Brier score.
    """
    scoring = ['roc_auc', 'f1', 'precision', 'recall']
    cv_res  = cross_validate(model, X, y, cv=cv, scoring=scoring,
                             return_train_score=False)

    # Brier score manually (cross_validate doesn't support it directly)
    brier_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        brier_scores.append(brier_score_loss(yte, proba))

    line = (f"\n{'─'*50}\n"
            f"  Model      : {name}\n"
            f"  ROC-AUC    : {cv_res['test_roc_auc'].mean():.3f} ± {cv_res['test_roc_auc'].std():.3f}\n"
            f"  F1         : {cv_res['test_f1'].mean():.3f} ± {cv_res['test_f1'].std():.3f}\n"
            f"  Precision  : {cv_res['test_precision'].mean():.3f} ± {cv_res['test_precision'].std():.3f}\n"
            f"  Recall     : {cv_res['test_recall'].mean():.3f} ± {cv_res['test_recall'].std():.3f}\n"
            f"  Brier score: {np.mean(brier_scores):.3f} ± {np.std(brier_scores):.3f}\n"
            f"  (Brier: 0=perfect, 0.25=chance, lower is better)\n")
    print(line)
    log.append(line)

    return {
        'auc':    cv_res['test_roc_auc'].mean(),
        'f1':     cv_res['test_f1'].mean(),
        'brier':  np.mean(brier_scores),
        'model':  model,
        'name':   name,
    }


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────
def main():
    log = []

    # ── Load data ────────────────────────────────────────────────
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} cells from {INPUT_CSV}")

    X_df = build_features(df)
    X    = X_df.values
    y    = df['runaway_binary'].values

    header = (
        f"Safety Classifier Training Report\n"
        f"{'='*50}\n"
        f"Dataset     : {len(df)} cells (LCO={sum(df.chemistry=='LCO')}, "
        f"LFP={sum(df.chemistry=='LFP')}, NMC={sum(df.chemistry=='NMC')})\n"
        f"Features    : {list(X_df.columns)}\n"
        f"Label       : runaway_binary (CHS >= 50)\n"
        f"Class dist  : runaway=1 → {y.sum()}, no_runaway=0 → {(1-y).sum()}\n"
        f"Per-chemistry runaway counts:\n"
        f"  LCO: {sum((df.chemistry=='LCO') & (df.runaway_binary==1))}/30\n"
        f"  LFP: {sum((df.chemistry=='LFP') & (df.runaway_binary==1))}/30\n"
        f"  NMC: {sum((df.chemistry=='NMC') & (df.runaway_binary==1))}/29\n"
        f"Imbalance strategy: SMOTE + class_weight='balanced'\n"
    )
    print(header)
    log.append(header)

    # ── Apply SMOTE ───────────────────────────────────────────────
    X_sm, y_sm = smote_oversample(X, y, k=3, random_state=42)
    smote_note = (f"After SMOTE: {Counter(y_sm)}  "
                  f"(added {len(y_sm)-len(y)} synthetic minority samples)\n")
    print(smote_note)
    log.append(smote_note)

    # ── CV splitter ───────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Define models ─────────────────────────────────────────────
    models = [
        ("Logistic Regression",
         Pipeline([
             ('scaler', StandardScaler()),
             ('clf',    LogisticRegression(
                            class_weight='balanced',
                            max_iter=1000,
                            random_state=42))
         ])),

        ("Random Forest",
         RandomForestClassifier(
             n_estimators=300,
             max_depth=4,
             class_weight='balanced',
             random_state=42)),

        ("Gradient Boosting",
         GradientBoostingClassifier(
             n_estimators=200,
             max_depth=3,
             learning_rate=0.05,
             subsample=0.8,
             random_state=42)),
    ]

    # ── Train & evaluate ──────────────────────────────────────────
    results = []
    for name, model in models:
        res = evaluate_model(name, model, X_sm, y_sm, cv, log)
        results.append(res)

    # ── Pick best model (by AUC) ─────────────────────────────────
    best = max(results, key=lambda r: r['auc'])
    best_note = f"\nBest model: {best['name']}  (AUC={best['auc']:.3f})\n"
    print(best_note)
    log.append(best_note)

    # ── Refit best model on full SMOTE dataset ────────────────────
    best['model'].fit(X_sm, y_sm)

    # ── Full-data report ──────────────────────────────────────────
    y_pred  = best['model'].predict(X)
    y_proba = best['model'].predict_proba(X)[:, 1]

    full_report = (
        f"\nFull-data report (best model fitted on all SMOTE data):\n"
        f"{classification_report(y, y_pred, target_names=['No Runaway','Runaway'])}\n"
        f"Confusion matrix:\n{confusion_matrix(y, y_pred)}\n"
        f"  Rows=Actual, Cols=Predicted\n"
        f"  [TN FP]\n"
        f"  [FN TP]\n"
        f"\nFeature importances:\n"
    )
    if hasattr(best['model'], 'feature_importances_'):
        for fname, imp in sorted(
                zip(X_df.columns, best['model'].feature_importances_),
                key=lambda x: -x[1]):
            full_report += f"  {fname:<15}: {imp:.4f}\n"
    elif hasattr(best['model'], 'named_steps'):
        clf = best['model'].named_steps.get('clf')
        if hasattr(clf, 'coef_'):
            for fname, coef in zip(X_df.columns, clf.coef_[0]):
                full_report += f"  {fname:<15}: {coef:.4f}\n"

    print(full_report)
    log.append(full_report)

    # ── Save results ──────────────────────────────────────────────
    with open(RESULTS_TXT, 'w') as f:
        f.write('\n'.join(log))
    print(f"Results saved to: {RESULTS_TXT}")

    # ── Save model ────────────────────────────────────────────────
    with open(MODEL_PKL, 'wb') as f:
        pickle.dump({
            'model':        best['model'],
            'model_name':   best['name'],
            'feature_cols': list(X_df.columns),
            'label':        'runaway_binary (CHS >= 50)',
            'chemistry_map':'0=LCO, 1=LFP, 2=NMC',
        }, f)
    print(f"Model saved to : {MODEL_PKL}")
    print("\nTo use in RL reward function:")
    print("  p = model.predict_proba([[soc/100, cap/15000, chem_id,")
    print("                            is_LCO, is_LFP, is_NMC,")
    print("                            (soc/100)**2, soc/100*cap/15000]])[0,1]")
    print("  reward -= lambda * p")


if __name__ == '__main__':
    main()