"""
BCICIV 2b - MNE Pipeline版 (正确实现)
使用MNE内置CSP + 滑动时间窗口特征
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MNE_WARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
import mne

DATA_DIR = r"D:\db\BCICIV_2b_gdf"
SFREQ = 250


def load_subject(fp, tmin=-0.5, tmax=4.0):
    raw = mne.io.read_raw_gdf(fp, preload=True, verbose=False)
    ch_map = {ch: ch.split(':')[1] if ':' in ch else ch for ch in raw.ch_names}
    raw.rename_channels(ch_map)
    raw.pick(['C3', 'Cz', 'C4'])
    # Bandpass filter
    raw.filter(0.5, 45, method='iir')
    raw.set_eeg_reference('average', projection=True)
    raw.apply_proj()
    evs_all, _ = mne.events_from_annotations(raw, verbose=False)
    mask = np.isin(evs_all[:, 2], [10, 11])
    evs = evs_all[mask].copy()
    evs[:, 2] = np.where(evs[:, 2] == 10, 0, 1)
    # Extract epochs with baseline correction
    ep = mne.Epochs(raw, evs, tmin=tmin, tmax=tmax, baseline=(None, 0), preload=True, verbose=False)
    return ep


def run_cv(epochs_list, n_splits=5, C=1.0):
    """5-fold CV using MNE CSP pipeline"""
    from mne.decoding import CSP
    
    # Combine all epochs
    X_all = np.vstack([e.get_data() for e in epochs_list])
    y_all = np.concatenate([e.events[:, 2] for e in epochs_list])
    
    print(f"  Data: {X_all.shape}, L={sum(y_all==0)}, R={sum(y_all==1)}")
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for fold_i, (tr_idx, te_idx) in enumerate(cv.split(X_all, y_all)):
        # Get fold data
        X_tr, X_te = X_all[tr_idx], X_all[te_idx]
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]
        
        # Create MNE EpochsArray for CSP fitting
        # Fit CSP on training data only (MNE 1.8 expects raw array)
        csp = CSP(n_components=4, reg='ledoit_wolf', log=True)
        csp.fit(X_tr, y_tr)
        
        # Transform both sets
        Fc_tr = csp.transform(X_tr)
        Fc_te = csp.transform(X_te)
        
        # Scale
        sc = StandardScaler()
        Fc_tr = sc.fit_transform(Fc_tr)
        Fc_te = sc.transform(Fc_te)
        
        # Classify
        clf = SVC(kernel='rbf', C=C)
        clf.fit(Fc_tr, y_tr)
        s = clf.score(Fc_te, y_te)
        scores.append(s)
        print(f"  Fold {fold_i+1}: {s:.1%}")
    
    return np.mean(scores), np.std(scores)


def run_loso(epochs_list, C=1.0):
    """Leave-one-subject-out evaluation"""
    from mne.decoding import CSP
    
    scores = []
    for si in range(len(epochs_list)):
        # Training: all except si
        tr_X = np.vstack([epochs_list[j].get_data() for j in range(len(epochs_list)) if j != si])
        tr_y = np.concatenate([epochs_list[j].events[:, 2] for j in range(len(epochs_list)) if j != si])
        # Test: subject si
        te_X = epochs_list[si].get_data()
        te_y = epochs_list[si].events[:, 2]
        
        # Fit CSP on training subjects
        # Fit CSP on training subjects only
        csp = CSP(n_components=4, reg='ledoit_wolf', log=True)
        csp.fit(tr_X, tr_y)
        
        Fc_tr = csp.transform(tr_X)
        Fc_te = csp.transform(te_X)
        
        sc = StandardScaler()
        Fc_tr = sc.fit_transform(Fc_tr)
        Fc_te = sc.transform(Fc_te)
        
        clf = SVC(kernel='rbf', C=C)
        clf.fit(Fc_tr, tr_y)
        s = clf.score(Fc_te, te_y)
        scores.append(s)
        print(f"  S{si+1}: {s:.1%} (n={len(te_X)})")
    
    return np.mean(scores), np.std(scores)


def main():
    print("=" * 60)
    print("BCICIV 2b - MNE Pipeline版")
    print("=" * 60)
    
    # Load
    print("\n[1] Loading data...")
    epochs_list = []
    for fp in sorted(Path(DATA_DIR).glob("*T.gdf")):
        try:
            ep = load_subject(str(fp))
            epochs_list.append(ep)
            print(f"  {fp.name}: {len(ep)} epochs")
        except Exception as e:
            print(f"  {fp.name}: ERROR - {str(e)[:50]}")
    
    if not epochs_list:
        print("No data loaded!")
        return
    
    # 5-fold CV with different C values
    print("\n[2] 5折交叉验证...")
    best_cv, best_C = 0, 1.0
    for C in [0.1, 0.5, 1, 5, 10, 50]:
        print(f"\n  CSP + SVM(C={C}):")
        mean_s, std_s = run_cv(epochs_list, n_splits=5, C=C)
        print(f"  Mean: {mean_s:.1%} +/- {std_s:.1%}")
        if mean_s > best_cv:
            best_cv, best_C = mean_s, C
    
    print(f"\n  Best: C={best_C} -> {best_cv:.1%}")
    
    # LOSO
    print(f"\n[3] 留一被试LOSO (C={best_C})...")
    loso_mean, loso_std = run_loso(epochs_list, C=best_C)
    print(f"\n  LOSO: {loso_mean:.1%} +/- {loso_std:.1%}")
    print("=" * 60)


if __name__ == '__main__':
    main()
