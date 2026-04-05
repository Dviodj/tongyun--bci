"""
BCICIV 2b - 快速验证版
策略: 预计算所有特征，只在每折中做快速CSP变换
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['OMP_NUM_THREADS'] = '4'
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.linalg import eigh
import mne

DATA_DIR = r"D:\db\BCICIV_2b_gdf"
SFREQ = 250


def load_subject(fp):
    raw = mne.io.read_raw_gdf(fp, preload=True, verbose=False)
    ch_map = {ch: ch.split(':')[1] if ':' in ch else ch for ch in raw.ch_names}
    raw.rename_channels(ch_map)
    raw.pick(['C3', 'Cz', 'C4'])
    raw.filter(0.5, 45, method='iir')
    raw.set_eeg_reference('average', projection=True)
    raw.apply_proj()
    evs_all, _ = mne.events_from_annotations(raw, verbose=False)
    mask = np.isin(evs_all[:, 2], [10, 11])
    evs = evs_all[mask].copy()
    evs[:, 2] = np.where(evs[:, 2] == 10, 0, 1)
    ep = mne.Epochs(raw, evs, tmin=-0.5, tmax=3.5, baseline=(None, 0), preload=True, verbose=False)
    return ep


def quick_csp(X_tr, y_tr, X_te, n_comp=2):
    """Fast CSP using manual implementation"""
    n_tr, nc, nt = X_tr.shape
    # Compute mean covariance
    cov1 = np.mean([np.cov(x) for x, y in zip(X_tr, y_tr) if y == 0], axis=0)
    cov2 = np.mean([np.cov(x) for x, y in zip(X_tr, y_tr) if y == 1], axis=0)
    cov_sum = cov1 + cov2 + 1e-6 * np.eye(nc)
    try:
        lam, W = eigh(cov2, cov_sum)
    except:
        return np.zeros((len(X_te), n_comp * 2))
    idx = np.argsort(lam)[::-1]
    W = W[:, idx[:n_comp * 2]]
    # Project and compute variance ratio
    proj_tr = np.tensordot(X_tr, W, axes=([1], [0]))
    proj_te = np.tensordot(X_te, W, axes=([1], [0]))
    var_tr = np.var(proj_tr, axis=2)
    var_te = np.var(proj_te, axis=2)
    var_tr = var_tr / (var_tr.sum(axis=1, keepdims=True) + 1e-10)
    var_te = var_te / (var_te.sum(axis=1, keepdims=True) + 1e-10)
    return np.log(var_te + 1e-10), np.log(var_tr + 1e-10)


def main():
    print("=" * 60)
    print("BCICIV 2b - 快速验证版")
    print("=" * 60)
    
    # Load
    print("\n[1] Loading...")
    epochs_list = []
    for fp in sorted(Path(DATA_DIR).glob("*T.gdf")):
        try:
            ep = load_subject(str(fp))
            epochs_list.append(ep)
            print(f"  {fp.name}: {len(ep)} epochs")
        except Exception as e:
            print(f"  {fp.name}: ERROR - {str(e)[:40]}")
    
    X_all = np.vstack([ep.get_data() for ep in epochs_list])
    y_all = np.concatenate([ep.events[:, 2] for ep in epochs_list])
    print(f"\nTotal: {X_all.shape}")
    
    # Extract band power features (simple variance per band)
    print("\n[2] Extracting features...")
    from mne.time_frequency import psd_array_welch
    
    def extract_band(X, bands=[(8, 13), (13, 30)]):
        n = len(X)
        F = []
        for i in range(n):
            fv = []
            for ch in range(3):
                s = X[i, ch]
                for f1, f2 in bands:
                    _, p = psd_array_welch(s, SFREQ, fmin=f1, fmax=f2, n_fft=256, average='mean')
                    fv.extend([10 * np.log10(np.mean(p) + 1e-12)])
                fv.extend([np.mean(s), np.std(s)])
            F.append(fv)
        return np.array(F)
    
    F = extract_band(X_all)
    print(f"  Features: {F.shape}")
    
    # Scale features
    sc_feat = StandardScaler()
    Fs = np.nan_to_num(sc_feat.fit_transform(F), nan=0, posinf=10, neginf=-10)
    
    # 5-fold CV
    print("\n[3] 5折交叉验证...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    for name, clf_fn in [
        ("LDA", lambda: LDA()),
        ("SVM C=1", lambda: SVC(kernel='rbf', C=1)),
        ("SVM C=5", lambda: SVC(kernel='rbf', C=5)),
        ("SVM C=10", lambda: SVC(kernel='rbf', C=10)),
    ]:
        scores = []
        for tr, te in cv.split(X_all, y_all):
            # CSP on train only
            Fc_te, Fc_tr = quick_csp(X_all[tr], y_all[tr], X_all[te])
            sc_csp = StandardScaler()
            Fc_tr = np.nan_to_num(sc_csp.fit_transform(Fc_tr), nan=0, posinf=5, neginf=-5)
            Fc_te = np.nan_to_num(sc_csp.transform(Fc_te), nan=0, posinf=5, neginf=-5)
            
            # Combine features
            Xf_tr = np.hstack([Fs[tr], Fc_tr])
            Xf_te = np.hstack([Fs[te], Fc_te])
            
            clf = clf_fn()
            clf.fit(Xf_tr, y_all[tr])
            scores.append(clf.score(Xf_te, y_all[te]))
        
        results.append((name, np.mean(scores), np.std(scores)))
        print(f"  {name}: {np.mean(scores):.1%} +/- {np.std(scores):.1%}")
    
    best = max(results, key=lambda x: x[1])
    print(f"\n  Best: {best[0]} = {best[1]:.1%}")
    
    # LOSO
    print("\n[4] 留一被试LOSO...")
    n_per = [len(ep) for ep in epochs_list]
    loso_scores = []
    pos = 0
    for si in range(len(epochs_list)):
        tr_idx = np.concatenate([np.arange(pos, pos + n_per[j]) for j in range(len(epochs_list)) if j != si])
        te_idx = np.arange(pos, pos + n_per[si])
        pos += n_per[si]
        
        X_tr, X_te = X_all[tr_idx], X_all[te_idx]
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]
        
        # Re-extract band features for this split
        F_tr = extract_band(X_tr)
        F_te = extract_band(X_te)
        sc2 = StandardScaler()
        Fs_tr = np.nan_to_num(sc2.fit_transform(F_tr), nan=0, posinf=10, neginf=-10)
        Fs_te = np.nan_to_num(sc2.transform(F_te), nan=0, posinf=10, neginf=-10)
        
        # CSP
        Fc_te, Fc_tr = quick_csp(X_tr, y_tr, X_te)
        sc_csp = StandardScaler()
        Fc_tr = np.nan_to_num(sc_csp.fit_transform(Fc_tr), nan=0, posinf=5, neginf=-5)
        Fc_te = np.nan_to_num(sc_csp.transform(Fc_te), nan=0, posinf=5, neginf=-5)
        
        Xf_tr = np.hstack([Fs_tr, Fc_tr])
        Xf_te = np.hstack([Fs_te, Fc_te])
        
        clf = SVC(kernel='rbf', C=5)
        clf.fit(Xf_tr, y_tr)
        s = clf.score(Xf_te, y_te)
        loso_scores.append(s)
        print(f"  Subject {si+1}: {s:.1%} (n={len(X_te)})")
    
    print(f"\n  LOSO: {np.mean(loso_scores):.1%} +/- {np.std(loso_scores):.1%}")
    print("=" * 60)


if __name__ == '__main__':
    main()
