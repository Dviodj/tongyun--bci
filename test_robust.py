"""
BCICIV 2b - 稳健版
策略: 简单特征 + LDA（EEG最常用）+ within-subject CV
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['OMP_NUM_THREADS'] = '4'
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import mne

SFREQ = 250
DATA_DIR = r"D:\db\BCICIV_2b_gdf"


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


def bp_filt(s, f1, f2, fs=SFREQ):
    nyq = fs / 2
    b, a = butter(4, [f1/nyq, f2/nyq], btype='band')
    return filtfilt(b, a, s)


def simple_features(X):
    """简单稳健特征"""
    bands = [(8, 13), (13, 30)]
    F = []
    for ep in X:
        fv = []
        # Per-channel band power
        for ch in range(3):
            s = ep[ch]
            for f1, f2 in bands:
                filt_s = bp_filt(s, f1, f2)
                fv.append(np.var(filt_s))
            fv.extend([np.mean(s), np.std(s)])
        # C3/C4 asymmetry
        for f1, f2 in bands:
            c3 = np.var(bp_filt(ep[0], f1, f2)) + 1e-12
            c4 = np.var(bp_filt(ep[2], f1, f2)) + 1e-12
            fv.extend([np.log(c3/c4), (c3-c4)/(c3+c4)])
        F.append(fv)
    return np.array(F)


def csp_features(X_tr, y_tr, X_te, n_comp=4):
    """CSP with regularization"""
    from scipy.linalg import eigh
    cov1 = np.mean([np.cov(x) for x, y in zip(X_tr, y_tr) if y==0], axis=0)
    cov2 = np.mean([np.cov(x) for x, y in zip(X_tr, y_tr) if y==1], axis=0)
    cov_sum = cov1 + cov2 + 1e-5 * np.eye(3)
    try:
        lam, W = eigh(cov2, cov_sum)
    except:
        return np.zeros((len(X_te), n_comp*2)), np.zeros((len(X_tr), n_comp*2))
    idx = np.argsort(lam)[::-1]
    W = W[:, idx[:n_comp*2]]
    def project(X, W):
        proj = np.tensordot(X, W, axes=([1],[0]))
        var = np.var(proj, axis=2)
        var = var / (var.sum(axis=1, keepdims=True) + 1e-10)
        return np.log(var + 1e-10)
    return project(X_te, W), project(X_tr, W)


def main():
    print("=" * 60)
    print("BCICIV 2b - 稳健评估版")
    print("=" * 60)
    
    # Load
    print("\n[1] Loading...")
    epochs_list = []
    for fp in sorted(Path(DATA_DIR).glob("*T.gdf")):
        try:
            ep = load_subject(str(fp))
            epochs_list.append(ep)
            print(f"  {fp.name}: {len(ep)} epochs")
        except:
            print(f"  {fp.name}: ERROR")
    
    X_all = np.vstack([e.get_data() for e in epochs_list])
    y_all = np.concatenate([e.events[:, 2] for e in epochs_list])
    n_per = [len(ep) for ep in epochs_list]
    print(f"\nTotal: {X_all.shape} - L={sum(y_all==0)}, R={sum(y_all==1)}")
    
    # Within-subject 5-fold CV (standard for BCICIV)
    print("\n[2] Within-subject 5折CV...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # For each subject, do within-subject CV
    within_cv_scores = []
    for si in range(len(epochs_list)):
        X_s = epochs_list[si].get_data()
        y_s = epochs_list[si].events[:, 2]
        
        subj_scores = []
        for tr, te in cv.split(X_s, y_s):
            # Features
            F_tr = simple_features(X_s[tr])
            F_te = simple_features(X_s[te])
            # CSP
            Fc_te, Fc_tr = csp_features(X_s[tr], y_s[tr], X_s[te])
            # Combine
            sc = StandardScaler()
            F_tr = np.nan_to_num(sc.fit_transform(np.hstack([F_tr, Fc_tr])), nan=0, posinf=10, neginf=-10)
            F_te = np.nan_to_num(sc.transform(np.hstack([F_te, Fc_te])), nan=0, posinf=10, neginf=-10)
            
            # LDA (best for EEG)
            clf = LDA()
            clf.fit(F_tr, y_s[tr])
            subj_scores.append(clf.score(F_te, y_s[te]))
        
        within_cv_scores.append(np.mean(subj_scores))
        print(f"  Subject {si+1}: {within_cv_scores[-1]:.1%}")
    
    print(f"\n  Within-subject Mean: {np.mean(within_cv_scores):.1%} +/- {np.std(within_cv_scores):.1%}")
    
    # Cross-subject (LOSO) - this is the hard one
    print("\n[3] Cross-subject LOSO...")
    loso_scores = []
    pos = 0
    for si in range(len(epochs_list)):
        tr_idx = np.concatenate([np.arange(pos, pos+n_per[j]) for j in range(len(epochs_list)) if j != si])
        te_idx = np.arange(pos, pos+n_per[si])
        pos += n_per[si]
        
        X_tr, X_te = X_all[tr_idx], X_all[te_idx]
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]
        
        # Simple features (less overfitting than complex ones)
        F_tr = simple_features(X_tr)
        F_te = simple_features(X_te)
        
        sc = StandardScaler()
        F_tr = np.nan_to_num(sc.fit_transform(F_tr), nan=0, posinf=10, neginf=-10)
        F_te = np.nan_to_num(sc.transform(F_te), nan=0, posinf=10, neginf=-10)
        
        # LDA (more robust than SVM)
        clf = LDA()
        clf.fit(F_tr, y_tr)
        s = clf.score(F_te, y_te)
        loso_scores.append(s)
        print(f"  Subject {si+1}: {s:.1%}")
    
    print(f"\n  Cross-subject LOSO: {np.mean(loso_scores):.1%} +/- {np.std(loso_scores):.1%}")
    print("=" * 60)


if __name__ == '__main__':
    main()
