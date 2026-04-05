"""
BCICIV 2b - 深度优化版
策略: 时间窗口滑动 + 多频段 + 伪共平均参考 + 深度学习
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['OMP_NUM_THREADS'] = '1'
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.signal import butter, filtfilt, welch
import mne

DATA_DIR = r"D:\db\BCICIV_2b_gdf"
SFREQ = 250


def load_subject(fp):
    raw = mne.io.read_raw_gdf(fp, preload=True, verbose=False)
    ch_map = {ch: ch.split(':')[1] if ':' in ch else ch for ch in raw.ch_names}
    raw.rename_channels(ch_map)
    raw.pick(['C3', 'Cz', 'C4'])
    raw.filter(0.5, 45, method='iir')
    # No average reference yet - handle manually
    raw.set_eeg_reference('average', projection=True)
    raw.apply_proj()
    evs_all, _ = mne.events_from_annotations(raw, verbose=False)
    mask = np.isin(evs_all[:, 2], [10, 11])
    evs = evs_all[mask].copy()
    evs[:, 2] = np.where(evs[:, 2] == 10, 0, 1)
    ep = mne.Epochs(raw, evs, tmin=-0.5, tmax=4.0, baseline=(None, 0), preload=True, verbose=False)
    return ep


def butter_bandpass(low, high, fs, order=4):
    nyq = fs / 2
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return b, a


def extract_deep_features(X, fs=SFREQ):
    """深度特征: 多时间窗口 + 多频段"""
    n_epochs, n_ch, n_times = X.shape
    
    # Bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),     # mu rhythm
        'beta_l': (13, 20),   # low beta
        'beta_h': (20, 30),   # high beta
        'gamma': (30, 45),
    }
    
    # Time windows (after cue at sample 125 = 0.5s)
    windows = {
        'early': (125, 500),      # 0.5-2s
        'mid': (500, 875),         # 2-3.5s
        'late': (875, 1125),       # 3.5-4.5s
    }
    
    # Pre-compute filter coefficients
    filts = {}
    for bname, (f1, f2) in bands.items():
        filts[bname] = butter_bandpass(f1, f2, fs)
    
    features = []
    for ep_i in range(n_epochs):
        fv = []
        
        # Per-channel, per-band, per-window features
        for ch in range(n_ch):
            for bname, (b, a) in filts.items():
                sig = X[ep_i, ch]
                # Filtered signal
                filt_sig = filtfilt(b, a, sig)
                
                # Per-window power
                for wname, (w_start, w_end) in windows.items():
                    window_data = filt_sig[w_start:w_end]
                    p = np.var(window_data)
                    fv.extend([10 * np.log10(p + 1e-12), p])
        
        # C3 vs C4 asymmetry per band, per window (most discriminative!)
        for bname, (b, a) in filts.items():
            sig_c3 = X[ep_i, 0]
            sig_c4 = X[ep_i, 2]
            fc3 = filtfilt(b, a, sig_c3)
            fc4 = filtfilt(b, a, sig_c4)
            
            for wname, (w_start, w_end) in windows.items():
                p_c3 = np.var(fc3[w_start:w_end]) + 1e-12
                p_c4 = np.var(fc4[w_start:w_end]) + 1e-12
                fv.extend([
                    np.log(p_c3 / p_c4),           # log ratio
                    (p_c3 - p_c4) / (p_c3 + p_c4), # CRD (change ratio detector)
                    p_c3 - p_c4,
                ])
        
        # Spatial pattern: C3-Cz and C4-Cz
        for bname, (b, a) in filts.items():
            if bname not in ['alpha', 'beta_l', 'beta_h']:
                continue
            sig_cz = X[ep_i, 1]
            fc3 = filtfilt(b, a, sig_c3)
            fcz = filtfilt(b, a, sig_cz)
            fc4 = filtfilt(b, a, sig_c4)
            
            for wname, (w_start, w_end) in windows.items():
                fv.extend([
                    np.var(fc3[w_start:w_end] - fcz[w_start:w_end]),
                    np.var(fc4[w_start:w_end] - fcz[w_start:w_end]),
                ])
        
        # Peak amplitude features
        for ch in range(n_ch):
            sig = X[ep_i, ch]
            fv.extend([
                np.mean(sig),
                np.std(sig),
                np.max(np.abs(sig[125:875])),  # max abs in active window
            ])
        
        features.append(fv)
    
    return np.array(features)


def run_cv_with_features(epochs_list, C=1.0, n_splits=5):
    """Run CV with deep features"""
    from mne.decoding import CSP
    
    X_all = np.vstack([e.get_data() for e in epochs_list])
    y_all = np.concatenate([e.events[:, 2] for e in epochs_list])
    
    print(f"  Data: {X_all.shape}")
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores_csp = []
    scores_feat = []
    scores_fusion = []
    
    for fold_i, (tr_idx, te_idx) in enumerate(cv.split(X_all, y_all)):
        X_tr, X_te = X_all[tr_idx], X_all[te_idx]
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]
        
        # CSP features
        csp = CSP(n_components=6, reg='ledoit_wolf', log=True)
        csp.fit(X_tr, y_tr)
        Fc_tr = csp.transform(X_tr)
        Fc_te = csp.transform(X_te)
        sc_csp = StandardScaler()
        Fc_tr = sc_csp.fit_transform(Fc_tr)
        Fc_te = sc_csp.transform(Fc_te)
        
        # Deep features
        print(f"    Extracting features fold {fold_i+1}...", flush=True)
        F_tr = extract_deep_features(X_tr)
        F_te = extract_deep_features(X_te)
        sc_feat = StandardScaler()
        F_tr = np.nan_to_num(sc_feat.fit_transform(F_tr), nan=0, posinf=10, neginf=-10)
        F_te = np.nan_to_num(sc_feat.transform(F_te), nan=0, posinf=10, neginf=-10)
        
        # Fusion
        Xf_tr = np.hstack([Fc_tr, F_tr])
        Xf_te = np.hstack([Fc_te, F_te])
        
        # Classifiers
        clf_csp = SVC(kernel='rbf', C=C)
        clf_csp.fit(Fc_tr, y_tr)
        scores_csp.append(clf_csp.score(Fc_te, y_te))
        
        clf_feat = SVC(kernel='rbf', C=C)
        clf_feat.fit(F_tr, y_tr)
        scores_feat.append(clf_feat.score(F_te, y_te))
        
        # Simple ensemble: average SVM predictions
        clf1 = SVC(kernel='rbf', C=C)
        clf2 = LDA()
        clf1.fit(F_tr, y_tr)
        clf2.fit(F_tr, y_tr)
        pred1 = clf1.predict(F_te)
        pred2 = clf2.predict(F_te)
        pred_fusion = (pred1 + pred2) // 2  # majority vote
        scores_fusion.append(accuracy_score(y_te, pred_fusion))
        scores_fusion.append(accuracy_score(y_te, pred_fusion))
        
        print(f"  Fold {fold_i+1}: CSP={scores_csp[-1]:.1%} Feat={scores_feat[-1]:.1%} Fusion={scores_fusion[-1]:.1%}")
    
    return {
        'csp': (np.mean(scores_csp), np.std(scores_csp)),
        'feat': (np.mean(scores_feat), np.std(scores_feat)),
        'fusion': (np.mean(scores_fusion), np.std(scores_fusion)),
    }


def main():
    print("=" * 60)
    print("BCICIV 2b - 深度优化版")
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
    
    if not epochs_list:
        print("No data!")
        return
    
    # Test different settings
    print("\n[2] 5折交叉验证...")
    results = {}
    
    for C in [0.1, 1, 10]:
        print(f"\n  === C={C} ===")
        r = run_cv_with_features(epochs_list, C=C)
        results[C] = r
        print(f"  CSP:    {r['csp'][0]:.1%} +/- {r['csp'][1]:.1%}")
        print(f"  Feat:   {r['feat'][0]:.1%} +/- {r['feat'][1]:.1%}")
        print(f"  Fusion: {r['fusion'][0]:.1%} +/- {r['fusion'][1]:.1%}")
    
    # Best
    best = max(results.items(), key=lambda x: x[1]['fusion'][0])
    print(f"\n  Best: C={best[0]} Fusion={best[1]['fusion'][0]:.1%}")
    print("=" * 60)


if __name__ == '__main__':
    main()
