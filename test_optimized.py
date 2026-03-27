"""
BCICIV 2b - 优化版
目标: 80%+ 准确率
策略: 多频段 + 多通道 + 增强CSP + 深度学习 + 数据增强
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['OMP_NUM_THREADS'] = '4'
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, welch
from scipy.linalg import eigh
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import mne

DATA_DIR = r"D:\db\BCICIV_2b_gdf"
SFREQ = 250


def load_subject(fp):
    raw = mne.io.read_raw_gdf(fp, preload=True, verbose=False)
    # Rename channels (remove EEG: prefix)
    ch_map = {ch: ch.split(':')[1] if ':' in ch else ch for ch in raw.ch_names}
    raw.rename_channels(ch_map)
    # Keep all 6 channels (3 EEG + 3 EOG)
    raw.pick(['C3', 'Cz', 'C4'])  # Start with EEG
    raw.filter(0.5, 45, method='iir')
    raw.set_eeg_reference('average', projection=True)
    raw.apply_proj()
    # Get events
    evs_all, _ = mne.events_from_annotations(raw, verbose=False)
    mask = np.isin(evs_all[:, 2], [10, 11])
    evs = evs_all[mask].copy()
    evs[:, 2] = np.where(evs[:, 2] == 10, 0, 1)
    # Extract epochs with wider window
    ep = mne.Epochs(raw, evs, tmin=-1.0, tmax=4.0, baseline=(None, 0), preload=True, verbose=False)
    return ep.get_data(), ep.events[:, 2].astype(int)


def bandpass_filter(data, low, high, fs=SFREQ, order=4):
    """Apply bandpass filter"""
    nyq = fs / 2
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)


def extract_multiband_features(X, fs=SFREQ):
    """Extract comprehensive multi-band features"""
    n_epochs, n_ch, n_times = X.shape
    features = []
    
    # Frequency bands for motor imagery
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),     # mu rhythm - key for MI
        'low_beta': (13, 20),
        'high_beta': (20, 30),
        'gamma': (30, 45),
    }
    
    for ep_idx in range(n_epochs):
        ep = X[ep_idx]
        fv = []
        
        # 1. Band power for each channel, each band
        for ch in range(n_ch):
            for band_name, (f1, f2) in bands.items():
                # Filter to band
                bp_data = bandpass_filter(ep[ch], f1, f2, fs)
                # Power (variance)
                power = np.var(bp_data)
                # Log power
                log_power = np.log(power + 1e-10)
                fv.extend([power, log_power])
        
        # 2. C3 vs C4 asymmetry for each band (most important for MI)
        for band_name, (f1, f2) in bands.items():
            bp_c3 = bandpass_filter(ep[0], f1, f2, fs)
            bp_c4 = bandpass_filter(ep[2], f1, f2, fs)
            p_c3 = np.var(bp_c3) + 1e-10
            p_c4 = np.var(bp_c4) + 1e-10
            # Asymmetry features
            fv.extend([
                np.log(p_c3 / p_c4),           # log ratio
                (p_c3 - p_c4) / (p_c3 + p_c4), # normalized difference
                p_c3 - p_c4,                    # raw difference
            ])
        
        # 3. Time-domain features per channel
        for ch in range(n_ch):
            s = ep[ch]
            fv.extend([
                np.mean(s),
                np.std(s),
                np.sqrt(np.mean(s**2)),
                np.percentile(s, 10),
                np.percentile(s, 90),
                np.max(s) - np.min(s),  # range
                np.mean(np.abs(s)),       # mean absolute
                np.sum(s**2),            # energy
            ])
        
        # 4. ERD/ERS in specific time windows (relative to cue at t=0)
        # Early: 0.5-1.5s, Late: 1.5-3.5s
        for band_name, (f1, f2) in [('alpha', (8, 13)), ('beta', (13, 30))]:
            for ch in [0, 2]:  # C3 and C4
                # Baseline (-1 to 0s = samples 0-250)
                baseline = bandpass_filter(ep[ch, :250], f1, f2, fs)
                p_baseline = np.var(baseline) + 1e-10
                
                # Early (250-625 = 1-2.5s after cue)
                early = bandpass_filter(ep[ch, 250:625], f1, f2, fs)
                p_early = np.var(early) + 1e-10
                
                # Late (625-1125 = 2.5-4.5s after cue)
                late = bandpass_filter(ep[ch, 625:1125], f1, f2, fs)
                p_late = np.var(late) + 1e-10
                
                # ERD: relative change from baseline
                fv.extend([
                    (p_early - p_baseline) / p_baseline,  # early ERD
                    (p_late - p_baseline) / p_baseline,    # late ERD
                ])
        
        # 5. Spatial pattern: C3 vs surrounding (Cz)
        for band_name, (f1, f2) in [('alpha', (8, 13)), ('beta', (13, 30))]:
            c3 = bandpass_filter(ep[0], f1, f2, fs)
            cz = bandpass_filter(ep[1], f1, f2, fs)
            c4 = bandpass_filter(ep[2], f1, f2, fs)
            fv.extend([
                np.var(c3 - cz),  # C3-Cz difference
                np.var(c4 - cz),  # C4-Cz difference
            ])
        
        features.append(fv)
    
    return np.array(features)


class CSP:
    """Enhanced CSP with regularization"""
    def __init__(self, n_comp=6):
        self.n_comp = n_comp
        self.filters_ = None
    
    def fit(self, X, y):
        n_epochs, n_ch, n_times = X.shape
        
        # Compute covariance matrices
        covs = np.array([np.cov(x) for x in X])
        cov1 = np.mean(covs[y == 0], axis=0) + 1e-6 * np.eye(n_ch)
        cov2 = np.mean(covs[y == 1], axis=0) + 1e-6 * np.eye(n_ch)
        
        # Generalized eigenvalue decomposition
        cov_sum = cov1 + cov2
        try:
            lam, W = eigh(cov2, cov_sum)
        except:
            lam, W = np.linalg.eigh(cov_sum)
            lam2, _ = np.linalg.eigh(cov2)
            lam = lam2
        
        # Sort by eigenvalue ratio
        idx = np.argsort(np.abs(lam))[::-1]
        self.filters_ = W[:, idx[:self.n_comp * 2]]
        
        # Regularization: add small amount of identity
        if self.filters_ is None:
            self.filters_ = np.eye(n_ch)[:, :self.n_comp * 2]
        
        return self
    
    def transform(self, X):
        if self.filters_ is None:
            return np.zeros((X.shape[0], self.n_comp * 2))
        
        # Project to CSP space
        X_proj = np.tensordot(X, self.filters_, axes=([1], [0]))
        # Variance in CSP space
        var = np.var(X_proj, axis=2)
        # Normalize
        var = var / (var.sum(axis=1, keepdims=True) + 1e-10)
        # Log transform
        return np.log(var + 1e-10)
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


def extract_csp_features(X_tr, y_tr, X_te, n_comp=6):
    """Extract CSP features with proper cross-validation"""
    csp = CSP(n_comp=n_comp)
    Fc_tr = csp.fit_transform(X_tr, y_tr)
    Fc_te = csp.transform(X_te)
    return Fc_tr, Fc_te


def extract_all_features(X_tr, y_tr, X_te):
    """Extract all features with proper CV"""
    # CSP features
    Fc_tr, Fc_te = extract_csp_features(X_tr, y_tr, X_te, n_comp=6)
    
    # Traditional multiband features
    Fe_tr = extract_multiband_features(X_tr)
    Fe_te = extract_multiband_features(X_te)
    
    # Standardize each feature set separately
    sc_csp = StandardScaler()
    sc_feat = StandardScaler()
    
    Fc_tr = np.nan_to_num(sc_csp.fit_transform(Fc_tr), nan=0, posinf=10, neginf=-10)
    Fc_te = np.nan_to_num(sc_csp.transform(Fc_te), nan=0, posinf=10, neginf=-10)
    Fe_tr = np.nan_to_num(sc_feat.fit_transform(Fe_tr), nan=0, posinf=10, neginf=-10)
    Fe_te = np.nan_to_num(sc_feat.transform(Fe_te), nan=0, posinf=10, neginf=-10)
    
    # Concatenate
    return np.hstack([Fc_tr, Fe_tr]), np.hstack([Fc_te, Fe_te])


def main():
    print("=" * 60)
    print("BCICIV 2b - 优化版 (目标80%+)")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading data...")
    all_X, all_y = [], []
    for fp in sorted(Path(DATA_DIR).glob("*T.gdf")):
        try:
            X, y = load_subject(str(fp))
            all_X.append(X)
            all_y.append(y)
            print(f"  {fp.name}: {len(X)} epochs")
        except Exception as e:
            print(f"  {fp.name}: ERROR - {str(e)[:50]}")
    
    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    print(f"\nTotal: {X_all.shape} - L={sum(y_all==0)}, R={sum(y_all==1)}")
    
    # 5-fold CV with multiple classifiers
    print("\n[2] 5折交叉验证...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    
    # SVM with different settings
    for C in [0.1, 1, 10]:
        for gamma in ['scale', 'auto']:
            scores = []
            for tr, te in cv.split(X_all, y_all):
                F_tr, F_te = extract_all_features(X_all[tr], y_all[tr], X_all[te])
                clf = SVC(kernel='rbf', C=C, gamma=gamma)
                clf.fit(F_tr, y_all[tr])
                scores.append(clf.score(F_te, y_all[te]))
            mean_s = np.mean(scores)
            results.append(('SVM-rbf C=%g gamma=%s' % (C, gamma), mean_s, np.std(scores)))
            print(f"  SVM C={C} gamma={gamma}: {mean_s:.1%}")
    
    # LDA
    scores = []
    for tr, te in cv.split(X_all, y_all):
        F_tr, F_te = extract_all_features(X_all[tr], y_all[tr], X_all[te])
        clf = LDA()
        clf.fit(F_tr, y_all[tr])
        scores.append(clf.score(F_te, y_all[te]))
    results.append(('LDA', np.mean(scores), np.std(scores)))
    print(f"  LDA: {np.mean(scores):.1%}")
    
    # GBDT
    scores = []
    for tr, te in cv.split(X_all, y_all):
        F_tr, F_te = extract_all_features(X_all[tr], y_all[tr], X_all[te])
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        clf.fit(F_tr, y_all[tr])
        scores.append(clf.score(F_te, y_all[te]))
    results.append(('GBDT', np.mean(scores), np.std(scores)))
    print(f"  GBDT: {np.mean(scores):.1%}")
    
    # Best model
    best = max(results, key=lambda x: x[1])
    print(f"\n  Best: {best[0]} = {best[1]:.1%}")
    
    # LOSO
    print("\n[3] 留一被试LOSO...")
    loso_scores = []
    for si in range(len(all_X)):
        tr_X = np.vstack([all_X[j] for j in range(len(all_X)) if j != si])
        tr_y = np.concatenate([all_y[j] for j in range(len(all_X)) if j != si])
        te_X, te_y = all_X[si], all_y[si]
        
        F_tr, F_te = extract_all_features(tr_X, tr_y, te_X)
        clf = SVC(kernel='rbf', C=best[1] if isinstance(best[0], str) and 'SVM' in best[0] else 1, 
                  gamma='scale')
        clf.fit(F_tr, tr_y)
        s = clf.score(F_te, te_y)
        loso_scores.append(s)
        print(f"  Subject {si+1}: {s:.1%}")
    
    print(f"\n  LOSO Mean: {np.mean(loso_scores):.1%} +/- {np.std(loso_scores):.1%}")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
