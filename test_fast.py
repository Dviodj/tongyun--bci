"""
BCICIV 2b - 快速优化版
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['OMP_NUM_THREADS'] = '4'
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, welch
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
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
    return ep.get_data(), ep.events[:, 2].astype(int)


def bp_filt(s, f1, f2, fs=SFREQ, order=4):
    """Bandpass filter"""
    nyq = fs / 2
    b, a = butter(order, [f1/nyq, f2/nyq], btype='band')
    return filtfilt(b, a, s)


def extract_features(X):
    """Fast multi-band feature extraction"""
    n, n_ch, n_t = X.shape
    F = []
    
    # Bands
    bands = [(8, 13), (13, 30), (6, 13), (18, 25)]
    
    for i in range(n):
        ep = X[i]
        f = []
        
        # 1. Band power for each channel
        for ch in range(n_ch):
            s = ep[ch]
            for f1, f2 in bands:
                bp = bp_filt(s, f1, f2)
                p = np.var(bp)
                f.append(10 * np.log10(p + 1e-12))  # dB
                f.append(p)  # linear
            f.extend([np.mean(s), np.std(s), np.sqrt(np.mean(s**2))])
        
        # 2. C3/C4 asymmetry
        for f1, f2 in bands:
            c3 = np.var(bp_filt(ep[0], f1, f2)) + 1e-12
            c4 = np.var(bp_filt(ep[2], f1, f2)) + 1e-12
            f.extend([np.log(c3/c4), (c3-c4)/(c3+c4), c3-c4])
        
        # 3. ERD: alpha power suppression (0.5-3.5s vs baseline -0.5-0s)
        # Baseline: samples 0-125 (-0.5 to 0s)
        # Active: samples 125-875 (0 to 3s)
        for f1, f2 in [(8, 13), (13, 30)]:
            for ch in [0, 2]:
                bp_bl = np.var(bp_filt(ep[ch, :125], f1, f2))
                bp_ac = np.var(bp_filt(ep[ch, 125:875], f1, f2))
                f.append((bp_ac - bp_bl) / (bp_bl + 1e-12))
        
        F.append(f)
    return np.array(F)


def main():
    print("=" * 60)
    print("BCICIV 2b - 快速优化版")
    print("=" * 60)
    
    # Load
    print("\n[1] Loading...")
    all_X, all_y = [], []
    for fp in sorted(Path(DATA_DIR).glob("*T.gdf")):
        try:
            X, y = load_subject(str(fp))
            all_X.append(X); all_y.append(y)
            print(f"  {fp.name}: {len(X)}")
        except:
            print(f"  {fp.name}: ERROR")
    
    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    print(f"\nTotal: {X_all.shape}")
    
    # Extract all features ONCE
    print("\n[2] Feature extraction...")
    F_all = extract_features(X_all)
    print(f"  Features: {F_all.shape}")
    
    # Standardize
    sc = StandardScaler()
    XF = np.nan_to_num(sc.fit_transform(F_all), nan=0, posinf=10, neginf=-10)
    
    # 5-fold CV
    print("\n[3] 5折交叉验证...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    for name, clf in [
        ("LDA", LDA()),
        ("SVM C=1", SVC(kernel='rbf', C=1)),
        ("SVM C=5", SVC(kernel='rbf', C=5)),
        ("SVM C=10", SVC(kernel='rbf', C=10)),
        ("SVM-lin C=1", SVC(kernel='linear', C=1)),
    ]:
        scores = cross_val_score(clf, XF, y_all, cv=cv)
        results.append((name, scores.mean(), scores.std()))
        print(f"  {name}: {scores.mean():.1%} +/- {scores.std():.1%}")
    
    best = max(results, key=lambda x: x[1])
    print(f"\n  Best: {best[0]} = {best[1]:.1%}")
    
    # Per-subject CV
    print("\n[4] 留一被试LOSO...")
    loso = []
    for si in range(len(all_X)):
        tr_X = np.vstack([all_X[j] for j in range(len(all_X)) if j != si])
        tr_y = np.concatenate([all_y[j] for j in range(len(all_X)) if j != si])
        te_X, te_y = all_X[si], all_y[si]
        
        # Re-extract features
        F_tr = extract_features(tr_X)
        F_te = extract_features(te_X)
        sc2 = StandardScaler()
        F_tr = np.nan_to_num(sc2.fit_transform(F_tr), nan=0, posinf=10, neginf=-10)
        F_te = np.nan_to_num(sc2.transform(F_te), nan=0, posinf=10, neginf=-10)
        
        clf = SVC(kernel='rbf', C=5)
        clf.fit(F_tr, tr_y)
        s = clf.score(F_te, te_y)
        loso.append(s)
        print(f"  S{si+1}: {s:.1%}")
    
    print(f"\n  LOSO: {np.mean(loso):.1%} +/- {np.std(loso):.1%}")
    print("=" * 60)


if __name__ == '__main__':
    main()
