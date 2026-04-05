"""
BCICIV 2b - 快速预滤波版
策略: 先预滤波所有频段，再批量提取特征
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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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


def csp_features(X_tr, y_tr, X_te, n_comp=2):
    """CSP using MNE's implementation for correctness"""
    n_tr = len(X_tr)
    n_te = len(X_te)
    
    # Reshape for MNE: (n_epochs, n_channels, n_times)
    epochs_tr = mne.EpochsArray(X_tr, mne.create_info(['C3', 'Cz', 'C4'], SFREQ, ch_types='eeg'), 
                                 events=np.column_stack([np.arange(n_tr), np.zeros(n_tr, int), y_tr]),
                                 tmin=-0.5, verbose=False)
    csp = mne.decoding.CSP(n_components=n_comp * 2, reg='ledoit_wolf', log=True)
    csp.fit(epochs_tr, y_tr)
    Fc_tr = csp.transform(X_tr)
    Fc_te = csp.transform(X_te)
    return Fc_tr, Fc_te


def band_feat(X):
    """Fast band power features from pre-filtered data"""
    n, nc, nt = X.shape
    F = []
    for i in range(n):
        f = []
        for ch in range(nc):
            s = X[i, ch]
            for p in [np.var(s)]:
                f.append(10 * np.log10(p + 1e-12))
                f.append(p)
            f.extend([np.mean(s), np.std(s)])
        # C3/C4 asymmetry
        for ch1, ch2 in [(0, 2)]:
            c3 = np.var(X[i, ch1]) + 1e-12
            c4 = np.var(X[i, ch2]) + 1e-12
            f.extend([np.log(c3/c4), (c3-c4)/(c3+c4)])
        F.append(f)
    return np.array(F)


class BandFilter:
    """预滤波所有频段"""
    def __init__(self, bands, fs=SFREQ, order=4):
        self.bands = bands
        self.fs = fs
        self.coeffs = {}
        for name, (f1, f2) in bands.items():
            nyq = fs / 2
            b, a = butter(order, [f1/nyq, f2/nyq], btype='band')
            self.coeffs[name] = (b, a)

    def filt(self, data, band):
        b, a = self.coeffs[band]
        if data.ndim == 1:
            return filtfilt(b, a, data)
        return np.array([filtfilt(b, a, data[i]) for i in range(len(data))])

    def filt_batch(self, X, band):
        """批量滤波所有epoch"""
        b, a = self.coeffs[band]
        n, nc, nt = X.shape
        out = np.zeros((n, nc, nt))
        for ch in range(nc):
            for ep in range(n):
                out[ep, ch] = filtfilt(b, a, X[ep, ch])
        return out


def extract_features_from_filtered(X, bands):
    """从预滤波数据中快速提取特征"""
    n, nc, nt = X.shape
    F = []

    for i in range(n):
        ep = X[i]
        f = []

        # Band power features (per channel, per band)
        for ch in range(nc):
            for bname, bp_data in [(bname, ep[ch]) for bname, bp_data in [(bn, ep[ch]) for bn in bands.keys()]]:
                pass

        # Per channel, per band
        for ch in range(nc):
            for bname, bX in bands.items():
                p = np.var(bX[i, ch])
                f.append(10 * np.log10(p + 1e-12))  # dB
                f.append(p)  # linear

        # C3/C4 asymmetry per band
        for bname in bands.keys():
            c3 = np.var(bands[bname][i, 0]) + 1e-12
            c4 = np.var(bands[bname][i, 2]) + 1e-12
            f.extend([np.log(c3/c4), (c3-c4)/(c3+c4), c3-c4])

        # Time domain
        for ch in range(nc):
            s = ep[ch]
            f.extend([np.mean(s), np.std(s), np.sqrt(np.mean(s**2)),
                     np.percentile(s, 5), np.percentile(s, 95)])

        # ERD (mu 8-13Hz and beta 13-30Hz)
        for bname, t_range in [('mu', slice(125, 875)), ('beta', slice(125, 875))]:
            for ch in [0, 2]:
                p_active = np.var(bands[bname][i, ch, t_range])
                p_base = np.var(bands[bname][i, ch, :125])
                f.append((p_active - p_base) / (p_base + 1e-12))

        F.append(f)
    return np.array(F)


def main():
    print("=" * 60)
    print("BCICIV 2b - 预滤波快速版")
    print("=" * 60)

    BANDS = {
        'mu': (8, 13),
        'beta': (13, 30),
        'low_mu': (6, 13),
        'high_beta': (18, 25),
    }

    # Load all data
    print("\n[1] Loading...")
    all_X, all_y = [], []
    for fp in sorted(Path(DATA_DIR).glob("*T.gdf")):
        try:
            X, y = load_subject(str(fp))
            all_X.append(X)
            all_y.append(y)
            print(f"  {fp.name}: {len(X)}")
        except Exception as e:
            print(f"  {fp.name}: ERROR")

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    print(f"\nTotal: {X_all.shape}")

    # Pre-filter all data for each band
    print("\n[2] Pre-filtering bands...")
    bf = BandFilter(BANDS)
    filtered = {}
    for bname in BANDS:
        print(f"  {bname}...", flush=True)
        filtered[bname] = bf.filt_batch(X_all, bname)

    # Extract features from pre-filtered data
    print("\n[3] Feature extraction...")
    F = extract_features_from_filtered(X_all, filtered)
    print(f"  Shape: {F.shape}")

    # Standardize
    sc = StandardScaler()
    XF = np.nan_to_num(sc.fit_transform(F), nan=0, posinf=10, neginf=-10)

    # CSP features
    print("\n[4] CSP features...")
    csp = CSP(n_comp=6)
    Fc = csp.fit_transform(X_all, y_all)
    sc_csp = StandardScaler()
    Fc = np.nan_to_num(sc_csp.fit_transform(Fc), nan=0, posinf=5, neginf=-5)
    print(f"  CSP: {Fc.shape}")

    # Combine
    X_feat = np.hstack([XF, Fc])
    print(f"  Total: {X_feat.shape}")

    # 5-fold CV with proper feature extraction
    print("\n[5] 5折交叉验证...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    for name, clf in [
        ("LDA", LDA()),
        ("SVM C=1", SVC(kernel='rbf', C=1, gamma='scale')),
        ("SVM C=5", SVC(kernel='rbf', C=5, gamma='scale')),
        ("SVM C=10", SVC(kernel='rbf', C=10, gamma='scale')),
        ("SVM C=50", SVC(kernel='rbf', C=50, gamma='scale')),
    ]:
        scores = []
        for tr, te in cv.split(X_all, y_all):
            # CSP fit on train only
            csp_cv = CSP(n_comp=6)
            Fc_tr = csp_cv.fit_transform(X_all[tr], y_all[tr])
            Fc_te = csp_cv.transform(X_all[te])
            sc_c = StandardScaler()
            Fc_tr = np.nan_to_num(sc_c.fit_transform(Fc_tr), nan=0, posinf=5, neginf=-5)
            Fc_te = np.nan_to_num(sc_c.transform(Fc_te), nan=0, posinf=5, neginf=-5)

            # Band features
            sc_b = StandardScaler()
            F_tr = np.nan_to_num(sc_b.fit_transform(F[tr]), nan=0, posinf=10, neginf=-10)
            F_te = np.nan_to_num(sc_b.transform(F[te]), nan=0, posinf=10, neginf=-10)

            Xf_tr = np.hstack([F_tr, Fc_tr])
            Xf_te = np.hstack([F_te, Fc_te])

            clf.fit(Xf_tr, y_all[tr])
            scores.append(clf.score(Xf_te, y_all[te]))

        results.append((name, np.mean(scores), np.std(scores)))
        print(f"  {name}: {np.mean(scores):.1%} +/- {np.std(scores):.1%}")

    best_name = max(results, key=lambda x: x[1])[0]
    print(f"\n  Best: {best_name}")

    # LOSO
    print("\n[6] 留一被试LOSO...")
    loso = []
    for si in range(len(all_X)):
        tr_X = np.vstack([all_X[j] for j in range(len(all_X)) if j != si])
        tr_y = np.concatenate([all_y[j] for j in range(len(all_X)) if j != si])
        te_X, te_y = all_X[si], all_y[si]

        # CSP on train only
        csp_l = CSP(n_comp=6)
        Fc_tr = csp_l.fit_transform(tr_X, tr_y)
        Fc_te = csp_l.transform(te_X)
        sc_c = StandardScaler()
        Fc_tr = np.nan_to_num(sc_c.fit_transform(Fc_tr), nan=0, posinf=5, neginf=-5)
        Fc_te = np.nan_to_num(sc_c.transform(Fc_te), nan=0, posinf=5, neginf=-5)

        # Band features on train only
        F_all_sub = np.vstack(all_X)
        F_sub = extract_features_from_filtered(F_all_sub, filtered)
        tr_F = np.vstack([all_X[j] for j in range(len(all_X)) if j != si])
        te_F_sub = all_X[si]
        # Re-extract
        F_tr = extract_features(tr_X)
        F_te = extract_features(te_X)
        sc_b = StandardScaler()
        F_tr = np.nan_to_num(sc_b.fit_transform(F_tr), nan=0, posinf=10, neginf=-10)
        F_te = np.nan_to_num(sc_b.transform(F_te), nan=0, posinf=10, neginf=-10)

        Xf_tr = np.hstack([F_tr, Fc_tr])
        Xf_te = np.hstack([F_te, Fc_te])

        clf = SVC(kernel='rbf', C=5, gamma='scale')
        clf.fit(Xf_tr, tr_y)
        s = clf.score(Xf_te, te_y)
        loso.append(s)
        print(f"  S{si+1}: {s:.1%}")

    print(f"\n  LOSO: {np.mean(loso):.1%} +/- {np.std(loso):.1%}")
    print("=" * 60)


if __name__ == '__main__':
    main()
