"""
BCICIV 2b - 优化版 v3
使用MNE内置CSP + 多频段特征
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


def multiband_psd(epochs, bands, fs=SFREQ):
    """Compute power in multiple bands using MNE's psd_array_welch"""
    from mne.time_frequency import psd_array_welch
    X = epochs.get_data()
    n, nc, nt = X.shape
    F = []
    
    for i in range(n):
        fv = []
        for ch in range(nc):
            s = X[i, ch]
            for (f1, f2) in bands:
                f, p = psd_array_welch(s, fs, fmin=f1, fmax=f2, n_fft=fs//2)
                bp = np.mean(p)
                fv.extend([10 * np.log10(bp + 1e-12), bp])
            fv.extend([np.mean(s), np.std(s), np.sqrt(np.mean(s**2))])
        # C3/C4 asymmetry per band
        for (f1, f2) in bands:
            _, p3 = psd_array_welch(X[i, 0], fs, fmin=f1, fmax=f2, n_fft=fs//2)
            _, p4 = psd_array_welch(X[i, 2], fs, fmin=f1, fmax=f2, n_fft=fs//2)
            c3 = np.mean(p3) + 1e-12
            c4 = np.mean(p4) + 1e-12
            fv.extend([np.log(c3/c4), (c3-c4)/(c3+c4)])
        F.append(fv)
    return np.array(F)


def extract_csp(X_tr, y_tr, X_te, n_comp=2):
    """CSP using MNE"""
    n_tr = len(X_tr)
    info = mne.create_info(['C3', 'Cz', 'C4'], SFREQ, ch_types='eeg')
    ev_tr = np.column_stack([np.arange(n_tr), np.zeros(n_tr, int), y_tr])
    
    ep_tr = mne.EpochsArray(X_tr, info, events=ev_tr, tmin=-0.5, verbose=False)
    csp = mne.decoding.CSP(n_components=n_comp * 2, reg='ledoit_wolf', log=True)
    csp.fit(X_tr, y_tr)
    
    Fc_tr = csp.transform(X_tr)
    Fc_te = csp.transform(X_te)
    return Fc_tr, Fc_te


def run_cv(all_X, all_y, cv, desc):
    """Run cross-validation with proper feature extraction"""
    BANDS = [(8, 13), (13, 30), (6, 13)]
    results = []
    
    for name, clf_fn in [
        ("LDA", lambda: LDA()),
        ("SVM-rbf C=1", lambda: SVC(kernel='rbf', C=1, gamma='scale')),
        ("SVM-rbf C=5", lambda: SVC(kernel='rbf', C=5, gamma='scale')),
        ("SVM-rbf C=10", lambda: SVC(kernel='rbf', C=10, gamma='scale')),
        ("SVM-lin C=1", lambda: SVC(kernel='linear', C=1)),
    ]:
        scores = []
        for tr, te in cv.split(all_X, all_y):
            X_tr, X_te = all_X[tr], all_X[te]
            y_tr, y_te = all_y[tr], all_y[te]
            
            # Band power features
            info = mne.create_info(['C3', 'Cz', 'C4'], SFREQ, ch_types='eeg')
            ep_tr = mne.EpochsArray(X_tr, info, verbose=False)
            Fb_tr = multiband_psd(ep_tr, BANDS)
            ep_te = mne.EpochsArray(X_te, info, verbose=False)
            Fb_te = multiband_psd(ep_te, BANDS)
            
            # CSP features
            Fc_tr, Fc_te = extract_csp(X_tr, y_tr, X_te)
            
            # Combine and scale
            sc1 = StandardScaler(); sc2 = StandardScaler()
            Fb_tr = np.nan_to_num(sc1.fit_transform(Fb_tr), nan=0, posinf=10, neginf=-10)
            Fb_te = np.nan_to_num(sc1.transform(Fb_te), nan=0, posinf=10, neginf=-10)
            Fc_tr = np.nan_to_num(sc2.fit_transform(Fc_tr), nan=0, posinf=5, neginf=-5)
            Fc_te = np.nan_to_num(sc2.transform(Fc_te), nan=0, posinf=5, neginf=-5)
            
            Xf_tr = np.hstack([Fb_tr, Fc_tr])
            Xf_te = np.hstack([Fb_te, Fc_te])
            
            clf = clf_fn()
            clf.fit(Xf_tr, y_tr)
            scores.append(clf.score(Xf_te, y_te))
        
        results.append((name, np.mean(scores), np.std(scores)))
        print(f"  {name}: {np.mean(scores):.1%} +/- {np.std(scores):.1%}")
    
    return results


def main():
    print("=" * 60)
    print("BCICIV 2b - 优化版 v3")
    print("=" * 60)
    
    # Load
    print("\n[1] Loading...")
    epochs_list, labels_list = [], []
    for fp in sorted(Path(DATA_DIR).glob("*T.gdf")):
        try:
            ep = load_subject(str(fp))
            epochs_list.append(ep)
            labels_list.append(ep.events[:, 2])
            print(f"  {fp.name}: {len(ep)} epochs")
        except Exception as e:
            print(f"  {fp.name}: ERROR - {str(e)[:40]}")
    
    X_all = np.vstack([ep.get_data() for ep in epochs_list])
    y_all = np.concatenate(labels_list)
    print(f"\nTotal: {X_all.shape} - L={sum(y_all==0)}, R={sum(y_all==1)}")
    
    # 5-fold CV
    print("\n[2] 5折交叉验证...")
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = run_cv(X_all, y_all, cv5, "5-fold")
    
    best = max(results, key=lambda x: x[1])
    print(f"\n  Best: {best[0]} = {best[1]:.1%}")
    
    # LOSO
    print("\n[3] 留一被试LOSO...")
    loso_scores = []
    n_per = [len(ep) for ep in epochs_list]
    
    for si in range(len(epochs_list)):
        tr_X = np.vstack([epochs_list[j].get_data() for j in range(len(epochs_list)) if j != si])
        tr_y = np.concatenate([epochs_list[j].events[:, 2] for j in range(len(epochs_list)) if j != si])
        te_X = epochs_list[si].get_data()
        te_y = epochs_list[si].events[:, 2]
        
        BANDS = [(8, 13), (13, 30), (6, 13)]
        
        # Band features
        info = mne.create_info(['C3', 'Cz', 'C4'], SFREQ, ch_types='eeg')
        ep_tr = mne.EpochsArray(tr_X, info, verbose=False)
        Fb_tr = multiband_psd(ep_tr, BANDS)
        ep_te = mne.EpochsArray(te_X, info, verbose=False)
        Fb_te = multiband_psd(ep_te, BANDS)
        
        # CSP
        Fc_tr, Fc_te = extract_csp(tr_X, tr_y, te_X)
        
        # Scale
        sc1 = StandardScaler(); sc2 = StandardScaler()
        Fb_tr = np.nan_to_num(sc1.fit_transform(Fb_tr), nan=0, posinf=10, neginf=-10)
        Fb_te = np.nan_to_num(sc1.transform(Fb_te), nan=0, posinf=10, neginf=-10)
        Fc_tr = np.nan_to_num(sc2.fit_transform(Fc_tr), nan=0, posinf=5, neginf=-5)
        Fc_te = np.nan_to_num(sc2.transform(Fc_te), nan=0, posinf=5, neginf=-5)
        
        Xf_tr = np.hstack([Fb_tr, Fc_tr])
        Xf_te = np.hstack([Fb_te, Fc_te])
        
        clf = SVC(kernel='rbf', C=5, gamma='scale')
        clf.fit(Xf_tr, tr_y)
        s = clf.score(Xf_te, te_y)
        loso_scores.append(s)
        print(f"  Subject {si+1}: {s:.1%} (n={len(te_X)})")
    
    print(f"\n  LOSO: {np.mean(loso_scores):.1%} +/- {np.std(loso_scores):.1%}")
    print("=" * 60)


if __name__ == '__main__':
    main()
