"""
BCICIV 2b - 严格验证 (精简版)
无数据泄露: CSP和Scaler都在每折训练集内拟合
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['OMP_NUM_THREADS'] = '1'
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from pathlib import Path
from scipy.signal import welch
from scipy.linalg import eigh
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import mne

DATA_DIR = r"D:\db\BCICIV_2b_gdf"
SFREQ = 250


def load_subject(fp):
    raw = mne.io.read_raw_gdf(fp, preload=True, verbose=False)
    ch_map = {ch: ch.split(':')[1] if ':' in ch else ch for ch in raw.ch_names}
    raw.rename_channels(ch_map)
    raw.pick(['C3', 'Cz', 'C4'])
    raw.filter(1, 40, method='iir')
    raw.set_eeg_reference('average', projection=True)
    raw.apply_proj()
    evs_all, _ = mne.events_from_annotations(raw, verbose=False)
    mask = np.isin(evs_all[:, 2], [10, 11])
    evs = evs_all[mask].copy()
    evs[:, 2] = np.where(evs[:, 2] == 10, 0, 1)
    ep = mne.Epochs(raw, evs, tmin=-0.5, tmax=3.5, baseline=(None, 0), preload=True, verbose=False)
    return ep.get_data(), ep.events[:, 2].astype(int)


class CSP:
    def __init__(self, n_comp=4):
        self.n_comp = n_comp
        self.filters_ = None

    def fit(self, X, y):
        covs = np.array([np.cov(x) for x in X])
        cov1 = np.mean(covs[y == 0], axis=0)
        cov2 = np.mean(covs[y == 1], axis=0)
        cov_r = cov1 + cov2 + 0.1 * np.eye(cov1.shape[0])
        lam, W = eigh(cov2, cov_r)
        idx = np.argsort(np.abs(lam - 0.5))[::-1]
        self.filters_ = W[:, idx[:self.n_comp * 2]]
        return self

    def transform(self, X):
        Xf = np.tensordot(X, self.filters_, axes=([1], [0]))
        var = np.var(Xf, axis=2)
        var = var / (var.sum(axis=1, keepdims=True) + 1e-10)
        return np.log(var + 1e-10)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


def feat(X):
    F = []
    for ep in X:
        f = []
        for ch in range(3):
            s = ep[ch]
            for f1, f2 in [(8, 13), (13, 30), (6, 13), (18, 25)]:
                try:
                    ff, pp = welch(s, fs=SFREQ, nperseg=min(64, len(s)//2))
                    idx = (ff >= f1) & (ff <= f2)
                    f.append(10 * np.log10(np.mean(pp[idx]) + 1e-12) if idx.sum() > 0 else -12)
                    f.append(np.mean(pp[idx]) if idx.sum() > 0 else 1e-12)
                except:
                    f.append(-12); f.append(1e-12)
            f.extend([np.mean(s), np.std(s), np.sqrt(np.mean(s**2)), np.percentile(s, 5), np.percentile(s, 95)])
        for f1, f2 in [(8, 13), (13, 30), (6, 13), (18, 25)]:
            try:
                ff, pp = welch(ep[0], fs=SFREQ, nperseg=64)
                c3 = np.mean(pp[(ff>=f1)&(ff<=f2)]) + 1e-12
                ff, pp = welch(ep[2], fs=SFREQ, nperseg=64)
                c4 = np.mean(pp[(ff>=f1)&(ff<=f2)]) + 1e-12
                f.extend([np.log(c3/c4), (c3-c4)/(c3+c4), c3-c4])
            except:
                f.extend([0, 0, 0])
        F.append(f)
    return np.array(F)


def extract(X_tr, y_tr, X_te):
    """正确流程: 只用训练数据拟合CSP和Scaler"""
    csp = CSP(n_comp=4)
    Fc_tr = csp.fit_transform(X_tr, y_tr)
    Fc_te = csp.transform(X_te)
    Fe_tr = feat(X_tr)
    Fe_te = feat(X_te)
    sc_c = StandardScaler(); sc_f = StandardScaler()
    Fc_tr = np.nan_to_num(sc_c.fit_transform(Fc_tr), nan=0, posinf=5, neginf=-5)
    Fc_te = np.nan_to_num(sc_c.transform(Fc_te), nan=0, posinf=5, neginf=-5)
    Fe_tr = np.nan_to_num(sc_f.fit_transform(Fe_tr), nan=0, posinf=5, neginf=-5)
    Fe_te = np.nan_to_num(sc_f.transform(Fe_te), nan=0, posinf=5, neginf=-5)
    return np.hstack([Fc_tr, Fe_tr]), np.hstack([Fc_te, Fe_te])


print("=" * 60)
print("BCICIV 2b - 严格验证 (无数据泄露)")
print("=" * 60)

# Load
print("\n[1] Loading data...")
all_X, all_y = [], []
for fp in sorted(Path(DATA_DIR).glob("*T.gdf")):
    try:
        X, y = load_subject(str(fp))
        all_X.append(X); all_y.append(y)
        print(f"  {fp.name}: {len(X)} epochs (L={sum(y==0)}, R={sum(y==1)})")
    except Exception as e:
        print(f"  {fp.name}: ERROR - {str(e)[:50]}")

X_all = np.vstack(all_X)
y_all = np.concatenate(all_y)
print(f"\nTotal: {X_all.shape} - L={sum(y_all==0)}, R={sum(y_all==1)}")

# 5-fold CV (strict)
print("\n[2] 5折交叉验证 (正确方法)")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for i, (tr, te) in enumerate(cv.split(X_all, y_all)):
    F_tr, F_te = extract(X_all[tr], y_all[tr], X_all[te])
    clf = SVC(kernel='rbf', C=5, gamma='scale')
    clf.fit(F_tr, y_all[tr])
    s = clf.score(F_te, y_all[te])
    scores.append(s)
    print(f"  Fold {i+1}: {s:.1%}")

print(f"\n  Mean: {np.mean(scores):.1%} +/- {np.std(scores):.1%}")

# LOSO
print("\n[3] 留一被试LOSO")
loso_scores = []
for si in range(len(all_X)):
    tr_X = np.vstack([all_X[j] for j in range(len(all_X)) if j != si])
    tr_y = np.concatenate([all_y[j] for j in range(len(all_X)) if j != si])
    te_X, te_y = all_X[si], all_y[si]
    F_tr, F_te = extract(tr_X, tr_y, te_X)
    clf = SVC(kernel='rbf', C=5, gamma='scale')
    clf.fit(F_tr, tr_y)
    s = clf.score(F_te, te_y)
    loso_scores.append(s)
    print(f"  Subject {si+1}: {s:.1%} (n={len(te_X)})")

print(f"\n  Mean: {np.mean(loso_scores):.1%} +/- {np.std(loso_scores):.1%}")
print("\n" + "=" * 60)
