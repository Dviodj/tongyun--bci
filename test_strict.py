"""
BCICIV 2b - 严格验证版 (无数据泄露)
- CSP在每折内部计算
- StandardScaler在每折内部计算
- 报告真实的交叉验证分数
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['OMP_NUM_THREADS'] = '1'
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import numpy as np
from pathlib import Path
from scipy.signal import welch
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import mne

DATA_DIR = r"D:\db\BCICIV_2b_gdf"
SFREQ = 250


def load_subject(fp):
    import warnings as W
    W.filterwarnings('ignore')
    raw = mne.io.read_raw_gdf(fp, preload=True, verbose=False)
    
    # Fix channel names (may have "EEG:" prefix)
    ch_map = {}
    for ch in raw.ch_names:
        if ':' in ch:
            ch_map[ch] = ch.split(':')[1]
    raw.rename_channels(ch_map)
    
    # Pick EEG channels
    raw.pick(['C3', 'Cz', 'C4'])
    raw.filter(1, 40, method='iir')
    raw.set_eeg_reference('average', projection=True)
    raw.apply_proj()
    
    # Get events from annotations (MNE reads them correctly now)
    evs_all, ev_dict = mne.events_from_annotations(raw, verbose=False)
    # 769->10, 770->11 in the annotation mapping
    mask = np.isin(evs_all[:, 2], [10, 11])  # 10=769(左手), 11=770(右手)
    evs = evs_all[mask].copy()
    evs[:, 2] = np.where(evs[:, 2] == 10, 0, 1)  # 0=左手, 1=右手
    return raw, evs


def extract_epochs(raw, evs, tmin=-0.5, tmax=3.5):
    ep = mne.Epochs(raw, evs, tmin=tmin, tmax=tmax, baseline=(None, 0), preload=True, verbose=False)
    X = ep.get_data()
    y = ep.events[:, 2].astype(int)
    return X, y


class CSP:
    """CSP特征提取器"""
    def __init__(self, n_comp=4):
        self.n_comp = n_comp
        self.filters_ = None

    def fit(self, X, y):
        covs = np.array([np.cov(x) for x in X])
        cov1 = np.mean(covs[y == 0], axis=0)
        cov2 = np.mean(covs[y == 1], axis=0)
        cov_r = cov1 + cov2 + 0.1 * np.eye(cov1.shape[0])
        lam, W = eigh(cov2, cov_r)
        idx = np.argsort(np.abs(lam - 0.5))[::-1]  # 特征值排序
        self.filters_ = W[:, idx[:self.n_comp * 2]]
        return self

    def transform(self, X):
        Xf = np.tensordot(X, self.filters_, axes=([1], [0]))
        var = np.var(Xf, axis=2)
        var = var / (var.sum(axis=1, keepdims=True) + 1e-10)
        return np.log(var + 1e-10)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


def band_power_db(s, f1, f2, fs=SFREQ):
    """对数频段功率 (dB)"""
    try:
        f, p = welch(s, fs=fs, nperseg=min(64, len(s)//2))
        idx = (f >= f1) & (f <= f2)
        return 10 * np.log10(np.mean(p[idx]) + 1e-12) if idx.sum() > 0 else -12
    except:
        return -12


def band_power(s, f1, f2, fs=SFREQ):
    """线性频段功率"""
    try:
        f, p = welch(s, fs=fs, nperseg=min(64, len(s)//2))
        idx = (f >= f1) & (f <= f2)
        return np.mean(p[idx]) if idx.sum() > 0 else 1e-12
    except:
        return 1e-12


def traditional_features(X):
    """传统特征: 频段功率 + C3/C4不对称性"""
    F = []
    BANDS = [(8, 13), (13, 30), (6, 13), (18, 25)]
    for ep in X:
        f = []
        # 各通道频段功率
        for ch in range(3):
            s = ep[ch]
            for f1, f2 in BANDS:
                f.append(band_power_db(s, f1, f2))
                f.append(band_power(s, f1, f2))
            f.extend([np.mean(s), np.std(s), np.sqrt(np.mean(s**2)),
                     np.percentile(s, 5), np.percentile(s, 95)])
        # C3/C4 不对称性
        for f1, f2 in BANDS:
            c3 = band_power(ep[0], f1, f2) + 1e-12
            c4 = band_power(ep[2], f1, f2) + 1e-12
            f.extend([np.log(c3/c4), (c3-c4)/(c3+c4), c3-c4])
        F.append(f)
    return np.array(F)


def extract_features(X_train, y_train, X_test, csp_n=4):
    """在训练数据上拟合CSP和Scaler，然后转换测试数据"""
    # CSP - 只用训练数据
    csp = CSP(n_comp=csp_n)
    Fc_tr = csp.fit_transform(X_train, y_train)
    Fc_te = csp.transform(X_test)

    # 传统特征
    Fe_tr = traditional_features(X_train)
    Fe_te = traditional_features(X_test)

    # 标准化 - 只用训练数据拟合
    sc_csp = StandardScaler()
    sc_fe = StandardScaler()
    Fc_tr = np.nan_to_num(sc_csp.fit_transform(Fc_tr), nan=0, posinf=5, neginf=-5)
    Fc_te = np.nan_to_num(sc_csp.transform(Fc_te), nan=0, posinf=5, neginf=-5)
    Fe_tr = np.nan_to_num(sc_fe.fit_transform(Fe_tr), nan=0, posinf=5, neginf=-5)
    Fe_te = np.nan_to_num(sc_fe.transform(Fe_te), nan=0, posinf=5, neginf=-5)

    # 融合
    F_tr = np.hstack([Fc_tr, Fe_tr])
    F_te = np.hstack([Fc_te, Fe_te])
    return F_tr, F_te


def nested_cv_evaluation(X_all, y_all, subjects, n_splits=5):
    """嵌套交叉验证 - 超参调优在内部循环"""
    print("\n[NESTED CV] 嵌套交叉验证 (超参在内部调优)")
    print("-" * 50)

    cv_outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    outer_scores = []
    best_configs = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_outer.split(X_all, y_all)):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        # 内部CV调参
        cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        best_score = 0
        best_cfg = None

        for C in [0.1, 1, 5, 10]:
            for gamma in ['scale', 0.01, 0.1]:
                # 在内部CV上评估
                inner_scores = []
                for inner_train_idx, inner_val_idx in cv_outer.split(X_train, y_train):
                    # 只用内部训练数据提取特征
                    X_itrain = X_train[inner_train_idx]
                    y_itrain = y_train[inner_train_idx]
                    X_ival = X_train[inner_val_idx]

                    F_itrain, F_ival = extract_features(X_itrain, y_itrain, X_ival)
                    clf = SVC(kernel='rbf', C=C, gamma=gamma)
                    clf.fit(F_itrain, y_itrain)
                    inner_scores.append(clf.score(F_ival, y_train[inner_val_idx]))

                mean_score = np.mean(inner_scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_cfg = (C, gamma)

        # 用最佳参数在外部测试集上评估
        F_train, F_test = extract_features(X_train, y_train, X_test)
        clf = SVC(kernel='rbf', C=best_cfg[0], gamma=best_cfg[1])
        clf.fit(F_train, y_train)
        test_score = clf.score(F_test, y_test)
        outer_scores.append(test_score)
        best_configs.append(best_cfg)

        print(f"  Fold {fold_idx+1}: {test_score:.3f} (C={best_cfg[0]}, gamma={best_cfg[1]})")

    print(f"\n[NESTED CV] Mean: {np.mean(outer_scores):.3f} +/- {np.std(outer_scores):.3f}")
    return np.mean(outer_scores), np.std(outer_scores)


def loso_evaluation(all_X, all_y):
    """留一被试验证 - 每位被试单独测试"""
    print("\n[LOSO] 留一被试验证")
    print("-" * 50)

    scores = []
    for si in range(len(all_X)):
        train_X = np.vstack([all_X[j] for j in range(len(all_X)) if j != si])
        train_y = np.concatenate([all_y[j] for j in range(len(all_X)) if j != si])
        test_X = all_X[si]
        test_y = all_y[si]

        if len(np.unique(train_y)) < 2:
            scores.append(0.5)
            print(f"  Subject {si+1}: 0.500 (single class)")
            continue

        # 在训练被试上提取特征
        F_train, F_test = extract_features(train_X, train_y, test_X)

        clf = SVC(kernel='rbf', C=5, gamma='scale')
        clf.fit(F_train, train_y)
        score = clf.score(F_test, test_y)
        scores.append(score)

        print(f"  Subject {si+1}: {score:.3f} (n={len(test_X)}, L={sum(test_y==0)}, R={sum(test_y==1)})")

    print(f"\n[LOSO] Mean: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")
    return np.mean(scores), np.std(scores), scores


def main():
    print("=" * 65)
    print("BCICIV 2b - 严格验证版 (无数据泄露)")
    print("=" * 65)

    # 1. 加载数据
    print("\n[1] 加载数据...")
    all_X, all_y = [], []

    for fp in sorted(Path(DATA_DIR).glob("*T.gdf")):
        print(f"  {fp.name}...", end='', flush=True)
        try:
            raw, evs = load_subject(str(fp))
            X, y = extract_epochs(raw, evs)
            lcnt, rcnt = sum(y==0), sum(y==1)
            print(f" {len(X)} epochs (L={lcnt}, R={rcnt})")
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
        except Exception as e:
            print(f" ERROR: {str(e)[:50]}")

    if not all_X:
        print("[ERROR] No data loaded!")
        return

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    print(f"\n[Data] {X_all.shape} - L={sum(y_all==0)}, R={sum(y_all==1)}")

    # 2. 5折交叉验证 (无泄露)
    print("\n[2] 5折交叉验证 (正确方法)")
    cv5_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_scores = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv5_splitter.split(X_all, y_all)):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        F_train, F_test = extract_features(X_train, y_train, X_test)
        clf = SVC(kernel='rbf', C=5, gamma='scale')
        clf.fit(F_train, y_train)
        score = clf.score(F_test, y_test)
        fold_scores.append(score)
        print(f"  Fold {fold_idx+1}: {score:.3f}")

    cv5_mean = np.mean(fold_scores)
    cv5_std = np.std(fold_scores)
    print(f"\n[5-Fold CV] {cv5_mean:.3f} +/- {cv5_std:.3f}")

    # 3. 10折x3次
    print("\n[3] 10折x3次重复...")
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    all_10x3_scores = []

    for train_idx, test_idx in rskf.split(X_all, y_all):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        F_train, F_test = extract_features(X_train, y_train, X_test)
        clf = SVC(kernel='rbf', C=5, gamma='scale')
        clf.fit(F_train, y_train)
        all_10x3_scores.append(clf.score(F_test, y_test))

    print(f"[10x3] {np.mean(all_10x3_scores):.3f} +/- {np.std(all_10x3_scores):.3f}")

    # 4. LOSO
    loso_mean, loso_std, loso_scores = loso_evaluation(all_X, all_y)

    # 5. 总结
    print("\n" + "=" * 65)
    print("最终结果 (无数据泄露)")
    print("=" * 65)
    print(f"  5折交叉验证:     {cv5_mean:.1%} +/- {cv5_std:.1%}")
    print(f"  10折x3次:        {np.mean(all_10x3_scores):.1%} +/- {np.std(all_10x3_scores):.1%}")
    print(f"  留一被试LOSO:    {loso_mean:.1%} +/- {loso_std:.1%}")
    print("=" * 65)


if __name__ == '__main__':
    main()
