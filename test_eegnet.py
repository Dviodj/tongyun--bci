"""
BCICIV 2b - EEGNet + 深度特征融合版
使用PyTorch EEGNet + 多频段特征
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['OMP_NUM_THREADS'] = '4'
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import mne
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)
device = torch.device('cpu')
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


class EEGNet(nn.Module):
    """Lightweight EEGNet for CPU training"""
    def __init__(self, n_ch=3, n_times=1126, n_classes=2, F1=8, D=2, F2=16, kernel_len=64):
        super().__init__()
        # Block 1: Temporal conv
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_len), padding=(0, kernel_len//2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        # Depthwise: spatial filter per band
        self.conv2 = nn.Conv2d(F1, F1*D, (n_ch, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1*D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(0.5)
        # Block 2: Separable conv
        self.conv3 = nn.Conv2d(F1*D, F1*D, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F1*D)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(0.5)
        # Compute flatten size
        out_len = n_times // 4 // 8
        self.fc = nn.Linear(F2 * out_len, n_classes)
    
    def forward(self, x):
        # x: (batch, 1, n_ch, n_times)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def extract_deep_features(X, fs=SFREQ):
    """多时间窗口 + 多频段特征"""
    n_epochs, n_ch, n_times = X.shape
    bands = [(8, 13), (13, 30), (6, 13), (18, 25)]
    windows = [
        (125, 500),    # 0.5-2s early
        (500, 875),    # 2-3.5s mid
    ]
    
    def bp(s, f1, f2, order=4):
        nyq = fs / 2
        b, a = butter(order, [f1/nyq, f2/nyq], btype='band')
        return filtfilt(b, a, s)
    
    features = []
    for i in range(n_epochs):
        fv = []
        ep = X[i]
        for ch in range(n_ch):
            for f1, f2 in bands:
                filt_s = bp(ep[ch], f1, f2)
                for w_s, w_e in windows:
                    p = np.var(filt_s[w_s:w_e])
                    fv.extend([10*np.log10(p+1e-12), p])
            fv.extend([np.mean(ep[ch]), np.std(ep[ch])])
        
        for f1, f2 in bands:
            c3 = np.var(bp(ep[0], f1, f2)) + 1e-12
            c4 = np.var(bp(ep[2], f1, f2)) + 1e-12
            fv.extend([np.log(c3/c4), (c3-c4)/(c3+c4)])
            
            for w_s, w_e in windows:
                c3w = np.var(bp(ep[0], f1, f2)[w_s:w_e]) + 1e-12
                c4w = np.var(bp(ep[2], f1, f2)[w_s:w_e]) + 1e-12
                fv.extend([np.log(c3w/c4w), (c3w-c4w)/(c3w+c4w)])
        
        features.append(fv)
    return np.array(features)


def train_eegnet(X_tr, y_tr, X_te, n_epochs=30, batch_size=64):
    """Train EEGNet and return predictions"""
    n_tr = len(X_tr)
    n_te = len(X_te)
    
    # Prepare data: (N, 1, C, T), normalized per-channel
    X_tr_t = torch.FloatTensor(X_tr)
    X_te_t = torch.FloatTensor(X_te)
    y_tr_t = torch.LongTensor(y_tr)
    
    # Normalize
    for ch in range(X_tr_t.shape[1]):
        mean = X_tr_t[:, ch].mean()
        std = X_tr_t[:, ch].std() + 1e-8
        X_tr_t[:, ch] = (X_tr_t[:, ch] - mean) / std
        X_te_t[:, ch] = (X_te_t[:, ch] - mean) / std
    
    X_tr_t = X_tr_t.unsqueeze(1)  # (N, 1, C, T)
    X_te_t = X_te_t.unsqueeze(1)
    
    model = EEGNet(n_ch=3, n_times=X_tr.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    model.train()
    for epoch in range(n_epochs):
        indices = torch.randperm(n_tr)
        for i in range(0, n_tr, batch_size):
            idx = indices[i:i+batch_size]
            xb = X_tr_t[idx].to(device)
            yb = y_tr_t[idx].to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
    
    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(X_te_t.to(device))
        probs = F.softmax(logits, dim=1)
    return probs[:, 1].cpu().numpy()


def run_cv(epochs_list, n_splits=5):
    """Run CV with EEGNet + features + fusion"""
    from mne.decoding import CSP
    
    X_all = np.vstack([e.get_data() for e in epochs_list])
    y_all = np.concatenate([e.events[:, 2] for e in epochs_list])
    
    print(f"  Data: {X_all.shape}")
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results_eegnet = []
    results_feat = []
    results_fusion = []
    
    for fold_i, (tr_idx, te_idx) in enumerate(cv.split(X_all, y_all)):
        X_tr, X_te = X_all[tr_idx], X_all[te_idx]
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]
        
        # EEGNet
        print(f"    Training EEGNet fold {fold_i+1}...", flush=True)
        probs_eegnet = train_eegnet(X_tr, y_tr, X_te)
        pred_eegnet = (probs_eegnet > 0.5).astype(int)
        acc_eegnet = accuracy_score(y_te, pred_eegnet)
        results_eegnet.append(acc_eegnet)
        
        # Deep features
        F_tr = extract_deep_features(X_tr)
        F_te = extract_deep_features(X_te)
        sc = StandardScaler()
        F_tr = np.nan_to_num(sc.fit_transform(F_tr), nan=0, posinf=10, neginf=-10)
        F_te = np.nan_to_num(sc.transform(F_te), nan=0, posinf=10, neginf=-10)
        
        # SVM + LDA ensemble
        clf_svm = SVC(kernel='rbf', C=10, probability=True)
        clf_svm.fit(F_tr, y_tr)
        prob_svm = clf_svm.predict_proba(F_te)[:, 1]
        acc_svm = clf_svm.score(F_te, y_te)
        results_feat.append(acc_svm)
        
        # Fusion
        fused = 0.5 * probs_eegnet + 0.5 * prob_svm
        pred_fusion = (fused > 0.5).astype(int)
        acc_fusion = accuracy_score(y_te, pred_fusion)
        results_fusion.append(acc_fusion)
        
        print(f"  Fold {fold_i+1}: EEGNet={acc_eegnet:.1%} Feat={acc_svm:.1%} Fusion={acc_fusion:.1%}")
    
    return {
        'eegnet': (np.mean(results_eegnet), np.std(results_eegnet)),
        'feat': (np.mean(results_feat), np.std(results_feat)),
        'fusion': (np.mean(results_fusion), np.std(results_fusion)),
    }


def main():
    print("=" * 60)
    print("BCICIV 2b - EEGNet + 深度特征版")
    print("=" * 60)
    
    # Load
    print("\n[1] Loading...")
    epochs_list = []
    for fp in sorted(Path(DATA_DIR).glob("*T.gdf")):
        try:
            ep = load_subject(str(fp))
            epochs_list.append(ep)
            print(f"  {fp.name}: {len(ep)}")
        except:
            print(f"  {fp.name}: ERROR")
    
    if not epochs_list:
        print("No data!")
        return
    
    # 5-fold CV
    print("\n[2] 5折交叉验证...")
    r = run_cv(epochs_list)
    
    print(f"\n  EEGNet:  {r['eegnet'][0]:.1%} +/- {r['eegnet'][1]:.1%}")
    print(f"  Feat:    {r['feat'][0]:.1%} +/- {r['feat'][1]:.1%}")
    print(f"  Fusion:  {r['fusion'][0]:.1%} +/- {r['fusion'][1]:.1%}")
    print("=" * 60)


if __name__ == '__main__':
    main()
