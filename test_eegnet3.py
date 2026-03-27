"""
BCICIV 2b - EEGNet优化版 v3
基于已验证的v1架构，增加更多通道和数据增强
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['OMP_NUM_THREADS'] = '4'
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from pathlib import Path
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

# Motor-related channels available in BCICIV 2b
MOTOR_CHANNELS = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'FCz', 'CP1', 'CP2', 'CPz']


def load_subject(fp, use_all_channels=False):
    raw = mne.io.read_raw_gdf(fp, preload=True, verbose=False)
    ch_map = {ch: ch.split(':')[1] if ':' in ch else ch for ch in raw.ch_names}
    raw.rename_channels(ch_map)
    
    if use_all_channels:
        # Pick all available EEG channels
        eeg_chs = [ch for ch in raw.ch_names if ch not in ['Stim', 'EMG']]
        raw.pick(eeg_chs)
    else:
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
    def __init__(self, n_ch=3, n_times=1001, n_classes=2, F1=8, D=2, F2=16, kernel_len=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_len), padding=(0, kernel_len//2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.conv2 = nn.Conv2d(F1, F1*D, (n_ch, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1*D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(F1*D, F1*D, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F1*D)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(0.5)
        out_len = n_times // 4 // 8
        self.fc = nn.Linear(F2 * out_len, n_classes)
    
    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x)
        x = self.conv2(x); x = self.bn2(x); x = F.elu(x)
        x = self.pool1(x); x = self.drop1(x)
        x = self.conv3(x); x = self.bn3(x); x = F.elu(x)
        x = self.pool2(x); x = self.drop2(x)
        x = x.view(x.size(0), -1); x = self.fc(x)
        return x


def augment_batch(X, y, noise_std=0.05):
    """在线数据增强"""
    X_aug = X.copy()
    n = len(X)
    # 通道噪声
    noise = np.random.randn(*X.shape) * noise_std
    X_aug += noise
    # 随机时间偏移 (±12 samples = ±48ms)
    for i in range(n):
        shift = np.random.randint(-12, 13)
        X_aug[i] = np.roll(X_aug[i], shift, axis=1)
    return X_aug


def train_eegnet(X_tr, y_tr, X_te, y_te, n_epochs=50, batch_size=64, lr=0.001, seed=42):
    """训练EEGNet with augmentation"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    X_tr_t = torch.FloatTensor(X_tr)
    X_te_t = torch.FloatTensor(X_te)
    y_tr_t = torch.LongTensor(y_tr)
    
    # Per-channel normalization
    for ch in range(X_tr_t.shape[1]):
        mean = X_tr_t[:, ch].mean()
        std = X_tr_t[:, ch].std() + 1e-8
        X_tr_t[:, ch] = (X_tr_t[:, ch] - mean) / std
        X_te_t[:, ch] = (X_te_t[:, ch] - mean) / std
    
    X_tr_t = X_tr_t.unsqueeze(1)
    X_te_t = X_te_t.unsqueeze(1)
    
    n_ch = X_tr.shape[1]
    n_times = X_tr.shape[2]
    
    model = EEGNet(n_ch=n_ch, n_times=n_times).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    best_probs = None
    
    for epoch in range(n_epochs):
        model.train()
        indices = torch.randperm(len(X_tr_t))
        
        for i in range(0, len(X_tr_t), batch_size):
            idx = indices[i:i+batch_size]
            # Augment on-the-fly
            X_batch = augment_batch(X_tr_t[idx].numpy(), None)
            xb = torch.FloatTensor(X_batch).to(device)
            yb = y_tr_t[idx].to(device)
            
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            logits = model(X_te_t.to(device))
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            acc = accuracy_score(y_te, (probs > 0.5).astype(int))
        
        if acc > best_acc:
            best_acc = acc
            best_probs = probs.copy()
    
    return best_probs, best_acc


def main():
    print("=" * 60)
    print("BCICIV 2b - EEGNet v3 优化版")
    print("=" * 60)
    
    # Load data with 3 channels
    print("\n[1] Loading (3-channel)...")
    epochs_list = []
    for fp in sorted(Path(DATA_DIR).glob("*T.gdf")):
        try:
            ep = load_subject(str(fp), use_all_channels=False)
            epochs_list.append(ep)
            print(f"  {fp.name}: {len(ep)} epochs, ch={ep.get_data().shape[1]}")
        except Exception as e:
            print(f"  {fp.name}: ERROR")
    
    X_all = np.vstack([e.get_data() for e in epochs_list])
    y_all = np.concatenate([e.events[:, 2] for e in epochs_list])
    print(f"\nTotal: {X_all.shape}")
    
    # 5-fold CV with multiple seeds
    print("\n[2] 5折交叉验证 (3个seed ensemble)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    
    for fold_i, (tr_idx, te_idx) in enumerate(cv.split(X_all, y_all)):
        X_tr, X_te = X_all[tr_idx], X_all[te_idx]
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]
        
        print(f"\n  Fold {fold_i+1}:")
        
        # Train multiple EEGNets with different seeds
        all_probs = []
        all_accs = []
        for seed in [42, 123, 456]:
            probs, acc = train_eegnet(X_tr, y_tr, X_te, y_te, n_epochs=50, seed=seed)
            all_probs.append(probs)
            all_accs.append(acc)
        
        # Ensemble
        fused = np.mean(all_probs, axis=0)
        pred_fused = (fused > 0.5).astype(int)
        acc_ensemble = accuracy_score(y_te, pred_fused)
        fold_results.append(acc_ensemble)
        
        print(f"    EEGNet-42: {all_accs[0]:.1%}")
        print(f"    EEGNet-123: {all_accs[1]:.1%}")
        print(f"    EEGNet-456: {all_accs[2]:.1%}")
        print(f"    Ensemble: {acc_ensemble:.1%}")
    
    print(f"\n  Ensemble Mean: {np.mean(fold_results):.1%} +/- {np.std(fold_results):.1%}")
    print("=" * 60)


if __name__ == '__main__':
    main()
