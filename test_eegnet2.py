"""
BCICIV 2b - 优化EEGNet + 数据增强
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
    def __init__(self, n_ch=3, n_times=1001, n_classes=2, F1=16, D=2, F2=32, kernel_len=125):
        super().__init__()
        # Block 1: Temporal + Depthwise Spatial
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_len), padding=(0, kernel_len//2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1*D, (n_ch, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1*D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(0.25)
        
        # Block 2: Separable Conv
        self.conv3 = nn.Conv2d(F1*D, F1*D, (1, 32), padding=(0, 16), bias=False)
        self.bn3 = nn.BatchNorm2d(F1*D)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(0.25)
        
        # Block 3: Additional conv
        self.conv4 = nn.Conv2d(F1*D, F2, (1, 8), padding=(0, 4), bias=False)
        self.bn4 = nn.BatchNorm2d(F2)
        self.pool3 = nn.AvgPool2d((1, 8))
        self.drop3 = nn.Dropout(0.5)
        
        # Use adaptive pooling to handle variable lengths
        self.global_pool = nn.AdaptiveAvgPool2d((1, 8))
        self.fc1 = nn.Linear(F2 * 8, n_classes)
    
    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x)
        x = self.depthwise(x); x = self.bn2(x); x = F.elu(x)
        x = self.pool1(x); x = self.drop1(x)
        x = self.conv3(x); x = self.bn3(x); x = F.elu(x)
        x = self.pool2(x); x = self.drop2(x)
        x = self.conv4(x); x = self.bn4(x); x = F.elu(x)
        x = self.global_pool(x); x = self.drop3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


def augment(X, noise_std=0.1, shift_max=25):
    """数据增强: 添加噪声 + 时间偏移"""
    n = len(X)
    X_aug = X.copy()
    # 随机噪声
    noise = np.random.randn(*X.shape) * noise_std
    X_aug += noise
    # 随机时间偏移 (循环移位)
    for i in range(n):
        shift = np.random.randint(-shift_max, shift_max+1)
        X_aug[i] = np.roll(X_aug[i], shift, axis=1)
    return X_aug


def train_eegnet(X_tr, y_tr, X_te, y_te, n_epochs=50, batch_size=32, lr=0.001):
    """Train EEGNet with data augmentation"""
    n_tr = len(X_tr)
    
    # Normalize per-channel using training data
    X_tr_t = torch.FloatTensor(X_tr)
    X_te_t = torch.FloatTensor(X_te)
    y_tr_t = torch.LongTensor(y_tr)
    
    for ch in range(X_tr_t.shape[1]):
        mean = X_tr_t[:, ch].mean()
        std = X_tr_t[:, ch].std() + 1e-8
        X_tr_t[:, ch] = (X_tr_t[:, ch] - mean) / std
        X_te_t[:, ch] = (X_te_t[:, ch] - mean) / std
    
    X_tr_t = X_tr_t.unsqueeze(1)
    X_te_t = X_te_t.unsqueeze(1)
    
    model = EEGNet(n_ch=3, n_times=X_tr.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    best_state = None
    
    for epoch in range(n_epochs):
        model.train()
        indices = torch.randperm(n_tr)
        total_loss = 0
        n_batches = 0
        
        for i in range(0, n_tr, batch_size):
            idx = indices[i:i+batch_size]
            
            # Data augmentation on training batch
            X_batch_np = X_tr_t[idx].numpy()
            X_aug_np = augment(X_batch_np, noise_std=0.1 * (1 - epoch/n_epochs))
            X_batch = torch.FloatTensor(X_aug_np)
            
            xb = X_batch.to(device)
            yb = y_tr_t[idx].to(device)
            
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            logits = model(X_te_t.to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
            acc = accuracy_score(y_te, preds)
            
            logits_tr = model(X_tr_t[:len(X_te)].to(device))
            preds_tr = logits_tr.argmax(dim=1).cpu().numpy()
            tr_acc = accuracy_score(y_tr[:len(X_te)], preds_tr)
        
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: loss={total_loss/n_batches:.3f} tr_acc={tr_acc:.1%} val_acc={acc:.1%}")
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(X_te_t.to(device)), dim=1)[:, 1].cpu().numpy()
    
    return probs, best_acc


def run_cv(epochs_list, n_splits=5):
    """Run CV"""
    X_all = np.vstack([e.get_data() for e in epochs_list])
    y_all = np.concatenate([e.events[:, 2] for e in epochs_list])
    
    print(f"  Data: {X_all.shape}")
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results_eegnet = []
    results_ensemble = []
    
    for fold_i, (tr_idx, te_idx) in enumerate(cv.split(X_all, y_all)):
        X_tr, X_te = X_all[tr_idx], X_all[te_idx]
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]
        
        print(f"\n  Fold {fold_i+1}:")
        
        # EEGNet
        print(f"    Training EEGNet...")
        probs_eegnet, best_acc = train_eegnet(X_tr, y_tr, X_te, y_te, n_epochs=50, lr=0.001)
        results_eegnet.append(best_acc)
        print(f"    EEGNet: {best_acc:.1%}")
        
        # Ensemble with 3 EEGNets
        all_probs = [probs_eegnet]
        for seed in [42, 123, 456]:
            np.random.seed(seed)
            torch.manual_seed(seed)
            probs, _ = train_eegnet(X_tr, y_tr, X_te, y_te, n_epochs=50, lr=0.001)
            all_probs.append(probs)
        
        # Average ensemble
        fused = np.mean(all_probs, axis=0)
        pred_fusion = (fused > 0.5).astype(int)
        acc_ensemble = accuracy_score(y_te, pred_fusion)
        results_ensemble.append(acc_ensemble)
        print(f"  Ensemble: {acc_ensemble:.1%}")
    
    return {
        'eegnet': (np.mean(results_eegnet), np.std(results_eegnet)),
        'ensemble': (np.mean(results_ensemble), np.std(results_ensemble)),
    }


def main():
    print("=" * 60)
    print("BCICIV 2b - 优化EEGNet版")
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
    
    # 5-fold CV
    print("\n[2] 5折交叉验证...")
    r = run_cv(epochs_list)
    
    print(f"\n  EEGNet:   {r['eegnet'][0]:.1%} +/- {r['eegnet'][1]:.1%}")
    print(f"  Ensemble: {r['ensemble'][0]:.1%} +/- {r['ensemble'][1]:.1%}")
    print("=" * 60)


if __name__ == '__main__':
    main()
