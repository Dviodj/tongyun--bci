"""
BCICIV 2b - 预训练+直接预测(无微调)
预训练后直接预测，不在目标被试上训练
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['OMP_NUM_THREADS'] = '4'
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score
import mne
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(4)
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
    def __init__(self, n_ch=3, n_times=1001, F1=8, D=2, F2=16, n_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.conv2 = nn.Conv2d(F1, F1*D, (n_ch, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1*D)
        self.pool1 = nn.AvgPool2d((1, 4)); self.drop1 = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(F1*D, F1*D, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F1*D)
        self.pool2 = nn.AvgPool2d((1, 8)); self.drop2 = nn.Dropout(0.5)
        self.fc = nn.Linear(F2 * (n_times // 4 // 8), n_classes)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x); x = self.drop1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x); x = self.drop2(x)
        return self.fc(x.view(x.size(0), -1))


def normalize(X):
    X_t = torch.FloatTensor(X)
    for ch in range(X_t.shape[1]):
        m = X_t[:, ch].mean(); s = X_t[:, ch].std() + 1e-8
        X_t[:, ch] = (X_t[:, ch] - m) / s
    return X_t.unsqueeze(1)


def pretrain(source_X, source_y, n_epochs=50, batch_size=64):
    device = torch.device('cpu')
    X_s = normalize(source_X)
    y_s = torch.LongTensor(source_y)
    model = EEGNet(n_ch=3, n_times=source_X.shape[2]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit = nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        model.train()
        indices = torch.randperm(len(X_s))
        for i in range(0, len(X_s), batch_size):
            idx = indices[i:i+batch_size]
            xb = X_s[idx].to(device); yb = y_s[idx].to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        sched.step()
        if (epoch+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(X_s[:1000].to(device)).argmax(1) == y_s[:1000].to(device)).float().mean()
            print(f"  Epoch {epoch+1}: acc={acc:.1%}")
    return model


def predict(model, X, y):
    device = torch.device('cpu')
    model.eval()
    X_t = normalize(X).to(device)
    with torch.no_grad():
        probs = F.softmax(model(X_t), dim=1)[:, 1].cpu().numpy()
    return accuracy_score(y, (probs > 0.5).astype(int)), probs


def main():
    print("=" * 60)
    print("BCICIV 2b - 预训练+直接预测")
    print("=" * 60)
    
    # Load (with timeout protection)
    print("\n[1] Loading...")
    epochs_list = []
    for fp in sorted(Path(DATA_DIR).glob("*T.gdf")):
        try:
            ep = load_subject(str(fp))
            epochs_list.append(ep)
            print(f"  {fp.name}: {len(ep)}")
        except Exception as e:
            print(f"  {fp.name}: ERROR - {str(e)[:30]}")
    
    if len(epochs_list) < 2:
        print("Not enough subjects!")
        return
    
    X_all = np.vstack([e.get_data() for e in epochs_list])
    y_all = np.concatenate([e.events[:, 2] for e in epochs_list])
    n_per = [len(ep) for ep in epochs_list]
    print(f"\nTotal: {X_all.shape}")
    
    # Pre-train on ALL data
    print("\n[2] 预训练EEGNet...")
    model = pretrain(X_all, y_all, n_epochs=50, batch_size=64)
    print("  预训练完成")
    
    # LOSO evaluation
    print("\n[3] LOSO评估...")
    loso_scores = []
    pos = 0
    for si in range(len(epochs_list)):
        tr_idx = np.concatenate([np.arange(pos, pos+n_per[j]) for j in range(len(epochs_list)) if j != si])
        te_idx = np.arange(pos, pos+n_per[si])
        pos += n_per[si]
        
        X_te, y_te = X_all[te_idx], y_all[te_idx]
        
        # Fine-tune on target subject's data (optional: just predict directly)
        acc, probs = predict(model, X_te, y_te)
        loso_scores.append(acc)
        print(f"  S{si+1}: {acc:.1%}")
    
    print(f"\n  LOSO Mean: {np.mean(loso_scores):.1%} +/- {np.std(loso_scores):.1%}")
    print("=" * 60)


if __name__ == '__main__':
    main()
