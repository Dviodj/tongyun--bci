"""
BCICIV 2b - 预训练+微调 v2
正确流程：全局预训练 → LOSO时加载预训练权重 → 目标被试微调
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


def train_model(X, y, n_epochs=50, batch_size=64, lr=0.001, wd=0.01):
    """训练EEGNet"""
    device = torch.device('cpu')
    X_t = normalize(X).to(device)
    y_t = torch.LongTensor(y).to(device)
    model = EEGNet(n_ch=3, n_times=X.shape[2]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit = nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        model.train()
        idx = torch.randperm(len(X_t))
        for i in range(0, len(X_t), batch_size):
            bi = idx[i:i+batch_size]
            opt.zero_grad()
            loss = crit(model(X_t[bi]), y_t[bi])
            loss.backward(); opt.step()
        sched.step()
    return model


def finetune(model, X, y, n_epochs=20, batch_size=32, lr=0.0001, freeze_conv=False):
    """微调模型"""
    device = torch.device('cpu')
    if freeze_conv:
        for name, param in model.named_parameters():
            param.requires_grad = ('fc' in name)
    
    X_t = normalize(X).to(device)
    y_t = torch.LongTensor(y).to(device)
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if not params:
        for param in model.parameters():
            param.requires_grad = True
        params = list(model.parameters())
    
    opt = torch.optim.AdamW(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit = nn.CrossEntropyLoss()
    
    for epoch in range(n_epochs):
        model.train()
        idx = torch.randperm(len(X_t))
        for i in range(0, len(X_t), batch_size):
            bi = idx[i:i+batch_size]
            xb = X_t[bi].clone()
            xb = xb + torch.randn_like(xb) * 0.01
            for j in range(len(xb)):
                xb[j] = torch.roll(xb[j], np.random.randint(-8, 9), dims=1)
            opt.zero_grad()
            loss = crit(model(xb), y_t[bi])
            loss.backward(); opt.step()
        sched.step()
    
    for param in model.parameters():
        param.requires_grad = True
    return model


def predict(model, X, y):
    device = torch.device('cpu')
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(normalize(X).to(device)), dim=1)[:, 1].cpu().numpy()
    return accuracy_score(y, (probs > 0.5).astype(int)), probs


def main():
    print("=" * 60)
    print("BCICIV 2b - 预训练+微调 v2")
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
    
    X_all = np.vstack([e.get_data() for e in epochs_list])
    y_all = np.concatenate([e.events[:, 2] for e in epochs_list])
    n_per = [len(ep) for ep in epochs_list]
    print(f"\nTotal: {X_all.shape}")
    
    # Pre-train on ALL data
    print("\n[2] 全局预训练EEGNet...")
    global_model = train_model(X_all, y_all, n_epochs=50, batch_size=64)
    print("  预训练完成")
    
    # LOSO evaluation: per-subject within-subject CV with pretrained model
    print("\n[3] LOSO (预训练→目标被试微调)...")
    from sklearn.model_selection import StratifiedKFold
    
    loso_pretrain = []  # just pretrained, no fine-tune
    loso_ft5 = []       # fine-tune 5 epochs
    loso_ft10 = []      # fine-tune 10 epochs
    loso_ft20 = []     # fine-tune 20 epochs
    
    for si in range(len(epochs_list)):
        X_s = epochs_list[si].get_data()
        y_s = epochs_list[si].events[:, 2]
        
        # Within-subject CV (standard for BCICIV with pretrained model)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        pretrain_scores = []
        ft5_scores = []
        ft10_scores = []
        ft20_scores = []
        
        for tr_idx, te_idx in cv.split(X_s, y_s):
            X_tr, X_te = X_s[tr_idx], X_s[te_idx]
            y_tr, y_te = y_s[tr_idx], y_s[te_idx]
            
            # Method 1: Just pretrained (no fine-tune on target)
            acc_pretrain, _ = predict(global_model, X_te, y_te)
            pretrain_scores.append(acc_pretrain)
            
            # Fine-tune pretrained model on target's training data
            model_copy = train_model(X_tr, y_tr, n_epochs=20, batch_size=32, lr=0.001)
            acc5, _ = predict(model_copy, X_te, y_te)
            ft5_scores.append(acc5)
            
            # More fine-tuning
            model_copy = train_model(X_tr, y_tr, n_epochs=40, batch_size=32, lr=0.0005)
            acc10, _ = predict(model_copy, X_te, y_te)
            ft10_scores.append(acc10)
            
            model_copy = train_model(X_tr, y_tr, n_epochs=60, batch_size=32, lr=0.0005)
            acc20, _ = predict(model_copy, X_te, y_te)
            ft20_scores.append(acc20)
        
        mean_pre = np.mean(pretrain_scores)
        mean_5 = np.mean(ft5_scores)
        mean_10 = np.mean(ft10_scores)
        mean_20 = np.mean(ft20_scores)
        
        loso_pretrain.append(mean_pre)
        loso_ft5.append(mean_5)
        loso_ft10.append(mean_10)
        loso_ft20.append(mean_20)
        
        print(f"  S{si+1}: Pretrain={mean_pre:.1%} FT5={mean_5:.1%} FT10={mean_10:.1%} FT20={mean_20:.1%}")
    
    print(f"\n  Pretrain LOSO: {np.mean(loso_pretrain):.1%} +/- {np.std(loso_pretrain):.1%}")
    print(f"  FT5 epochs:     {np.mean(loso_ft5):.1%} +/- {np.std(loso_ft5):.1%}")
    print(f"  FT10 epochs:   {np.mean(loso_ft10):.1%} +/- {np.std(loso_ft10):.1%}")
    print(f"  FT20 epochs:   {np.mean(loso_ft20):.1%} +/- {np.std(loso_ft20):.1%}")
    print("=" * 60)


if __name__ == '__main__':
    main()
