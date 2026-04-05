"""
BCICIV 2b - 快速预训练评估
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['OMP_NUM_THREADS'] = '4'
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(4)
CACHE_DIR = r"D:\brainwave-morse\cache"


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


def norm(X):
    X_t = torch.FloatTensor(X)
    for ch in range(X_t.shape[1]):
        m = X_t[:, ch].mean(); s = X_t[:, ch].std() + 1e-8
        X_t[:, ch] = (X_t[:, ch] - m) / s
    return X_t.unsqueeze(1)


def train(X, y, n_epochs=50, batch=64):
    dev = torch.device('cpu')
    Xt = norm(X).to(dev); yt = torch.LongTensor(y).to(dev)
    model = EEGNet(n_ch=3, n_times=X.shape[2]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit = nn.CrossEntropyLoss()
    for ep in range(n_epochs):
        model.train()
        idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            bi = idx[i:i+batch]
            opt.zero_grad()
            crit(model(Xt[bi]), yt[bi]).backward()
            opt.step()
        sch.step()
    return model


def eval_model(model, X, y):
    dev = torch.device('cpu')
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(norm(X).to(dev)), dim=1)[:, 1].cpu().numpy()
    return accuracy_score(y, (probs > 0.5).astype(int))


def main():
    print("=" * 60)
    print("BCICIV 2b - 预训练快速评估")
    print("=" * 60)
    
    # Load cache
    print("\n[1] Loading from cache...")
    data = []
    for fp in sorted(Path(CACHE_DIR).glob("*_X.npy")):
        name = fp.stem.replace('_X', '')
        X = np.load(fp)
        y = np.load(os.path.join(CACHE_DIR, f"{name}_y.npy"))
        data.append((name, X, y))
        print(f"  {name}: {len(X)}")
    
    Xa = np.vstack([x[1] for x in data])
    ya = np.concatenate([x[2] for x in data])
    print(f"\nTotal: {Xa.shape}")
    
    # Pretrain
    print("\n[2] 预训练EEGNet (50 epochs)...")
    model = train(Xa, ya, n_epochs=50, batch=64)
    print("  完成")
    
    # Direct LOSO
    print("\n[3] LOSO (预训练直接预测)...")
    scores = []
    for name, X, y in data:
        acc = eval_model(model, X, y)
        scores.append(acc)
        print(f"  {name}: {acc:.1%}")
    
    print(f"\n  LOSO Mean: {np.mean(scores):.1%} +/- {np.std(scores):.1%}")
    print("=" * 60)


if __name__ == '__main__':
    main()
