"""
BCICIV 2b - 多次运行取平均
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['OMP_NUM_THREADS'] = '4'
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(4)
CACHE_DIR = r"D:\brainwave-morse\cache"


class EEGNet(nn.Module):
    def __init__(self, n_ch=3, n_times=1001, n_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, (n_ch, 1), groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(16, 16, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(0.5)
        self.fc = nn.Linear(16 * (n_times // 4 // 8), n_classes)
    
    def forward(self, x):
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = self.pool1(x); x = self.drop1(x)
        x = F.elu(self.bn3(self.conv3(x)))
        x = self.pool2(x); x = self.drop2(x)
        return self.fc(x.view(x.size(0), -1))


def normalize(X):
    X_t = torch.FloatTensor(X)
    for ch in range(X_t.shape[1]):
        m = X_t[:, ch].mean(); s = X_t[:, ch].std() + 1e-8
        X_t[:, ch] = (X_t[:, ch] - m) / s
    return X_t.unsqueeze(1)


def train(X, y, n_epochs=50, batch=64, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device('cpu')
    Xt = normalize(X).to(dev); yt = torch.LongTensor(y).to(dev)
    model = EEGNet(n_ch=3, n_times=X.shape[2]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit = nn.CrossEntropyLoss()
    for ep in range(n_epochs):
        model.train()
        idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            bi = idx[i:i+batch]
            xb = Xt[bi].clone()
            xb = xb + torch.randn_like(xb) * 0.02
            for j in range(len(xb)):
                xb[j] = torch.roll(xb[j], np.random.randint(-12, 13), dims=1)
            opt.zero_grad()
            crit(model(xb), yt[bi]).backward()
            opt.step()
        sch.step()
    return model


def eval_model(model, X, y):
    dev = torch.device('cpu')
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(normalize(X).to(dev)), dim=1)[:, 1].cpu().numpy()
    return accuracy_score(y, (probs > 0.5).astype(int)), probs


def main():
    print("=" * 60)
    print("BCICIV 2b - 多次运行评估")
    print("=" * 60)
    
    print("\n[1] Loading...")
    data = []
    for fp in sorted(Path(CACHE_DIR).glob("*_X.npy")):
        name = fp.stem.replace('_X', '')
        X = np.load(fp); y = np.load(os.path.join(CACHE_DIR, f"{name}_y.npy"))
        data.append((name, X, y))
    
    Xa = np.vstack([x[1] for x in data])
    ya = np.concatenate([x[2] for x in data])
    print(f"  Total: {Xa.shape}")
    
    # 多次运行
    all_means = []
    for run_id in range(5):
        print(f"\n[2.{run_id+1}] Run {run_id+1} (seed={42+run_id*100})...")
        model = train(Xa, ya, n_epochs=50, batch=64, seed=42+run_id*100)
        
        scores = []
        all_probs = []
        for name, X, y in data:
            acc, probs = eval_model(model, X, y)
            scores.append(acc)
            all_probs.append(probs)
        
        mean_acc = np.mean(scores)
        all_means.append(mean_acc)
        print(f"  Run {run_id+1}: {mean_acc:.1%} +/- {np.std(scores):.1%}")
    
    print(f"\n[3] 5次运行平均: {np.mean(all_means):.1%} +/- {np.std(all_means):.1%}")
    print("=" * 60)


if __name__ == '__main__':
    main()
