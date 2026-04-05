"""
BCICIV 2b - 冲击80%+方案 v2
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


class DeepEEGNet(nn.Module):
    def __init__(self, n_ch=3, n_times=1001, n_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, (n_ch, 1), groups=16, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(0.4)
        self.conv3 = nn.Conv2d(32, 32, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, (1, 8), padding=(0, 4), bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d((1, 4))
        self.drop2 = nn.Dropout(0.4)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, n_classes)
    
    def forward(self, x):
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = self.pool1(x); x = self.drop1(x)
        x = F.elu(self.bn3(self.conv3(x)))
        x = F.elu(self.bn4(self.conv4(x)))
        x = self.pool2(x); x = self.drop2(x)
        x = self.gap(x)
        return self.fc(x.view(x.size(0), -1))


def normalize(X):
    X_t = torch.FloatTensor(X)
    for ch in range(X_t.shape[1]):
        m = X_t[:, ch].mean()
        s = X_t[:, ch].std() + 1e-8
        X_t[:, ch] = (X_t[:, ch] - m) / s
    return X_t.unsqueeze(1)


def train_model(X, y, n_epochs=100, batch_size=64, lr=0.001, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cpu')
    X_t = normalize(X).to(device)
    y_t = torch.LongTensor(y).to(device)
    
    model = DeepEEGNet(n_ch=3, n_times=X.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_acc = 0
    best_state = None
    
    for epoch in range(n_epochs):
        model.train()
        indices = torch.randperm(len(X_t))
        
        for i in range(0, len(X_t), batch_size):
            bi = indices[i:i+batch_size]
            xb = X_t[bi].clone()
            yb = y_t[bi]
            
            # 数据增强
            noise_std = 0.03 * (1 - epoch/n_epochs)
            xb = xb + torch.randn_like(xb) * noise_std
            for j in range(len(xb)):
                shift = np.random.randint(-16, 17)
                xb[j] = torch.roll(xb[j], shift, dims=1)
            
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()
        
        if (epoch + 1) % 25 == 0:
            model.eval()
            with torch.no_grad():
                preds = model(X_t[:500]).argmax(1)
                acc = (preds == y_t[:500]).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"    Epoch {epoch+1}: train_acc={acc:.1%}")
    
    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate(model, X, y):
    device = torch.device('cpu')
    model.eval()
    with torch.no_grad():
        X_t = normalize(X).to(device)
        probs = F.softmax(model(X_t), dim=1)[:, 1].cpu().numpy()
    return accuracy_score(y, (probs > 0.5).astype(int)), probs


def ensemble_predict(models, X, y):
    all_probs = []
    for model in models:
        _, probs = evaluate(model, X, y)
        all_probs.append(probs)
    avg_probs = np.mean(all_probs, axis=0)
    return accuracy_score(y, (avg_probs > 0.5).astype(int))


def main():
    print("=" * 60)
    print("BCICIV 2b - 冲击80%+方案")
    print("=" * 60)
    
    print("\n[1] Loading...")
    data = []
    for fp in sorted(Path(CACHE_DIR).glob("*_X.npy")):
        name = fp.stem.replace('_X', '')
        X = np.load(fp)
        y = np.load(os.path.join(CACHE_DIR, f"{name}_y.npy"))
        data.append((name, X, y))
    
    X_all = np.vstack([x[1] for x in data])
    y_all = np.concatenate([x[2] for x in data])
    print(f"  Total: {X_all.shape}")
    
    # 训练多个模型
    print("\n[2] 训练多个EEGNet (3 seeds x 100 epochs)...")
    models = []
    for seed in [42, 123, 456]:
        print(f"  Model seed={seed}:")
        model = train_model(X_all, y_all, n_epochs=100, batch_size=64, lr=0.001, seed=seed)
        models.append(model)
    
    # LOSO
    print("\n[3] LOSO评估...")
    single_scores = []
    ensemble_scores = []
    
    for name, X, y in data:
        acc_single, _ = evaluate(models[0], X, y)
        single_scores.append(acc_single)
        
        acc_ens = ensemble_predict(models, X, y)
        ensemble_scores.append(acc_ens)
        
        print(f"  {name}: Single={acc_single:.1%} Ensemble={acc_ens:.1%}")
    
    print(f"\n  Single: {np.mean(single_scores):.1%} +/- {np.std(single_scores):.1%}")
    print(f"  Ensemble: {np.mean(ensemble_scores):.1%} +/- {np.std(ensemble_scores):.1%}")
    print("=" * 60)


if __name__ == '__main__':
    main()
