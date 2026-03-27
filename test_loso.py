"""
BCICIV 2b - LOSO评估版
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
from sklearn.metrics import accuracy_score
from scipy.linalg import eigh
import mne
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, n_ch=3, n_times=1001, F1=8, D=2, F2=16, kernel_len=64):
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
        self.fc = nn.Linear(F2 * (n_times // 4 // 8), 2)
    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x)
        x = self.conv2(x); x = self.bn2(x); x = F.elu(x)
        x = self.pool1(x); x = self.drop1(x)
        x = self.conv3(x); x = self.bn3(x); x = F.elu(x)
        x = self.pool2(x); x = self.drop2(x)
        x = x.view(x.size(0), -1); x = self.fc(x)
        return x


def deep_features(X):
    """快速多频段特征"""
    bands = [(8,13),(13,30),(6,13),(18,25)]
    def bp(s,f1,f2):
        nyq=SFREQ/2; b,a=butter(4,[f1/nyq,f2/nyq],btype='band')
        return filtfilt(b,a,s)
    F=[]
    for ep in X:
        fv=[]
        for ch in range(3):
            s=ep[ch]
            for f1,f2 in bands:
                p=np.var(bp(s,f1,f2)); fv.extend([10*np.log10(p+1e-12),p])
            fv.extend([np.mean(s),np.std(s)])
        for f1,f2 in bands:
            c3=np.var(bp(ep[0],f1,f2))+1e-12; c4=np.var(bp(ep[2],f1,f2))+1e-12
            fv.extend([np.log(c3/c4),(c3-c4)/(c3+c4)])
            for w in [(125,500),(500,875)]:
                c3w=np.var(bp(ep[0],f1,f2)[w[0]:w[1]])+1e-12
                c4w=np.var(bp(ep[2],f1,f2)[w[0]:w[1]])+1e-12
                fv.extend([np.log(c3w/c4w),(c3w-c4w)/(c3w+c4w)])
        F.append(fv)
    return np.array(F)


def train_eegnet(X_tr, y_tr, X_te, n_epochs=30):
    """快速训练EEGNet"""
    device = torch.device('cpu')
    X_tr_t=torch.FloatTensor(X_tr); X_te_t=torch.FloatTensor(X_te)
    y_tr_t=torch.LongTensor(y_tr)
    for ch in range(3):
        m=X_tr_t[:,ch].mean(); s=X_tr_t[:,ch].std()+1e-8
        X_tr_t[:,ch]=(X_tr_t[:,ch]-m)/s; X_te_t[:,ch]=(X_te_t[:,ch]-m)/s
    X_tr_t=X_tr_t.unsqueeze(1); X_te_t=X_te_t.unsqueeze(1)
    model=EEGNet(n_ch=3,n_times=X_tr.shape[2]).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=0.001)
    crit=nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        model.train()
        idx=torch.randperm(len(X_tr_t))
        for i in range(0,len(X_tr_t),64):
            xi=idx[i:i+64]; xb=X_tr_t[xi]; yb=y_tr_t[xi]
            opt.zero_grad(); out=model(xb.to(device)); loss=crit(out,yb.to(device)); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logits=model(X_te_t.to(device)); probs=F.softmax(logits,dim=1)[:,1].cpu().numpy()
    return probs


def main():
    print("=" * 60)
    print("BCICIV 2b - LOSO评估")
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
    
    # LOSO: EEGNet
    print("\n[2] LOSO - EEGNet...")
    n_per = [len(ep) for ep in epochs_list]
    loso_eegnet = []
    loso_feat = []
    loso_fusion = []
    
    pos = 0
    for si in range(len(epochs_list)):
        tr_idx = np.concatenate([np.arange(pos, pos+n_per[j]) for j in range(len(epochs_list)) if j != si])
        te_idx = np.arange(pos, pos+n_per[si])
        pos += n_per[si]
        
        X_tr, X_te = X_all[tr_idx], X_all[te_idx]
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]
        
        # EEGNet
        probs_en = train_eegnet(X_tr, y_tr, X_te, n_epochs=30)
        acc_en = accuracy_score(y_te, (probs_en > 0.5).astype(int))
        loso_eegnet.append(acc_en)
        
        # Deep features
        F_tr = deep_features(X_tr); F_te = deep_features(X_te)
        sc = StandardScaler()
        F_tr = np.nan_to_num(sc.fit_transform(F_tr), nan=0, posinf=10, neginf=-10)
        F_te = np.nan_to_num(sc.transform(F_te), nan=0, posinf=10, neginf=-10)
        clf = SVC(kernel='rbf', C=10, probability=True)
        clf.fit(F_tr, y_tr)
        prob_svm = clf.predict_proba(F_te)[:, 1]
        acc_feat = accuracy_score(y_te, clf.predict(F_te))
        loso_feat.append(acc_feat)
        
        # Fusion
        fused = 0.6 * probs_en + 0.4 * prob_svm
        acc_fus = accuracy_score(y_te, (fused > 0.5).astype(int))
        loso_fusion.append(acc_fus)
        
        print(f"  S{si+1}: EEGNet={acc_en:.1%} Feat={acc_feat:.1%} Fusion={acc_fus:.1%}")
    
    print(f"\n  EEGNet LOSO: {np.mean(loso_eegnet):.1%} +/- {np.std(loso_eegnet):.1%}")
    print(f"  Feat LOSO:   {np.mean(loso_feat):.1%} +/- {np.std(loso_feat):.1%}")
    print(f"  Fusion LOSO: {np.mean(loso_fusion):.1%} +/- {np.std(loso_fusion):.1%}")
    print("=" * 60)


if __name__ == '__main__':
    main()
