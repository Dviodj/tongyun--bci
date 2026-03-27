"""BCICIV 2b - 简化快速评估"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
os.environ['OMP_NUM_THREADS'] = '4'
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import mne
import torch, torch.nn as nn, torch.nn.functional as F

SFREQ = 250
DATA_DIR = r"D:\db\BCICIV_2b_gdf"
OUT_FILE = r"C:\Users\DoubleJ\Desktop\bciciv_results.txt"

def log(msg=''):
    print(msg, flush=True)

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
    def __init__(self, n_ch=3, n_times=1001, F1=8, D=2, F2=16):
        super().__init__()
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.conv2 = nn.Conv2d(F1, F1*D, (n_ch, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1*D)
        self.pool1 = nn.AvgPool2d((1, 4)); self.drop1 = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(F1*D, F1*D, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F1*D)
        self.pool2 = nn.AvgPool2d((1, 8)); self.drop2 = nn.Dropout(0.5)
        self.fc = nn.Linear(F2 * (n_times // 4 // 8), 2)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x); x = self.drop1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x); x = self.drop2(x)
        return self.fc(x.view(x.size(0), -1))

def feat(X):
    def bp(s,f1,f2):
        nyq=SFREQ/2; b,a=butter(4,[f1/nyq,f2/nyq],btype='band')
        return filtfilt(b,a,s)
    F=[]
    for ep in X:
        fv=[]
        for ch in range(3):
            s=ep[ch]
            for f1,f2 in [(8,13),(13,30),(6,13),(18,25)]:
                p=np.var(bp(s,f1,f2)); fv.extend([10*np.log10(p+1e-12),p])
            fv.extend([np.mean(s),np.std(s)])
        for f1,f2 in [(8,13),(13,30),(6,13),(18,25)]:
            c3=np.var(bp(ep[0],f1,f2))+1e-12; c4=np.var(bp(ep[2],f1,f2))+1e-12
            fv.extend([np.log(c3/c4),(c3-c4)/(c3+c4)])
        F.append(fv)
    return np.array(F)

def train_en(X_tr, y_tr, X_te, n_epochs=30):
    device = torch.device('cpu')
    X_tr_t=torch.FloatTensor(X_tr); X_te_t=torch.FloatTensor(X_te); y_tr_t=torch.LongTensor(y_tr)
    for ch in range(3):
        m=X_tr_t[:,ch].mean(); s=X_tr_t[:,ch].std()+1e-8
        X_tr_t[:,ch]=(X_tr_t[:,ch]-m)/s; X_te_t[:,ch]=(X_te_t[:,ch]-m)/s
    X_tr_t.unsqueeze_(1); X_te_t.unsqueeze_(1)
    model=EEGNet(n_ch=3,n_times=X_tr.shape[2]).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=0.001)
    for epoch in range(n_epochs):
        model.train()
        idx=torch.randperm(len(X_tr_t))
        for i in range(0,len(X_tr_t),64):
            xi=idx[i:i+64]
            opt.zero_grad(); model(X_tr_t[xi].to(device)).mean().backward(); opt.step()
    model.eval()
    with torch.no_grad():
        probs=F.softmax(model(X_te_t.to(device)),dim=1)[:,1].cpu().numpy()
    return probs

def main():
    log("=" * 50)
    log("BCICIV 2b - 最终评估")
    log("=" * 50)
    
    log("\n[1] Loading...")
    epochs_list = []
    for fp in sorted(Path(DATA_DIR).glob("*T.gdf")):
        try:
            ep = load_subject(str(fp)); epochs_list.append(ep)
            log(f"  {fp.name}: {len(ep)}")
        except: log(f"  {fp.name}: ERROR")
    
    X_all = np.vstack([e.get_data() for e in epochs_list])
    y_all = np.concatenate([e.events[:,2] for e in epochs_list])
    n_per = [len(ep) for ep in epochs_list]
    log(f"\nTotal: {X_all.shape}")
    
    # 5-fold CV
    log("\n[2] 5折交叉验证...")
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    cv_en=[]; cv_ft=[]
    
    for fi,(tr,te) in enumerate(cv.split(X_all,y_all)):
        X_tr,X_te,y_tr,y_te = X_all[tr],X_all[te],y_all[tr],y_all[te]
        
        # EEGNet
        probs_en = train_en(X_tr, y_tr, X_te, n_epochs=30)
        acc_en = accuracy_score(y_te, (probs_en>0.5).astype(int))
        cv_en.append(acc_en)
        
        # Features
        F_tr=feat(X_tr); F_te=feat(X_te)
        sc=StandardScaler(); F_tr=np.nan_to_num(sc.fit_transform(F_tr),nan=0,posinf=10,neginf=-10)
        F_te=np.nan_to_num(sc.transform(F_te),nan=0,posinf=10,neginf=-10)
        clf=SVC(kernel='rbf',C=10,probability=True); clf.fit(F_tr,y_tr)
        prob_ft=clf.predict_proba(F_te)[:,1]
        acc_ft=clf.score(F_te,y_te)
        cv_ft.append(acc_ft)
        
        # Fusion
        fused=0.6*probs_en+0.4*prob_ft
        acc_fus=accuracy_score(y_te,(fused>0.5).astype(int))
        log(f"  Fold {fi+1}: EN={acc_en:.1%} Feat={acc_ft:.1%} Fus={acc_fus:.1%}")
    
    log(f"\n  EN Mean: {np.mean(cv_en):.1%}")
    log(f"  Feat Mean: {np.mean(cv_ft):.1%}")
    
    # LOSO
    log("\n[3] LOSO评估...")
    loso_en=[]; loso_ft=[]
    pos=0
    for si in range(len(epochs_list)):
        tr_idx=np.concatenate([np.arange(pos,pos+n_per[j]) for j in range(len(epochs_list)) if j!=si])
        te_idx=np.arange(pos,pos+n_per[si]); pos+=n_per[si]
        X_tr,X_te,y_tr,y_te=X_all[tr_idx],X_all[te_idx],y_all[tr_idx],y_all[te_idx]
        
        probs_en=train_en(X_tr,y_tr,X_te,n_epochs=30)
        acc_en=accuracy_score(y_te,(probs_en>0.5).astype(int))
        loso_en.append(acc_en)
        
        F_tr=feat(X_tr); F_te=feat(X_te)
        sc=StandardScaler(); F_tr=np.nan_to_num(sc.fit_transform(F_tr),nan=0,posinf=10,neginf=-10)
        F_te=np.nan_to_num(sc.transform(F_te),nan=0,posinf=10,neginf=-10)
        clf=SVC(kernel='rbf',C=10,probability=True); clf.fit(F_tr,y_tr)
        acc_ft=clf.score(F_te,y_te)
        loso_ft.append(acc_ft)
        
        log(f"  S{si+1}: EN={acc_en:.1%} Feat={acc_ft:.1%}")
    
    log(f"\n  EN LOSO: {np.mean(loso_en):.1%} +/- {np.std(loso_en):.1%}")
    log(f"  Feat LOSO: {np.mean(loso_ft):.1%} +/- {np.std(loso_ft):.1%}")
    log("=" * 50)
    
    # Save results
    results = {
        '5cv_eegnet': float(np.mean(cv_en)),
        '5cv_feat': float(np.mean(cv_ft)),
        'loso_eegnet': float(np.mean(loso_en)),
        'loso_feat': float(np.mean(loso_ft)),
        'loso_eegnet_std': float(np.std(loso_en)),
        'loso_feat_std': float(np.std(loso_ft)),
    }
    with open(r"C:\Users\DoubleJ\Desktop\bciciv_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to desktop")

if __name__ == '__main__':
    main()
