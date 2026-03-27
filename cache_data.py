"""
缓存BCICIV数据到numpy文件
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['OMP_NUM_THREADS'] = '4'
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from pathlib import Path
import mne

DATA_DIR = r"D:\db\BCICIV_2b_gdf"
CACHE_DIR = r"D:\brainwave-morse\cache"
os.makedirs(CACHE_DIR, exist_ok=True)

print("Caching BCICIV 2b data...")

epochs_list = []
for fp in sorted(Path(DATA_DIR).glob("*T.gdf")):
    print(f"  {fp.name}...", end='', flush=True)
    try:
        raw = mne.io.read_raw_gdf(str(fp), preload=True, verbose=False)
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
        
        X = ep.get_data()
        y = ep.events[:, 2]
        name = fp.stem
        
        np.save(os.path.join(CACHE_DIR, f"{name}_X.npy"), X)
        np.save(os.path.join(CACHE_DIR, f"{name}_y.npy"), y)
        print(f" {len(X)} epochs")
    except Exception as e:
        print(f" ERROR: {str(e)[:40]}")

print("\nDone! Cached files in:", CACHE_DIR)
