import sys
sys.stdout.reconfigure(encoding='utf-8')
import warnings
warnings.filterwarnings('ignore')
import mne
import numpy as np

fp = r"D:\db\BCICIV_2b_gdf\B0101T.gdf"

# Try with different parameters
for name, kwargs in [
    ("default", {}),
    ("exclude=[]", {'exclude': []}),
    ("eog=[]", {'eog': []}),
    ("misc=[]", {'misc': []}),
]:
    print(f"Trying {name}...")
    try:
        raw = mne.io.read_raw_gdf(fp, preload=True, verbose=False, **kwargs)
        print(f"  SUCCESS: {raw.get_data().shape}, chs={len(raw.ch_names)}")
        print(f"  Channels: {raw.ch_names}")
        break
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {str(e)[:80]}")
