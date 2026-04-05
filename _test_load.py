import sys
sys.stdout.reconfigure(encoding='utf-8')
import warnings
warnings.filterwarnings('ignore')
import mne
import numpy as np

fp = r"D:\db\BCICIV_2b_gdf\B0101T.gdf"

print("Trying method 1: preload=True...")
try:
    raw = mne.io.read_raw_gdf(fp, preload=True, verbose=False)
    print(f"  SUCCESS: {raw.get_data().shape}")
except Exception as e:
    print(f"  FAILED: {e}")

print("\nTrying method 2: preload=False then load_data()...")
try:
    raw = mne.io.read_raw_gdf(fp, preload=False, verbose=False)
    raw.load_data()
    print(f"  SUCCESS: {raw.get_data().shape}")
except Exception as e:
    print(f"  FAILED: {e}")

print("\nTrying method 3: exclude Stim channel...")
try:
    raw = mne.io.read_raw_gdf(fp, preload=True, verbose=False, 
                              exclude=['Stim'])
    print(f"  SUCCESS: {raw.get_data().shape}")
except Exception as e:
    print(f"  FAILED: {e}")
