import sys
sys.stdout.reconfigure(encoding='utf-8')
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import mne

fp = r"D:\db\BCICIV_2b_gdf\B0101T.gdf"

# Try different GDF reading approaches
print("Method 1: Basic read_raw_gdf...")
try:
    raw = mne.io.read_raw_gdf(fp, preload=True, verbose=False)
    print(f"  SUCCESS: {raw.get_data().shape}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {str(e)[:100]}")

print("\nMethod 2: With stim_channel='Stim'...")
try:
    raw = mne.io.read_raw_gdf(fp, preload=True, verbose=False, stim_channel='Stim')
    print(f"  SUCCESS: {raw.get_data().shape}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}")

print("\nMethod 3: With montage=None...")
try:
    raw = mne.io.read_raw_gdf(fp, preload=True, verbose=False, montage=None)
    print(f"  SUCCESS: {raw.get_data().shape}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}")

print("\nMethod 4: Binary read custom...")
with open(fp, 'rb') as f:
    # Read header to get parameters
    f.seek(0)
    header = f.read(256)
    # GDF 1.x: positions 1-4 = "GDF " 
    # Position 188-192: number of channels (int32)
    nchan = int.from_bytes(header[184:188], 'int32')
    print(f"  Nchannels from header: {nchan}")
    # Position 236-244: sampling rate (float64)
    sfreq = np.frombuffer(header[236:244], dtype='float64')[0]
    print(f"  Sampling rate: {sfreq}")
