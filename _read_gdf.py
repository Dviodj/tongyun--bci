import sys
sys.stdout.reconfigure(encoding='utf-8')
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import mne

fp = r"D:\db\BCICIV_2b_gdf\B0101T.gdf"

# BCICIV uses GDF 1.99 format with 6 channels
# Read header manually
with open(fp, 'rb') as f:
    f.seek(0, 2)
    fsize = f.tell()
    f.seek(0)
    h = bytearray(f.read(256))
    
    # Number of signals at byte 252 as uint8
    nchan = h[252]
    print(f"Number of channels: {nchan}")
    
    # Channel descriptor starts at byte 256
    # Each channel: 16 bytes label, 80 bytes type, etc.
    # GDF 1.x: 16 + 80 + 8 + 8 + 8 + 8 + 8 = 136 bytes per channel
    
    # Read channel labels
    ch_labels = []
    ch_sizes = []  # bytes per sample
    for i in range(nchan):
        offset = 256 + i * 136
        label = h[offset:offset+16].decode('ascii', errors='replace').strip()
        ch_labels.append(label)
        # Channel type at offset+80 (first byte = data type)
        dtype_code = h[offset + 80]
        print(f"  Channel {i}: {label}, dtype_code={dtype_code}")
        # Type codes: 3=int16, 4=int32, 5=float32, 6=float64
        if dtype_code == 3:
            ch_sizes.append(2)
        elif dtype_code == 4:
            ch_sizes.append(4)
        elif dtype_code == 5:
            ch_sizes.append(4)
        elif dtype_code == 6:
            ch_sizes.append(8)
        else:
            ch_sizes.append(2)  # default int16
    
    # Read nsamp per record from channel descriptors
    # nsamp at offset+16 as int32
    import struct
    total_samples_per_record = 0
    for i in range(nchan):
        offset = 256 + i * 136 + 16
        nsamp_ch = struct.unpack('<i', bytes(h[offset:offset+4]))[0]
        total_samples_per_record += nsamp_ch
        print(f"  Channel {i} nsamp: {nsamp_ch}")
    
    # Data starts at DATA_START
    DATA_START = 256 + nchan * 136
    print(f"\nData starts at: {DATA_START}")
    print(f"File size: {fsize}")
    print(f"Bytes per record: {total_samples_per_record * 2}")  # Assuming int16
    print(f"Number of records: {(fsize - DATA_START) // total_samples_per_record}")
    
    # Read a sample of data
    f.seek(DATA_START)
    sample = np.frombuffer(f.read(total_samples_per_record * 2), dtype='int16')
    print(f"\nFirst sample (int16): {sample}")
    print(f"Max/Min: {sample.max()}, {sample.min()}")
