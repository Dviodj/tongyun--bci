import sys
sys.stdout.reconfigure(encoding='utf-8')
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import neo

fp = r"D:\db\BCICIV_2b_gdf\B0101T.gdf"

print("Loading with neo...")
reader = neo.rawio.GDFRawIO(filename=fp)
reader.parse_header()
print(f"  Channels: {reader.header['signal_channels']}")
print(f"  Events: {reader.header['event_channels']}")

# Load data
reader.load_cached_raw_data()
raw_data = reader._raw_data
print(f"  Data shape: {raw_data.shape}")
print(f"  Sampling rate: {reader.get_signal_sampling_rate(0)}")

# Get events
events = reader.get_all_events()
print(f"  Events: {events}")
