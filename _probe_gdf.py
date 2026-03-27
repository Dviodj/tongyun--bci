import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

fp = r"D:\db\BCICIV_2b_gdf\B0101T.gdf"

with open(fp, 'rb') as f:
    data = f.read(1024)

# Look for the problematic byte
print("First 1024 bytes (hex):")
for i in range(0, min(256, len(data)), 16):
    hex_part = ' '.join(f'{b:02x}' for b in data[i:i+16])
    ascii_part = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data[i:i+16])
    print(f"  {i:04x}: {hex_part}  {ascii_part}")

# Find byte 256 (0x100)
print(f"\nByte at position 0x100 (256): {data[0x100]:02x}")

# Check GDF header structure
# GDF type should be at position 1
print(f"\nGDF Type byte[1]: {data[1]:02x}")
print(f"GDF Type should be 3 (GDF 3.x) or 1 (GDF 1.x)")
