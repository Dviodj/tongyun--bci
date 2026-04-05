"""
生成 HELLO WORLD EEG 数据 - 正确保存为 RAW 格式
"""

import numpy as np
import mne
from pathlib import Path


# HELLO WORLD 莫尔斯码
HELLO_WORLD = "HELLO WORLD"
MORSE_MAP = {
    'H': [0,0,0,0], 'E': [0], 'L': [0,1,0,0], 'O': [1,1,1],
    'W': [0,1,1], 'R': [0,1,0], 'D': [1,0,0], ' ': [2]
}

# 构建信号序列
signal = []
for char in HELLO_WORLD:
    signal.extend(MORSE_MAP.get(char, []))

print(f"信号: {signal}")
print(f"长度: {len(signal)}")

# 参数
sfreq = 250
samples_per_symbol = 250

# 生成连续数据
total_samples = len(signal) * samples_per_symbol
data = np.zeros((3, total_samples))

for i, sig in enumerate(signal):
    start = i * samples_per_symbol
    end = start + samples_per_symbol
    t = np.linspace(0, 1, samples_per_symbol)
    
    # C3: 左手=0增强
    if sig == 0:
        data[0, start:end] = 15*np.sin(2*np.pi*10*t) + 3*np.sin(2*np.pi*20*t) + np.random.randn(samples_per_symbol)*2
    else:
        data[0, start:end] = 5*np.sin(2*np.pi*10*t) + 1*np.sin(2*np.pi*20*t) + np.random.randn(samples_per_symbol)*2
    
    # Cz
    data[1, start:end] = 8*np.sin(2*np.pi*10*t) + 2*np.sin(2*np.pi*20*t) + np.random.randn(samples_per_symbol)*2
    
    # C4: 右手=1增强
    if sig == 1:
        data[2, start:end] = 15*np.sin(2*np.pi*10*t) + 3*np.sin(2*np.pi*20*t) + np.random.randn(samples_per_symbol)*2
    else:
        data[2, start:end] = 5*np.sin(2*np.pi*10*t) + 1*np.sin(2*np.pi*20*t) + np.random.randn(samples_per_symbol)*2

print(f"数据形状: {data.shape}")

# 创建 Raw 对象
info = mne.create_info(ch_names=['C3', 'Cz', 'C4'], sfreq=sfreq, ch_types='eeg')
raw = mne.io.RawArray(data, info)

# 添加 events 作为 annotations
# 0 = 左手(点), 1 = 右手(划), 2 = 空格
descriptions = []
for sig in signal:
    if sig == 0:
        descriptions.append('769')  # 左手
    elif sig == 1:
        descriptions.append('770')  # 右手
    else:
        descriptions.append('space')

# 创建 events
samples = [i * samples_per_symbol + samples_per_symbol // 2 for i in range(len(signal))]
events = np.column_stack([samples, np.zeros(len(signal), dtype=int), signal])

print(f"Events:\n{events}")

# 添加 annotations
raw.set_annotations(mne.Annotations(
    onset=[s/sfreq for s in samples],
    duration=[0.5] * len(signal),
    description=descriptions
))

# 保存为 FIF
output_path = r"C:\Users\DoubleJ\Desktop\hello_world_eeg.fif"
raw.save(output_path, overwrite=True)
print(f"\n已保存: {output_path}")

# 验证
print("\n=== 验证 ===")
raw2 = mne.io.read_raw_gdf(output_path, preload=True, verbose=False)
print(f"通道: {raw2.ch_names}")
print(f"数据形状: {raw2.get_data().shape}")
print(f"Annotations: {raw2.annotations}")

# 加载测试
events2, event_id = mne.events_from_annotations(raw2, verbose=False)
print(f"\nEvents: {events2[:5]}")
print(f"Event ID: {event_id}")