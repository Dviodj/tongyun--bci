"""
生成 HELLO WORLD 测试数据 - 简化版（无空格标记）
直接在信号中嵌入字母边界信息
"""

import numpy as np
import mne

# 正确的莫尔斯码
MORSE = {
    'H': '....', 'E': '.', 'L': '.-..', 'O': '---',
    'W': '.--', 'R': '.-.', 'D': '-..', ' ': ' '
}

text = "HELLO WORLD"
print(f"文本: {text}")

# 构建信号序列 - 添加特殊标记（用连续的1表示字母边界）
signal_seq = []
for char in text:
    if char == ' ':
        signal_seq.extend([2, 2])  # 单词间空格
    else:
        morse = MORSE.get(char, '')
        for m in morse:
            if m == '.':
                signal_seq.append(0)
            elif m == '-':
                signal_seq.append(1)
        # 字母结束标记 - 后面加一个短暂的"无信号"作为分隔
        signal_seq.append(3)  # 字母边界标记

# 打印莫尔斯
morse_str = ''
for s in signal_seq:
    if s == 0: morse_str += '.'
    elif s == 1: morse_str += '-'
    elif s == 2: morse_str += ' '
    elif s == 3: morse_str += '|'
print(f"莫尔斯: {morse_str}")
print(f"信号数: {len(signal_seq)}")

# 参数
sfreq = 250
samples_per_symbol = 250

# 生成 EEG 数据
total_samples = len(signal_seq) * samples_per_symbol
data = np.zeros((3, total_samples))

for i, sig in enumerate(signal_seq):
    start = i * samples_per_symbol
    end = start + samples_per_symbol
    t = np.linspace(0, 1, samples_per_symbol)
    
    base = 5 * np.sin(2 * np.pi * 10 * t) + 2 * np.sin(2 * np.pi * 20 * t)
    noise = np.random.randn(samples_per_symbol) * 2
    
    if sig == 0:  # 点 - C3增强
        data[0, start:end] = 15 * np.sin(2 * np.pi * 10 * t) + noise
    else:
        data[0, start:end] = base + noise
    
    data[1, start:end] = base + noise
    
    if sig == 1:  # 划 - C4增强
        data[2, start:end] = 15 * np.sin(2 * np.pi * 10 * t) + noise
    else:
        data[2, start:end] = base + noise

print(f"数据: {data.shape}")

# 创建 Raw
info = mne.create_info(ch_names=['C3', 'Cz', 'C4'], sfreq=sfreq, ch_types='eeg')
raw = mne.io.RawArray(data, info)

# Annotations - 将边界标记(3)映射为特殊事件
descriptions = []
samples = []
for i, sig in enumerate(signal_seq):
    if sig == 0:
        descriptions.append('769')  # 左手-点
    elif sig == 1:
        descriptions.append('770')  # 右手-划
    elif sig == 2:
        descriptions.append('space')
    else:  # sig == 3 边界
        descriptions.append('boundary')
    samples.append(i * samples_per_symbol + samples_per_symbol // 2)

raw.set_annotations(mne.Annotations(
    onset=[s / sfreq for s in samples],
    duration=[0.5] * len(signal_seq),
    description=descriptions
))

# 保存
output_path = r"C:\Users\DoubleJ\Desktop\helloworldtest.fif"
raw.save(output_path, overwrite=True)
print(f"\n已保存: {output_path}")

# 验证
raw2 = mne.io.read_raw_fif(output_path, preload=True, verbose=False)
events, event_id = mne.events_from_annotations(raw2, verbose=False)
print(f"Events: {len(events)}")
print(f"Event ID: {event_id}")
sig_list = [e[2] for e in events]
print(f"信号: {sig_list}")