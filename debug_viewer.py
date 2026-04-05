"""调试 eeg_viewer - 添加日志"""
import sys
sys.path.insert(0, '.')

# 模拟运行
print("=== 模拟 Viewer 运行 ===")

import numpy as np
import mne

file_path = r'C:\Users\DoubleJ\Desktop\hello_world_eeg.fif'
raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
events, event_id = mne.events_from_annotations(raw, verbose=False)

left_id = event_id.get('769')
right_id = event_id.get('770')
print(f"Event ID: left={left_id}, right={right_id}")

valid_types = [left_id, right_id]
events = events[np.isin(events[:, 2], valid_types)]
print(f"Events count: {len(events)}")

tmin, tmax = -0.5, 4.0
epochs = mne.Epochs(raw, events, 
                    event_id={'left': left_id, 'right': right_id},
                    tmin=tmin, tmax=tmax, preload=True, verbose=False)

X = epochs.get_data()
y = epochs.events[:, 2]
y = np.where(y == left_id, 0, 1)
print(f"y values: {y}")

# 模拟 OutputPanel 逻辑
import time

MORSE_CODE = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z'
}

morse_buffer = ""
char_buffer = ""
last_event_time = time.time()

# 模拟 100ms 间隔播放
results = []
for i, event in enumerate(y):
    # 模拟时间间隔
    time.sleep(0.1)
    current_time = time.time()
    
    # 超时检测 (3秒)
    if current_time - last_event_time > 3.0:
        if char_buffer:
            letter = MORSE_CODE.get(char_buffer, '?')
            results.append(f"转换: {char_buffer} -> {letter}")
            char_buffer = ""
        morse_buffer = ""
    
    last_event_time = current_time
    
    if event == 0:
        morse_buffer += "."
        char_buffer += "."
    else:
        morse_buffer += "-"
        char_buffer += "-"
    
    print(f"[{i}] event={event}, buffer={char_buffer}")
    
    # 手动触发转换 (每4个字符)
    if len(char_buffer) >= 4:
        letter = MORSE_CODE.get(char_buffer, '?')
        results.append(f"转换: {char_buffer} -> {letter}")
        char_buffer = ""

print("\n=== 转换结果 ===")
for r in results:
    print(r)