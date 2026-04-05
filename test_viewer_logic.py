"""测试 Viewer 逻辑"""
import sys
sys.path.insert(0, '.')
import numpy as np
import mne
from pathlib import Path

file_path = r'C:\Users\DoubleJ\Desktop\hello_world_eeg.fif'
raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
events, event_id = mne.events_from_annotations(raw, verbose=False)

left_id = event_id.get('769')
right_id = event_id.get('770')

valid_types = [left_id, right_id] if left_id and right_id else []
events = events[np.isin(events[:, 2], valid_types)]

tmin, tmax = -0.5, 4.0
epochs = mne.Epochs(raw, events, 
                    event_id={'left': left_id, 'right': right_id},
                    tmin=tmin, tmax=tmax, preload=True, verbose=False)

X = epochs.get_data()
y = epochs.events[:, 2]
y = np.where(y == left_id, 0, 1)

print(f'数据: {len(X)} epochs')
print(f'标签: {y}')
print(f'左手(0): {np.sum(y==0)}, 右手(1): {np.sum(y==1)}')

# Test morse conversion
MORSE_CODE = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z'
}

morse_buffer = ''
char_buffer = ''

for label in y:
    if label == 0:
        morse_buffer += '.'
        char_buffer += '.'
    else:
        morse_buffer += '-'
        char_buffer += '-'
    
    if len(char_buffer) >= 4:
        letter = MORSE_CODE.get(char_buffer, '?')
        print(f'字符: {char_buffer} -> {letter}')
        char_buffer = ''

print(f'\n最终莫尔斯: {morse_buffer}')