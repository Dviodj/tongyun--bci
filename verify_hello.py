"""
验证 hello_world_eeg.gdf 文件能正确输出 HELLO WORLD
"""

import numpy as np
import mne

# Morse code dictionary
MORSE_CODE = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
    '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9',
    '-----': '0'
}

# Load the file
file_path = r"C:\Users\DoubleJ\Desktop\hello_world_eeg.gdf"
epochs = mne.read_epochs(file_path, verbose=False)
X = epochs.get_data()
y = epochs.events[:, 2]

print(f"数据形状: {X.shape}")
print(f"标签: {y}")

# Convert to morse
morse_output = []
char_buffer = []
text_output = []

for label in y:
    if label == 0:  # 左手 = 点
        morse_output.append("·")
        char_buffer.append(".")
    elif label == 1:  # 右手 = 划
        morse_output.append("－")
        char_buffer.append("-")
    else:  # 空格
        morse_output.append(" ")
        # 尝试转换
        if char_buffer:
            m = "".join(char_buffer)
            letter = MORSE_CODE.get(m, "?")
            text_output.append(letter)
            char_buffer = []
        text_output.append(" ")

print("\n=== 莫尔斯码 ===")
print("".join(morse_output))

print("\n=== 识别文字 ===")
print("".join(text_output))

# 重新解析
print("\n=== 按字符解析 ===")
current = ""
for label in y:
    if label == 2:  # 空格 - 字符间隔
        if current:
            letter = MORSE_CODE.get(current, "?")
            print(f"  {current} -> {letter}")
            current = ""
    else:
        if label == 0:
            current += "."
        else:
            current += "-"

if current:
    letter = MORSE_CODE.get(current, "?")
    print(f"  {current} -> {letter}")