"""
生成 HELLO WORLD 莫尔斯码脑电数据
"""

import numpy as np
import mne
from pathlib import Path


# HELLO WORLD 莫尔斯编码
# 点 = 左手(0), 划 = 右手(1)
MORSE_HELLO_WORLD = {
    'H': '....',    # LLLL
    'E': '.',       # L
    'L': '.-..',    # LRLL
    'L': '.-..',    # LRLL
    'O': '---',     # RRR
    'W': '.--',     # RRL
    'O': '---',     # RRR
    'R': '.-.',     # LRL
    'L': '.-..',    # LRLL
    'D': '-..',     # RLL
}

# 转换为信号序列
def text_to_signal(text):
    """将文字转为左右手信号序列 (0=左手, 1=右手)"""
    signal = []
    for char in text.upper():
        if char == ' ':
            signal.append(2)  # 空格标记
        elif char in MORSE_HELLO_WORLD:
            for m in MORSE_HELLO_WORLD[char]:
                signal.append(0 if m == '.' else 1)
    return signal


def create_eeg_file(output_path):
    """创建包含HELLO WORLD信号的GDF文件"""
    
    # 获取信号序列
    signal_seq = text_to_signal("HELLO WORLD")
    print(f"信号序列长度: {len(signal_seq)}")
    print(f"信号序列: {signal_seq}")
    
    # 每个"点"或"划"持续 1 秒 (250 samples at 250Hz)
    n_samples_per_symbol = 250
    n_epochs = len(signal_seq)
    
    # 创建模拟EEG数据 (3 channels, C3/Cz/C4)
    sfreq = 250
    duration = n_samples_per_symbol / sfreq  # 1秒 per symbol
    
    # 基础参数
    t = np.linspace(0, duration, n_samples_per_symbol)
    
    # C3 (左手运动想象时活跃) - 当 signal=0 时增强
    # C4 (右手运动想象时活跃) - 当 signal=1 时增强
    
    X = []
    y = []
    
    for sig in signal_seq:
        # 创建模拟的EEG epoch
        # 使用mu节律 (8-13Hz) 的alpha波作为基础
        base_mu = 10  # mu rhythm 10Hz
        base_beta = 20  # beta rhythm 20Hz
        
        # C3信号
        if sig == 0:  # 左手想象 - C3增强
            c3 = 15 * np.sin(2 * np.pi * base_mu * t) + 5 * np.sin(2 * np.pi * base_beta * t)
            c3 += np.random.randn(n_samples_per_symbol) * 2
        else:  # 右手或空格
            c3 = 5 * np.sin(2 * np.pi * base_mu * t) + 2 * np.sin(2 * np.pi * base_beta * t)
            c3 += np.random.randn(n_samples_per_symbol) * 2
        
        # Cz信号
        cz = 8 * np.sin(2 * np.pi * base_mu * t) + 3 * np.sin(2 * np.pi * base_beta * t)
        cz += np.random.randn(n_samples_per_symbol) * 2
        
        # C4信号
        if sig == 1:  # 右手想象 - C4增强
            c4 = 15 * np.sin(2 * np.pi * base_mu * t) + 5 * np.sin(2 * np.pi * base_beta * t)
            c4 += np.random.randn(n_samples_per_symbol) * 2
        else:  # 左手或空格
            c4 = 5 * np.sin(2 * np.pi * base_mu * t) + 2 * np.sin(2 * np.pi * base_beta * t)
            c4 += np.random.randn(n_samples_per_symbol) * 2
        
        # 合并通道
        epoch = np.array([c3, cz, c4])
        X.append(epoch)
        
        # 标签: 0=左手, 1=右手, 2=空格(休息)
        y.append(sig)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n数据形状: {X.shape}")
    print(f"标签: {y}")
    print(f"左手(点): {np.sum(y==0)}, 右手(划): {np.sum(y==1)}, 空格: {np.sum(y==2)}")
    
    # 创建MNE对象
    info = mne.create_info(ch_names=['C3', 'Cz', 'C4'], sfreq=sfreq, ch_types='eeg')
    epochs = mne.EpochsArray(X, info)
    
    # 保存为GDF
    epochs.save(output_path, overwrite=True)
    print(f"\n已保存到: {output_path}")
    
    # 同时保存为numpy格式
    npy_path = output_path.replace('.gdf', '_X.npy')
    np.save(npy_path, X)
    
    y_path = output_path.replace('.gdf', '_y.npy')
    np.save(y_path, y)
    
    print(f"已保存numpy: {npy_path}, {y_path}")
    
    return X, y


# 生成文件
output_path = r"C:\Users\DoubleJ\Desktop\hello_world_eeg.gdf"
X, y = create_eeg_file(output_path)

print("\n=== 使用说明 ===")
print(f"文件: {output_path}")
print("在可视化界面中加载此文件，播放即可看到HELLO WORLD的莫尔斯码输出")
print("\n莫尔斯对照:")
print("H: .... (点点点点)")
print("E: . (点)")
print("L: .-.. (点划点点)")
print("L: .-..")
print("O: --- (划划划)")
print("W: .-- (点划划)")
print("O: ---")
print("R: .-. (点划点)")
print("L: .-..")
print("D: -.. (划点点)")