"""摩斯密码编码器 - 将文字转换为摩斯序列（用于测试/模拟）"""

import time
from typing import List, Tuple
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import MORSE_CONFIG, MORSE_CODE


class MorseEncoder:
    """摩斯密码编码器"""
    
    def __init__(self):
        self.config = MORSE_CONFIG
        self.morse_table = MORSE_CODE
    
    def encode_text(self, text: str) -> List[Tuple[int, float]]:
        """
        将文字编码为摩斯信号序列
        
        Args:
            text: 输入文字（英文/数字）
            
        Returns:
            signals: 信号列表 [(signal_type, duration), ...]
                    signal_type: 0=点(左手), 1=划(右手)
                    duration: 持续时间 (秒)
        """
        text = text.upper()
        signals = []
        
        for i, char in enumerate(text):
            if char == ' ':
                # 单词间隔
                signals.append((-1, self.config['word_gap']))
                continue
            
            if char not in self.morse_table:
                continue
            
            morse = self.morse_table[char]
            
            for j, symbol in enumerate(morse):
                # 点或划
                if symbol == '.':
                    signal_type = 0  # 左手 = 点
                    duration = self.config['dot_duration']
                elif symbol == '-':
                    signal_type = 1  # 右手 = 划
                    duration = self.config['dash_duration']
                else:
                    continue
                
                signals.append((signal_type, duration))
                
                # 符号间间隔（不是最后一个符号时）
                if j < len(morse) - 1:
                    signals.append((-1, self.config['dot_duration']))
            
            # 字符间间隔（不是最后一个字符时）
            if i < len(text) - 1 and text[i+1] != ' ':
                signals.append((-1, self.config['char_gap']))
        
        return signals
    
    def simulate_stream(self, text: str, callback):
        """
        模拟实时信号流
        
        Args:
            text: 输入文字
            callback: 回调函数 callback(signal_type, timestamp)
                    signal_type: 0=左手, 1=右手, -1=间隔
        """
        signals = self.encode_text(text)
        start_time = time.time()
        current_time = start_time
        
        for signal_type, duration in signals:
            if signal_type != -1:
                callback(signal_type, current_time)
            
            current_time += duration
            time.sleep(duration * 0.1)  # 加速模拟（0.1倍速）
