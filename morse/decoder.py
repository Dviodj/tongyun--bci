"""摩斯密码解码器 - 将信号序列转换为文字"""

import time
from collections import deque
from typing import List, Optional
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import MORSE_CONFIG, MORSE_CODE_REVERSE


class MorseDecoder:
    """摩斯密码解码器"""
    
    def __init__(self):
        self.config = MORSE_CONFIG
        self.morse_table = MORSE_CODE_REVERSE
        
        # 状态
        self.current_morse = []  # 当前字符的摩斯序列
        self.current_word = []  # 当前单词
        self.result = []  # 完整结果
        
        # 时间记录
        self.last_signal_time = None
        self.last_type = None  # 'dot', 'dash', None
        
        # 信号缓冲（用于时间判定）
        self.signal_buffer = deque(maxlen=100)
    
    def process_signal(self, signal_type: int, timestamp: Optional[float] = None) -> str:
        """
        处理单个信号
        
        Args:
            signal_type: 0=左手(点), 1=右手(划)
            timestamp: 时间戳（默认当前时间）
            
        Returns:
            解码出的字符/单词（如果有）
        """
        if timestamp is None:
            timestamp = time.time()
        
        # 转换为摩斯符号
        morse_symbol = self.config['left_hand'] if signal_type == 0 \
                      else self.config['right_hand']
        
        # 记录信号
        self.signal_buffer.append((morse_symbol, timestamp))
        
        # 检查时间间隔，决定是字符内、字符间还是单词间
        output = ""
        
        if self.last_signal_time is not None:
            gap = timestamp - self.last_signal_time
            
            # 单词间隔
            if gap >= self.config['word_gap']:
                if self.current_morse:
                    char = self._decode_char(''.join(self.current_morse))
                    if char:
                        self.current_word.append(char)
                    self.current_morse = []
                
                if self.current_word:
                    word = ''.join(self.current_word)
                    self.result.append(word)
                    output = word + " "
                    self.current_word = []
            
            # 字符间隔
            elif gap >= self.config['char_gap']:
                if self.current_morse:
                    char = self._decode_char(''.join(self.current_morse))
                    if char:
                        self.current_word.append(char)
                        output = char
                    self.current_morse = []
        
        # 添加当前信号
        self.current_morse.append(morse_symbol)
        self.last_signal_time = timestamp
        self.last_type = morse_symbol
        
        return output
    
    def _decode_char(self, morse: str) -> Optional[str]:
        """将摩斯符号解码为单个字符"""
        return self.morse_table.get(morse, None)
    
    def flush(self) -> str:
        """
        强制输出剩余的内容（用于结束时）
        
        Returns:
            剩余的字符/单词
        """
        output = ""
        
        # 解码剩余的当前字符
        if self.current_morse:
            char = self._decode_char(''.join(self.current_morse))
            if char:
                self.current_word.append(char)
                output = char
            self.current_morse = []
        
        # 解码剩余的当前单词
        if self.current_word:
            word = ''.join(self.current_word)
            self.result.append(word)
            if output:
                output = " " + word
            else:
                output = word
            self.current_word = []
        
        return output
    
    def get_full_text(self) -> str:
        """获取完整的解码文本"""
        return ' '.join(self.result)
    
    def reset(self):
        """重置解码器状态"""
        self.current_morse = []
        self.current_word = []
        self.result = []
        self.last_signal_time = None
        self.last_type = None
        self.signal_buffer.clear()
