"""实时处理模块 - 用于实时脑电信号采集和处理"""

import time
import numpy as np
from typing import Optional, Callable
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.pipeline import BrainwaveMorsePipeline


class RealTimeProcessor:
    """实时脑电信号处理器"""
    
    def __init__(self, pipeline: BrainwaveMorsePipeline):
        """
        初始化实时处理器
        
        Args:
            pipeline: 处理流程对象
        """
        self.pipeline = pipeline
        self.buffer = []
        self.sample_rate = None
        self.last_output = ""
        self.on_text_callback = None
    
    def set_on_text_callback(self, callback: Callable[[str], None]):
        """
        设置文本输出回调
        
        Args:
            callback: 回调函数，参数为解码出的文本
        """
        self.on_text_callback = callback
    
    def feed_data(self, data: np.ndarray, timestamp: Optional[float] = None):
        """
        输入新的脑电数据
        
        Args:
            data: 脑电数据 (n_channels, n_samples)
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.buffer.append((data, timestamp))
        
        # 当缓冲区足够大时进行处理
        if len(self.buffer) >= 10:  # 假设需要10个epoch
            self._process_buffer()
    
    def _process_buffer(self):
        """处理缓冲区中的数据"""
        if not self.buffer:
            return
        
        for data, timestamp in self.buffer:
            try:
                pred, morse = self.pipeline.predict_single(data)
                
                output = self.pipeline.decoder.process_signal(pred, timestamp)
                
                if output:
                    self.last_output = output
                    if self.on_text_callback:
                        self.on_text_callback(output)
            except Exception as e:
                print(f"处理错误: {e}")
        
        self.buffer.clear()
    
    def get_current_text(self) -> str:
        """获取当前解码出的文本"""
        return self.pipeline.get_full_text()
    
    def reset(self):
        """重置处理器状态"""
        self.buffer.clear()
        self.last_output = ""
        self.pipeline.decoder.reset()


class MockEEGStream:
    """模拟 EEG 数据流（用于测试）"""
    
    def __init__(self, sample_rate: int = 250, n_channels: int = 3):
        """
        初始化模拟流
        
        Args:
            sample_rate: 采样率
            n_channels: 通道数
        """
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self.is_running = False
    
    def generate_epoch(self, signal_type: int = 0) -> np.ndarray:
        """
        生成一个模拟 epoch
        
        Args:
            signal_type: 0=左手, 1=右手
            
        Returns:
            epoch: 模拟的 epoch 数据
        """
        n_samples = int(self.sample_rate * 1.0)  # 1秒 epoch
        epoch = np.random.randn(self.n_channels, n_samples) * 0.1
        
        # 添加简单的信号特征
        if signal_type == 0:
            # 左手：C3 通道增强
            epoch[0, :] *= 1.5
        else:
            # 右手：C4 通道增强
            epoch[-1, :] *= 1.5
        
        return epoch
    
    def start_stream(self, callback: Callable[[np.ndarray, float], None], 
                    duration: float = 10.0):
        """
        启动模拟流
        
        Args:
            callback: 数据回调
            duration: 持续时间 (秒)
        """
        self.is_running = True
        start_time = time.time()
        
        while self.is_running and (time.time() - start_time) < duration:
            # 随机生成信号
            signal_type = np.random.randint(0, 2)
            epoch = self.generate_epoch(signal_type)
            callback(epoch, time.time())
            
            time.sleep(1.0)
        
        self.is_running = False
    
    def stop_stream(self):
        """停止模拟流"""
        self.is_running = False
