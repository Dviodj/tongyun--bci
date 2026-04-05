"""主处理流程 - 将脑电数据识别并转换为摩斯密码输出文字"""

import os
import numpy as np
from typing import Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_CONFIG, CLASSIFIER_CONFIG
from data.loader import load_eeg_data, create_epochs, get_labels
from data.preprocessing import preprocess_raw, extract_features
from models.classifier import EEGClassifier
from models.metabci_wrapper import MetaBCIClassifier
from morse.decoder import MorseDecoder


class BrainwaveMorsePipeline:
    """脑电 → 摩斯密码 → 文字 完整处理流程"""
    
    def __init__(self, mode: str = 'custom'):
        """
        初始化处理流程
        
        Args:
            mode: 处理模式 ('custom' 或 'metabci')
        """
        self.mode = mode
        
        # 初始化分类器
        if mode == 'custom':
            self.classifier = EEGClassifier()
        elif mode == 'metabci':
            self.classifier = MetaBCIClassifier()
        else:
            raise ValueError(f"不支持的模式: {mode}")
        
        # 初始化摩斯解码器
        self.decoder = MorseDecoder()
        
        self.is_trained = False
    
    def train(self, data_dir: str):
        """
        训练模型
        
        Args:
            data_dir: 训练数据目录
        """
        print(f"开始训练 (模式: {self.mode})...")
        print(f"数据目录: {data_dir}")
        
        # 收集所有数据文件
        data_files = self._find_data_files(data_dir)
        
        if not data_files:
            raise ValueError(f"在 {data_dir} 中未找到数据文件！")
        
        # 加载和处理数据
        all_features = []
        all_labels = []
        all_epochs_data = []
        
        for file_path in data_files:
            print(f"处理文件: {os.path.basename(file_path)}")
            
            # 加载数据
            raw, events = load_eeg_data(file_path)
            
            # 预处理
            raw_processed = preprocess_raw(raw)
            
            if events is not None:
                # 创建 epochs
                epochs = create_epochs(raw_processed, events)
                
                # 提取标签
                labels = get_labels(epochs)
                
                if self.mode == 'custom':
                    # 提取特征（自定义模式）
                    features = extract_features(epochs)
                    all_features.append(features)
                else:
                    # 保存原始 epoch 数据（MetaBCI 模式）
                    all_epochs_data.append(epochs.get_data())
                
                all_labels.append(labels)
        
        # 合并数据
        if all_labels:
            y = np.concatenate(all_labels)
            
            if self.mode == 'custom':
                X = np.vstack(all_features)
            else:
                X = np.vstack(all_epochs_data)
            
            print(f"训练样本数: {len(y)}")
            print(f"特征维度: {X.shape}")
            
            # 训练分类器
            self.classifier.fit(X, y)
            self.is_trained = True
            
            print("✓ 训练完成！")
        else:
            raise ValueError("未找到有效的事件数据！")
    
    def predict_single(self, epoch_data: np.ndarray) -> Tuple[int, str]:
        """
        预测单个 epoch
        
        Args:
            epoch_data: 单个 epoch 数据 (n_channels, n_times)
                       或特征向量 (n_features,)
            
        Returns:
            signal_type: 0=左手, 1=右手
            morse_char: '.' 或 '-'
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练！")
        
        if epoch_data.ndim == 2:
            # 原始 EEG 数据
            if self.mode == 'custom':
                # 先转成 mne epoch 再提取特征
                features = self._extract_features_from_array(epoch_data)
                pred, _ = self.classifier.predict_single(features)
            else:
                pred = self.classifier.predict(epoch_data[np.newaxis, ...])[0]
        else:
            # 已经是特征
            pred, _ = self.classifier.predict_single(epoch_data)
        
        morse_char = '.' if pred == 0 else '-'
        return pred, morse_char
    
    def predict_stream(self, epochs_generator) -> str:
        """
        处理连续的 epoch 流
        
        Args:
            epochs_generator: epoch 数据生成器
            
        Yields:
            解码出的字符/单词
        """
        self.decoder.reset()
        
        for epoch_data, timestamp in epochs_generator:
            pred, _ = self.predict_single(epoch_data)
            output = self.decoder.process_signal(pred, timestamp)
            
            if output:
                yield output
        
        # 刷新剩余内容
        final_output = self.decoder.flush()
        if final_output:
            yield final_output
    
    def get_full_text(self) -> str:
        """获取完整的解码文本"""
        return self.decoder.get_full_text()
    
    def _find_data_files(self, data_dir: str) -> list:
        """查找数据文件"""
        supported_extensions = ['.edf', '.fif', '.set', '.mat']
        data_files = []
        
        if os.path.isfile(data_dir):
            if os.path.splitext(data_dir)[1].lower() in supported_extensions:
                data_files.append(data_dir)
        else:
            for root, _, files in os.walk(data_dir):
                for file in files:
                    if os.path.splitext(file)[1].lower() in supported_extensions:
                        data_files.append(os.path.join(root, file))
        
        return data_files
    
    def _extract_features_from_array(self, data: np.ndarray) -> np.ndarray:
        """从 numpy 数组中提取特征（简单版）"""
        # 简单的特征提取
        bands = [(0.5, 4), (4, 8), (8, 13), (13, 30)]
        features = []
        
        # 计算简单功率
        mean_power = np.mean(data ** 2, axis=1)
        features.append(mean_power)
        
        # 标准差
        std_power = np.std(data, axis=1)
        features.append(std_power)
        
        return np.concatenate(features)
