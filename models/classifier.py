"""自定义分类器模块"""

import numpy as np
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from typing import Tuple, Optional
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import CLASSIFIER_CONFIG


class EEGClassifier:
    """EEG 信号分类器（左右手识别）"""
    
    def __init__(self, mode: str = None):
        """
        初始化分类器
        
        Args:
            mode: 分类器模式 ('svm', 'lda', 'random_forest')
        """
        self.mode = mode or CLASSIFIER_CONFIG['custom']['classifier']
        self.scaler = StandardScaler()
        
        # 初始化分类器
        if self.mode == 'svm':
            self.clf = SVC(kernel='rbf', probability=True)
        elif self.mode == 'lda':
            self.clf = LDA()
        elif self.mode == 'random_forest':
            self.clf = RandomForestClassifier(n_estimators=100)
        else:
            raise ValueError(f"不支持的分类器: {self.mode}")
        
        self.is_trained = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练分类器
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签 (0=左手, 1=右手)
        """
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练
        self.clf.fit(X_scaled, y)
        self.is_trained = True
        
        # 打印交叉验证准确率
        cv_scores = cross_val_score(self.clf, X_scaled, y, cv=5)
        print(f"交叉验证准确率: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            
        Returns:
            predictions: 预测标签 (0=左手, 1=右手)
        """
        if not self.is_trained:
            raise ValueError("分类器尚未训练！")
        
        X_scaled = self.scaler.transform(X)
        return self.clf.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            
        Returns:
            probabilities: 概率矩阵 (n_samples, 2)
        """
        if not self.is_trained:
            raise ValueError("分类器尚未训练！")
        
        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X_scaled)
    
    def predict_single(self, X: np.ndarray) -> Tuple[int, float]:
        """
        预测单个样本
        
        Args:
            X: 单样本特征 (n_features,) 或 (1, n_features)
            
        Returns:
            pred: 预测标签 (0=左手, 1=右手)
            prob: 预测概率
        """
        if X.ndim == 1:
            X = X[np.newaxis, :]
        
        pred = self.predict(X)[0]
        prob = self.predict_proba(X)[0, pred]
        
        return pred, prob
