"""MetaBCI 包装器模块 - 调用 MetaBCI 库进行 EEG 分类"""

import numpy as np
from typing import Tuple, Optional
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import CLASSIFIER_CONFIG


class MetaBCIClassifier:
    """MetaBCI 分类器包装器"""
    
    def __init__(self, method: str = None):
        """
        初始化 MetaBCI 分类器
        
        Args:
            method: MetaBCI 方法名
        """
        self.method = method or CLASSIFIER_CONFIG['metabci']['method']
        self.model = None
        self.is_trained = False
        self._import_metabci()
    
    def _import_metabci(self):
        """尝试导入 MetaBCI"""
        try:
            global metabci
            import metabci
            print(f"✓ MetaBCI 已加载 (版本: {getattr(metabci, '__version__', 'unknown')})")
        except ImportError:
            print("⚠ MetaBCI 未安装，将使用自定义模式")
            print("  安装命令: pip install metabci")
            self.metabci_available = False
        else:
            self.metabci_available = True
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        使用 MetaBCI 训练分类器
        
        Args:
            X: 数据 (n_samples, n_channels, n_times)
            y: 标签 (0=左手, 1=右手)
        """
        if not self.metabci_available:
            raise ImportError("MetaBCI 未安装，无法使用此模式")
        
        from metabci.brainda.algorithms.decomposition import CSP
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        
        # CSP + LDA 是经典组合
        if self.method == 'csp_lda':
            print("使用 CSP + LDA 分类器...")
            
            # CSP
            self.csp = CSP(n_components=4)
            X_csp = self.csp.fit_transform(X, y)
            
            # LDA
            self.lda = LDA()
            self.lda.fit(X_csp, y)
            
            self.model = (self.csp, self.lda)
            self.is_trained = True
            
            # 简单评估
            X_pred = self.csp.transform(X)
            acc = np.mean(self.lda.predict(X_pred) == y)
            print(f"训练准确率: {acc:.3f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 数据 (n_samples, n_channels, n_times)
            
        Returns:
            predictions: 预测标签
        """
        if not self.is_trained:
            raise ValueError("分类器尚未训练！")
        
        if self.method == 'csp_lda':
            csp, lda = self.model
            X_csp = csp.transform(X)
            return lda.predict(X_csp)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if not self.is_trained:
            raise ValueError("分类器尚未训练！")
        
        if self.method == 'csp_lda':
            csp, lda = self.model
            X_csp = csp.transform(X)
            return lda.predict_proba(X_csp)
