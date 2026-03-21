"""脑电数据可视化窗口"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                             QWidget, QPushButton, QLabel, QFileDialog,
                             QHBoxLayout, QComboBox, QSlider)
from PyQt5.QtCore import Qt
from pathlib import Path


class EEGViewer(QMainWindow):
    """脑电数据查看器主窗口"""
    
    def __init__(self):
        super().__init__()
        
        self.raw_data = None
        self.sfreq = 250
        self.ch_names = ['C3', 'Cz', 'C4']
        self.current_offset = 0
        self.window_size = 10  # 显示窗口大小（秒）
        
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle('脑电数据查看器')
        self.setGeometry(100, 100, 1200, 800)
        
        # 主窗口组件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # 顶部控制栏
        control_layout = QHBoxLayout()
        
        # 加载文件按钮
        self.load_btn = QPushButton('加载数据')
        self.load_btn.clicked.connect(self.load_data)
        control_layout.addWidget(self.load_btn)
        
        # 通道选择
        control_layout.addWidget(QLabel('通道:'))
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(self.ch_names)
        self.channel_combo.currentIndexChanged.connect(self.update_plot)
        control_layout.addWidget(self.channel_combo)
        
        # 窗口大小
        control_layout.addWidget(QLabel('窗口大小(秒):'))
        self.window_slider = QSlider(Qt.Horizontal)
        self.window_slider.setMinimum(1)
        self.window_slider.setMaximum(30)
        self.window_slider.setValue(self.window_size)
        self.window_slider.valueChanged.connect(self.on_window_size_change)
        control_layout.addWidget(self.window_slider)
        
        self.window_label = QLabel(f'{self.window_size}s')
        control_layout.addWidget(self.window_label)
        
        # 刷新按钮
        self.refresh_btn = QPushButton('刷新')
        self.refresh_btn.clicked.connect(self.update_plot)
        control_layout.addWidget(self.refresh_btn)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # matplotlib 图形
        self.figure, self.axes = plt.subplots(len(self.ch_names), 1, 
                                               figsize=(12, 8), sharex=True)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # 状态栏
        self.status_label = QLabel('就绪')
        layout.addWidget(self.status_label)
        
        # 初始化空图
        self.init_empty_plot()
    
    def init_empty_plot(self):
        """初始化空图"""
        x = np.linspace(0, self.window_size, int(self.sfreq * self.window_size))
        
        for i, ax in enumerate(self.axes):
            ax.plot(x, np.zeros_like(x))
            ax.set_ylabel(self.ch_names[i])
            ax.grid(True, alpha=0.3)
        
        self.axes[-1].set_xlabel('时间 (秒)')
        self.figure.tight_layout()
        self.canvas.draw()
    
    def load_data(self):
        """加载数据文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, '选择脑电数据文件', '',
            'EEG Files (*.gdf *.edf *.fif *.set);;All Files (*)'
        )
        
        if file_path:
            self.status_label.setText(f'加载中: {Path(file_path).name}...')
            QApplication.processEvents()
            
            try:
                # 尝试用不同方式加载
                self._load_file(file_path)
                self.status_label.setText(f'已加载: {Path(file_path).name}')
                self.update_plot()
            except Exception as e:
                self.status_label.setText(f'加载失败: {str(e)}')
    
    def _load_file(self, file_path: str):
        """加载文件（内部方法）"""
        ext = Path(file_path).suffix.lower()
        
        if ext in ['.fif']:
            # MNE format
            import mne
            self.raw_data = mne.io.read_raw_fif(file_path, preload=True)
            self.sfreq = self.raw_data.info['sfreq']
            self.ch_names = self.raw_data.ch_names
        elif ext in ['.edf']:
            import mne
            self.raw_data = mne.io.read_raw_edf(file_path, preload=True)
            self.sfreq = self.raw_data.info['sfreq']
            self.ch_names = self.raw_data.ch_names
        elif ext in ['.gdf']:
            # 尝试用 biosig 或其他方式
            # 先模拟一些数据
            self._load_dummy_data()
        else:
            # 默认加载模拟数据
            self._load_dummy_data()
        
        # 更新通道下拉框
        self.channel_combo.clear()
        self.channel_combo.addItems(self.ch_names)
    
    def _load_dummy_data(self):
        """加载模拟数据（演示用）"""
        n_samples = int(self.sfreq * 60)  # 60秒数据
        t = np.linspace(0, 60, n_samples)
        
        # 生成模拟脑电信号
        data = np.zeros((3, n_samples))
        
        # Alpha 波 (8-13 Hz)
        alpha = 0.5 * np.sin(2 * np.pi * 10 * t)
        # Beta 波 (13-30 Hz)
        beta = 0.3 * np.sin(2 * np.pi * 20 * t)
        # 噪声
        noise = 0.2 * np.random.randn(n_samples)
        
        data[0] = alpha + beta + noise  # C3
        data[1] = 0.8 * alpha + 0.5 * beta + noise  # Cz
        data[2] = 0.6 * alpha + 0.7 * beta + noise  # C4
        
        # 创建简单的 Raw 对象替代品
        class DummyRaw:
            def __init__(self, data, sfreq, ch_names):
                self._data = data
                self.info = {'sfreq': sfreq}
                self.ch_names = ch_names
            
            def get_data(self, start=None, stop=None):
                return self._data[:, start:stop]
        
        self.raw_data = DummyRaw(data, self.sfreq, self.ch_names)
    
    def on_window_size_change(self, value):
        """窗口大小变化"""
        self.window_size = value
        self.window_label.setText(f'{value}s')
        self.update_plot()
    
    def update_plot(self):
        """更新图形"""
        if self.raw_data is None:
            return
        
        # 计算数据范围
        n_total = self.raw_data._data.shape[1] if hasattr(self.raw_data, '_data') else \
                   int(self.raw_data.times[-1] * self.sfreq)
        
        start = self.current_offset
        stop = min(start + int(self.window_size * self.sfreq), n_total)
        
        # 获取数据
        if hasattr(self.raw_data, 'get_data'):
            data = self.raw_data.get_data(start=start, stop=stop)
        else:
            data = self.raw_data._data[:, start:stop]
        
        # 时间轴
        t = np.linspace(start/self.sfreq, stop/self.sfreq, data.shape[1])
        
        # 更新每个子图
        for i, ax in enumerate(self.axes):
            ax.clear()
            if i < len(self.ch_names):
                ax.plot(t, data[i], linewidth=0.8)
                ax.set_ylabel(self.ch_names[i])
                ax.grid(True, alpha=0.3)
        
        self.axes[-1].set_xlabel('时间 (秒)')
        self.figure.tight_layout()
        self.canvas.draw()


def main():
    app = QApplication(sys.argv)
    viewer = EEGViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
