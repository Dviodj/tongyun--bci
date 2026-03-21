"""简单版脑电数据可视化（仅 matplotlib，无需 PyQt5）"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from pathlib import Path


class SimpleEEGViewer:
    """简单版脑电查看器"""
    
    def __init__(self):
        self.raw_data = None
        self.sfreq = 250
        self.ch_names = ['C3', 'Cz', 'C4']
        self.current_offset = 0
        self.window_size = 10  # 秒
        
        self.fig = None
        self.axes = None
        self.slider = None
        
    def load_dummy_data(self):
        """加载模拟数据"""
        print("加载模拟数据...")
        n_samples = int(self.sfreq * 60)  # 60秒
        t = np.linspace(0, 60, n_samples)
        
        # 生成模拟信号
        data = np.zeros((3, n_samples))
        
        # Alpha + Beta + Noise
        alpha = 0.5 * np.sin(2 * np.pi * 10 * t)
        beta = 0.3 * np.sin(2 * np.pi * 20 * t)
        noise = 0.2 * np.random.randn(n_samples)
        
        data[0] = alpha + beta + noise
        data[1] = 0.8 * alpha + 0.5 * beta + noise
        data[2] = 0.6 * alpha + 0.7 * beta + noise
        
        class DummyRaw:
            def __init__(self, data, sfreq, ch_names):
                self._data = data
                self.info = {'sfreq': sfreq}
                self.ch_names = ch_names
            
            def get_data(self, start=None, stop=None):
                return self._data[:, start:stop]
        
        self.raw_data = DummyRaw(data, self.sfreq, self.ch_names)
        print("[OK] 模拟数据已加载")
    
    def plot(self):
        """绘制图形"""
        if self.raw_data is None:
            self.load_dummy_data()
        
        # 创建图形
        self.fig, self.axes = plt.subplots(
            len(self.ch_names), 1, 
            figsize=(12, 8), 
            sharex=True
        )
        
        plt.subplots_adjust(bottom=0.2)
        
        # 初始绘制
        self._update_plot()
        
        # 时间滑块
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        n_total = self.raw_data._data.shape[1]
        max_offset = max(0, n_total - int(self.window_size * self.sfreq))
        
        self.slider = Slider(
            ax_slider, '时间偏移', 
            0, max_offset,
            valinit=0, valstep=int(self.sfreq)
        )
        self.slider.on_changed(self._on_slide)
        
        # 刷新按钮
        ax_refresh = plt.axes([0.85, 0.1, 0.1, 0.04])
        btn_refresh = Button(ax_refresh, '刷新')
        btn_refresh.on_clicked(lambda event: self._update_plot())
        
        # 窗口大小按钮
        ax_win_up = plt.axes([0.7, 0.05, 0.1, 0.04])
        btn_win_up = Button(ax_win_up, '窗口+')
        btn_win_up.on_clicked(self._increase_window)
        
        ax_win_down = plt.axes([0.55, 0.05, 0.1, 0.04])
        btn_win_down = Button(ax_win_down, '窗口-')
        btn_win_down.on_clicked(self._decrease_window)
        
        plt.show()
    
    def _update_plot(self, val=None):
        """更新图形"""
        if self.raw_data is None or self.axes is None:
            return
        
        start = int(self.slider.val) if self.slider else 0
        stop = min(start + int(self.window_size * self.sfreq), 
                   self.raw_data._data.shape[1])
        
        data = self.raw_data.get_data(start=start, stop=stop)
        t = np.linspace(start/self.sfreq, stop/self.sfreq, data.shape[1])
        
        for i, ax in enumerate(self.axes):
            ax.clear()
            ax.plot(t, data[i], linewidth=0.8)
            ax.set_ylabel(self.ch_names[i])
            ax.grid(True, alpha=0.3)
            ax.set_ylim(np.percentile(data[i], 1) - 0.5, 
                       np.percentile(data[i], 99) + 0.5)
        
        self.axes[-1].set_xlabel('时间 (秒)')
        self.fig.suptitle(f'脑电信号 (窗口大小: {self.window_size}秒)', fontsize=14)
        self.fig.canvas.draw_idle()
    
    def _on_slide(self, val):
        """滑块变化"""
        self._update_plot()
    
    def _increase_window(self, event):
        """增大窗口"""
        self.window_size = min(30, self.window_size + 2)
        self._update_plot()
    
    def _decrease_window(self, event):
        """减小窗口"""
        self.window_size = max(2, self.window_size - 2)
        self._update_plot()


def main():
    print("=" * 60)
    print("EEG Data Viewer (Simple)")
    print("=" * 60)
    print("\nControls:")
    print("  - Drag slider: Navigate timeline")
    print("  - Window+/Window-: Adjust display window")
    print("  - Refresh: Redraw plot")
    print("\nClose window to exit\n")
    
    viewer = SimpleEEGViewer()
    viewer.plot()


if __name__ == "__main__":
    main()
