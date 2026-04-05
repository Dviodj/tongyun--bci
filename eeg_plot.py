"""EEG Data Visualization - Simple English Version"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


class EEGPlotter:
    """Simple EEG Plotter"""
    
    def __init__(self):
        self.data = None
        self.sfreq = 250
        self.ch_names = ['C3', 'Cz', 'C4']
        self.offset = 0
        self.window = 10
        
    def generate_dummy_data(self):
        """Generate dummy EEG data"""
        print("Generating dummy EEG data...")
        n_samples = int(self.sfreq * 60)  # 60 seconds
        t = np.linspace(0, 60, n_samples)
        
        # Alpha (8-13 Hz) + Beta (13-30 Hz) + Noise
        alpha = 0.5 * np.sin(2 * np.pi * 10 * t)
        beta = 0.3 * np.sin(2 * np.pi * 20 * t)
        noise = 0.2 * np.random.randn(n_samples)
        
        self.data = np.zeros((3, n_samples))
        self.data[0] = alpha + beta + noise
        self.data[1] = 0.8 * alpha + 0.5 * beta + noise
        self.data[2] = 0.6 * alpha + 0.7 * beta + noise
        
        print("[OK] Data ready")
    
    def show(self):
        """Show the plot"""
        if self.data is None:
            self.generate_dummy_data()
        
        # Create figure
        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        plt.subplots_adjust(bottom=0.2)
        
        # Initial plot
        self._update()
        
        # Time slider
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        max_offset = max(0, self.data.shape[1] - int(self.window * self.sfreq))
        self.slider = Slider(ax_slider, 'Time Offset', 0, max_offset,
                              valinit=0, valstep=int(self.sfreq))
        self.slider.on_changed(self._on_slide)
        
        # Buttons
        ax_refresh = plt.axes([0.85, 0.1, 0.1, 0.04])
        btn_refresh = Button(ax_refresh, 'Refresh')
        btn_refresh.on_clicked(lambda e: self._update())
        
        ax_win_up = plt.axes([0.7, 0.05, 0.1, 0.04])
        btn_win_up = Button(ax_win_up, 'Window +')
        btn_win_up.on_clicked(self._win_up)
        
        ax_win_down = plt.axes([0.55, 0.05, 0.1, 0.04])
        btn_win_down = Button(ax_win_down, 'Window -')
        btn_win_down.on_clicked(self._win_down)
        
        plt.show()
    
    def _update(self, val=None):
        """Update plot"""
        start = int(self.slider.val) if hasattr(self, 'slider') else 0
        stop = min(start + int(self.window * self.sfreq), self.data.shape[1])
        
        chunk = self.data[:, start:stop]
        t = np.linspace(start/self.sfreq, stop/self.sfreq, chunk.shape[1])
        
        for i, ax in enumerate(self.axes):
            ax.clear()
            ax.plot(t, chunk[i], linewidth=0.8)
            ax.set_ylabel(self.ch_names[i])
            ax.grid(True, alpha=0.3)
            ax.set_ylim(np.percentile(chunk[i], 1) - 0.5,
                       np.percentile(chunk[i], 99) + 0.5)
        
        self.axes[-1].set_xlabel('Time (seconds)')
        self.fig.suptitle(f'EEG Signals (Window: {self.window}s)', fontsize=14)
        self.fig.canvas.draw_idle()
    
    def _on_slide(self, val):
        self._update()
    
    def _win_up(self, event):
        self.window = min(30, self.window + 2)
        self._update()
    
    def _win_down(self, event):
        self.window = max(2, self.window - 2)
        self._update()


def main():
    print("=" * 60)
    print("EEG Data Visualizer")
    print("=" * 60)
    print("\nControls:")
    print("  - Drag slider: Navigate timeline")
    print("  - Window + / Window -: Adjust window size")
    print("  - Refresh: Redraw plot")
    print("\nClose window to exit\n")
    
    plotter = EEGPlotter()
    plotter.show()


if __name__ == "__main__":
    main()
