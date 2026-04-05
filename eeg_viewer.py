"""
EEG Signal Viewer - BCICIV 2b with Morse Code Display
"""

import sys
import os
from pathlib import Path
import numpy as np
import mne
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QDockWidget, QToolBar, QAction, QStatusBar,
    QGroupBox, QFormLayout, QDoubleSpinBox, QColorDialog, QComboBox, QTextEdit, QSplitter, QLineEdit)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QPalette, QFont
import pyqtgraph as pg
import time


# Morse code dictionary
MORSE_CODE = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
    '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9',
    '-----': '0'
}


def load_eeg_file(file_path):
    """Load EEG file - supports GDF, EDF, FIF"""
    ext = Path(file_path).suffix.lower()
    
    if ext == '.gdf':
        raw = mne.io.read_raw_gdf(file_path, preload=True, verbose=False)
    elif ext == '.edf':
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    elif ext == '.fif':
        raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
    else:
        raise ValueError(f"Unsupported format: {ext}")
    
    # Bandpass filter 8-30Hz
    raw.filter(8, 30, fir_design='firwin', verbose=False)
    raw.set_eeg_reference('average', verbose=False)
    
    # Get events - try annotations first, then stim channel
    try:
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        left_id = event_id.get('769', event_id.get('left', None))
        right_id = event_id.get('770', event_id.get('right', None))
        space_id = event_id.get('space', None)
        boundary_id = event_id.get('boundary', None)
    except:
        events = None
        event_id = {}
        left_id = right_id = space_id = boundary_id = None
    
    # If no annotations found, try finding events from stim channel
    if events is None or len(events) == 0:
        try:
            events = mne.find_events(raw, stim_channel='STI 014', shortest_event=1, verbose=False)
            if len(events) > 0:
                # Use actual event values from data
                unique_events = set(events[:, 2])
                # Map common event IDs: 1=left/dot, 2=right/dash, 3=boundary, 4=space
                left_id = 1 if 1 in unique_events else None
                right_id = 2 if 2 in unique_events else None
                boundary_id = 3 if 3 in unique_events else None
                space_id = 4 if 4 in unique_events else None
        except:
            events = None
    
    valid_types = []
    if left_id is not None:
        valid_types.append(left_id)
    if right_id is not None:
        valid_types.append(right_id)
    if space_id is not None:
        valid_types.append(space_id)
    if boundary_id is not None:
        valid_types.append(boundary_id)
    
    if len(valid_types) == 0 or events is None or len(events) == 0:
        return None, None, None, raw.info['sfreq'], raw
    
    events = events[np.isin(events[:, 2], valid_types)]
    
    # 尝试提取 epochs，如果失败则使用原始数据
    try:
        # 使用更短的 epoch 时间以避免事件丢失
        tmin, tmax = 0, 0.4  # 0.4秒 epoch
        epochs = mne.Epochs(raw, events, 
                            event_id={'left': left_id, 'right': right_id, 'space': space_id, 'boundary': boundary_id},
                            tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False)
        X = epochs.get_data()
        epoch_event_ids = epochs.events[:, 2]
        
        # 转换标签: left/dot=0, right/dash=1, boundary=2, space=3
        y = np.zeros(len(epoch_event_ids), dtype=int)
        for i, yy in enumerate(epoch_event_ids):
            if yy == left_id:
                y[i] = 0  # dot
            elif yy == right_id:
                y[i] = 1  # dash
            elif yy == boundary_id:
                y[i] = 2  # boundary
            elif yy == space_id:
                y[i] = 3  # space
            else:
                y[i] = 2  # default to boundary
        
    except Exception as e:
        # 如果 epoch 提取失败，直接返回 sfreq 和 raw 对象
        print(f"Epoch extraction failed: {e}")
        return None, None, None, raw.info['sfreq'], raw
    
    return X, y, events, raw.info['sfreq'], raw


class EEGViewer(QWidget):
    """EEG Viewer Widget"""
    
    new_event = pyqtSignal(int)  # 0=left, 1=right
    
    def __init__(self):
        super().__init__()
        self.X = None
        self.y = None
        self.events = None
        self.sfreq = 250
        self.current_idx = 0
        self.is_playing = False
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Controls - 高对比度按钮
        controls = QHBoxLayout()
        
        btn_load = QPushButton("加载数据")
        btn_load.setStyleSheet("""
            QPushButton {
                background-color: #1E88E5; 
                color: white; 
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #42A5F5; }
        """)
        btn_load.clicked.connect(self.load_file)
        controls.addWidget(btn_load)
        
        self.btn_play = QPushButton("播放")
        self.btn_play.setStyleSheet("""
            QPushButton {
                background-color: #43A047; 
                color: white; 
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #66BB6A; }
        """)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setEnabled(False)
        controls.addWidget(self.btn_play)
        
        btn_reset = QPushButton("重置")
        btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #E53935; 
                color: white; 
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #EF5350; }
        """)
        btn_reset.clicked.connect(self.reset)
        controls.addWidget(btn_reset)
        
        controls.addStretch()
        
        self.lbl_info = QLabel("请加载 EEG 文件...")
        self.lbl_info.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        controls.addWidget(self.lbl_info)
        
        layout.addLayout(controls)
        
        # Plot
        self.plot = pg.PlotWidget(title="Processed EEG Signal (8-30Hz)")
        self.plot.setBackground('#1a1a1a')
        self.plot.setLabel('left', 'uV', color='white')
        self.plot.setLabel('bottom', 'Samples', color='white')
        self.plot.addLegend()
        self.plot.setMinimumHeight(350)
        
        self.curves = []
        colors = ['#ff4444', '#44ff44', '#4444ff']
        names = ['C3 (左手)', 'Cz (中心)', 'C4 (右手)']
        
        for c, n in zip(colors, names):
            curve = self.plot.plot(pen=pg.mkPen(c, width=1.5), name=n)
            self.curves.append(curve)
        
        layout.addWidget(self.plot)
        self.setLayout(layout)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_sample)
        self.timer.setInterval(100)
    
    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择EEG文件", r"D:\db\BCICIV_2b_gdf",
            "EEG Files (*.gdf *.edf *.fif)"
        )
        
        if path:
            try:
                self.lbl_info.setText("加载中...")
                QApplication.processEvents()
                
                X, y, events, sfreq, raw_obj = load_eeg_file(path)
                
                if X is None:
                    self.lbl_info.setText("未找到运动想象事件")
                    return
                
                self.X = X
                self.y = y
                self.events = events
                self.sfreq = sfreq
                self.current_idx = 0
                
                self.lbl_info.setText(f"已加载: {X.shape[0]} epochs | 左手={np.sum(y==0)}, 右手={np.sum(y==1)}, 边界={np.sum(y>=3)}")
                self.btn_play.setEnabled(True)
                self.update_plot(0)
                
                # 发射第一个事件信号，确保第一个epoch被处理
                if self.y is not None and len(self.y) > 0:
                    self.new_event.emit(self.y[0])
                
            except Exception as e:
                self.lbl_info.setText(f"错误: {str(e)}")
    
    def toggle_play(self):
        if self.X is None:
            return
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.setText("暂停")
            self.timer.start()
        else:
            self.btn_play.setText("播放")
            self.timer.stop()
    
    def reset(self):
        self.current_idx = 0
        self.is_playing = False
        self.btn_play.setText("播放")
        self.timer.stop()
        if self.X is not None:
            self.update_plot(0)
    
    def next_sample(self):
        if self.X is None:
            return
        self.current_idx += 1
        
        # 播放完一遍后停止，不循环
        if self.current_idx >= len(self.X):
            self.current_idx = len(self.X) - 1
            self.is_playing = False
            self.btn_play.setText("播放")
            self.timer.stop()
            return
        
        self.update_plot(self.current_idx)
        
        if self.y is not None:
            self.new_event.emit(self.y[self.current_idx])
    
    def update_plot(self, idx):
        if self.X is None:
            return
        data = self.X[idx]
        
        for i, curve in enumerate(self.curves):
            if i < min(3, data.shape[0]):
                curve.setData(data[i])


class SettingsPanel(QDockWidget):
    def __init__(self):
        super().__init__("设置")
        self.setup_ui()
    
    def setup_ui(self):
        w = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 时间设置
        g1 = QGroupBox("时间设置")
        f1 = QFormLayout()
        self.spin_window = QDoubleSpinBox()
        self.spin_window.setRange(0.5, 4.0)
        self.spin_window.setValue(1.0)
        self.spin_window.setSuffix(" 秒")
        f1.addRow("分析窗口:", self.spin_window)
        g1.setLayout(f1)
        layout.addWidget(g1)
        
        # 颜色设置
        g2 = QGroupBox("通道颜色")
        f2 = QFormLayout()
        
        self.btn_c3 = QPushButton("C3 (左手)")
        self.btn_c3.setStyleSheet("background: #ff4444; color: white; padding: 8px;")
        self.btn_c3.clicked.connect(lambda: self.pick_color(self.btn_c3))
        f2.addRow("C3:", self.btn_c3)
        
        self.btn_cz = QPushButton("Cz (中心)")
        self.btn_cz.setStyleSheet("background: #44ff44; color: black; padding: 8px;")
        self.btn_cz.clicked.connect(lambda: self.pick_color(self.btn_cz))
        f2.addRow("Cz:", self.btn_cz)
        
        self.btn_c4 = QPushButton("C4 (右手)")
        self.btn_c4.setStyleSheet("background: #4444ff; color: white; padding: 8px;")
        self.btn_c4.clicked.connect(lambda: self.pick_color(self.btn_c4))
        f2.addRow("C4:", self.btn_c4)
        
        g2.setLayout(f2)
        layout.addWidget(g2)
        
        # 大模型设置 - 移除温度框
        g3 = QGroupBox("大模型")
        f3 = QFormLayout()
        
        self.combo = QComboBox()
        self.combo.addItems(['qwen2.5:7b', 'llama3.1:8b', 'deepseek-coder-v2:16b', 'gpt-4', '自定义模型'])
        f3.addRow("模型:", self.combo)
        
        g3.setLayout(f3)
        layout.addWidget(g3)
        
        layout.addStretch()
        w.setLayout(layout)
        self.setWidget(w)
    
    def pick_color(self, btn):
        c = QColorDialog.getColor()
        if c.isValid():
            btn.setStyleSheet(f"background: {c.name()}; color: white; padding: 8px;")


class OutputPanel(QDockWidget):
    def __init__(self):
        super().__init__("莫尔斯码")
        self.morse_buffer = ""
        self.char_buffer = ""
        self.result_text = ""  # 存储识别结果
        self.last_event_time = 0
        self.setup_ui()
    
    def setup_ui(self):
        w = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 莫尔斯显示
        g1 = QGroupBox("莫尔斯码序列")
        l1 = QHBoxLayout()
        
        self.lbl_morse = QLabel("")
        self.lbl_morse.setFont(QFont("Courier", 18, QFont.Bold))
        self.lbl_morse.setStyleSheet("color: #00FF00; background-color: #000000; padding: 15px; border: 2px solid #00FF00;")
        self.lbl_morse.setMinimumHeight(60)
        self.lbl_morse.setAlignment(Qt.AlignCenter)
        l1.addWidget(self.lbl_morse)
        
        btn_clear = QPushButton("清空")
        btn_clear.setStyleSheet("background: #757575; color: white; padding: 10px;")
        btn_clear.clicked.connect(self.clear_all)
        btn_clear.setMaximumWidth(80)
        l1.addWidget(btn_clear)
        
        g1.setLayout(l1)
        g1.setMinimumHeight(100)
        layout.addWidget(g1)
        
        # 事件指示
        g2 = QGroupBox("当前事件")
        l2 = QHBoxLayout()
        
        self.lbl_event = QLabel("等待数据...")
        self.lbl_event.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        self.lbl_event.setAlignment(Qt.AlignCenter)
        self.lbl_event.setMinimumHeight(50)
        l2.addWidget(self.lbl_event)
        
        g2.setLayout(l2)
        g2.setMinimumHeight(80)
        layout.addWidget(g2)
        
        # 文字输出
        g3 = QGroupBox("识别结果 (莫尔斯转文字)")
        l3 = QVBoxLayout()
        
        self.text = QLineEdit()
        self.text.setReadOnly(True)
        self.text.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        self.text.setStyleSheet("background: #1a1a1a; color: #00FF00; padding: 10px;")
        self.text.setMinimumHeight(50)
        self.text.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        l3.addWidget(self.text)
        
        g3.setLayout(l3)
        layout.addWidget(g3)
        
        w.setLayout(layout)
        self.setWidget(w)
    
    def show_event(self, event):
        # Write to log file
        try:
            with open(r"C:\Users\DoubleJ\Desktop\viewer_log.txt", "a") as f:
                f.write(f"show_event: event={event}, buffer={self.char_buffer}\n")
        except:
            pass
        
        current_time = time.time()
        self.last_event_time = current_time
        
        # 空格事件 (event == 3): 转换当前buffer并添加空格
        if event == 3:
            if self.char_buffer:
                letter = MORSE_CODE.get(self.char_buffer, '?')
                self.result_text += letter
                self.text.setText(self.result_text)
                self.char_buffer = ""
            self.result_text += " "
            self.text.setText(self.result_text)
            self.morse_buffer += " "
            self.lbl_event.setText("␣ 空格 (Space)")
            self.lbl_event.setStyleSheet("color: #FFFFFF; background-color: #333333; padding: 15px; font-weight: bold; border: 2px solid #FFFFFF;")
            return
        
        # 边界事件 (event == 2): 转换当前buffer为字母
        if event == 2:
            if self.char_buffer:
                letter = MORSE_CODE.get(self.char_buffer, '?')
                self.result_text += letter
                self.text.setText(self.result_text)
                try:
                    with open(r"C:\Users\DoubleJ\Desktop\viewer_log.txt", "a") as f:
                        f.write(f"CONVERTED: {self.char_buffer} -> {letter}\n")
                except:
                    pass
                self.char_buffer = ""
            self.lbl_event.setText("| 字母边界 (Boundary)")
            self.lbl_event.setStyleSheet("color: #FFAA00; background-color: #332200; padding: 15px; font-weight: bold; border: 2px solid #FFAA00;")
            return
        
        if event == 0:
            # 左手 = 点
            self.morse_buffer += "·"
            self.char_buffer += "."
            self.lbl_event.setText("● 左手 (Dot ·)")
            self.lbl_event.setStyleSheet("color: #FF4444; background-color: #2a0000; padding: 15px; font-weight: bold; border: 2px solid #FF4444;")
        elif event == 1:
            # 右手 = 划
            self.morse_buffer += "－"
            self.char_buffer += "-"
            self.lbl_event.setText("━━ 右手 (Dash －)")
            self.lbl_event.setStyleSheet("color: #4444FF; background-color: #00002a; padding: 15px; font-weight: bold; border: 2px solid #4444FF;")
        
        # 显示最后25个字符
        display = self.morse_buffer[-25:] if len(self.morse_buffer) > 25 else self.morse_buffer
        self.lbl_morse.setText(display)
    
    def clear_all(self):
        self.morse_buffer = ""
        self.char_buffer = ""
        self.result_text = ""
        self.lbl_morse.setText("")
        self.lbl_event.setText("等待数据...")
        self.lbl_event.setStyleSheet("")
        self.text.clear()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Viewer - BCICIV 2b 莫尔斯码识别系统")
        self.setMinimumSize(1100, 850)
        
        # EEG Viewer
        self.eeg = EEGViewer()
        self.setCentralWidget(self.eeg)
        
        # Settings
        self.settings = SettingsPanel()
        self.addDockWidget(Qt.RightDockWidgetArea, self.settings)
        
        # Output
        self.output = OutputPanel()
        self.addDockWidget(Qt.BottomDockWidgetArea, self.output)
        self.output.setMinimumHeight(320)
        self.output.setMaximumHeight(400)
        
        # Connect
        self.eeg.new_event.connect(self.output.show_event)
        
        # Toolbar
        tb = QToolBar()
        tb.setMovable(False)
        self.addToolBar(tb)
        tb.addAction("加载", self.eeg.load_file)
        tb.addAction("播放", self.eeg.toggle_play)
        tb.addAction("重置", self.eeg.reset)
        
        self.statusBar().showMessage("就绪 - 请加载 EEG 文件")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Dark theme
    p = QPalette()
    p.setColor(QPalette.Window, QColor(30, 30, 30))
    p.setColor(QPalette.WindowText, Qt.white)
    p.setColor(QPalette.Base, QColor(20, 20, 20))
    p.setColor(QPalette.Text, Qt.white)
    p.setColor(QPalette.Button, QColor(50, 50, 50))
    p.setColor(QPalette.ButtonText, Qt.white)
    p.setColor(QPalette.Highlight, QColor(0, 120, 215))
    app.setPalette(p)
    
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

