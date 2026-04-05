"""全局配置"""

import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据相关配置
DATA_CONFIG = {
    'sample_rate': 250,  # 采样率 (Hz)
    'channels': ['C3', 'Cz', 'C4'],  # 常用通道
    'left_hand_event': 1,  # 左手事件标记
    'right_hand_event': 2,  # 右手事件标记
    'epoch_tmin': -0.2,  # 事件前 (s)
    'epoch_tmax': 0.8,  # 事件后 (s)
}

# 摩斯密码配置
MORSE_CONFIG = {
    'dot_duration': 0.3,  # 点的时长 (s)
    'dash_duration': 0.9,  # 划的时长 (s，约 3 倍点)
    'char_gap': 1.2,  # 字符间间隔 (s)
    'word_gap': 2.4,  # 单词间间隔 (s)
    'left_hand': '.',  # 左手 = 点
    'right_hand': '-',  # 右手 = 划
}

# 摩斯密码表
MORSE_CODE = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 
    'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 
    'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---', 
    'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-', 
    'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--', 
    'Z': '--..',
    '0': '-----', '1': '.----', '2': '..---', '3': '...--', 
    '4': '....-', '5': '.....', '6': '-....', '7': '--...', 
    '8': '---..', '9': '----.',
    ' ': ' ',
}

# 反转摩斯密码表
MORSE_CODE_REVERSE = {v: k for k, v in MORSE_CODE.items()}

# 分类器配置
CLASSIFIER_CONFIG = {
    'mode': 'custom',  # 'custom' 或 'metabci'
    'custom': {
        'features': ['power', 'csp', 'psd'],  # 使用的特征
        'classifier': 'svm',  # 'svm', 'lda', 'random_forest'
    },
    'metabci': {
        'method': 'csp_lda',  # MetaBCI 方法
    }
}

# 可视化配置
VIS_CONFIG = {
    'plot_epochs': True,
    'plot_features': True,
    'save_plots': True,
}
