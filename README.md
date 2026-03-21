# 脑电信号 → 摩斯密码识别项目

## 项目简介

通过识别左右手脑电信号，将其转换为摩斯密码，最终输出文字。

## 项目结构

```
brainwave-morse/
├── README.md                          # 项目说明
├── config/                              # 配置文件
│   └── settings.py                    # 全局配置
├── data/                              # 数据相关
│   ├── loader.py                      # 数据加载
│   └── preprocessing.py             # 数据预处理
├── models/                            # 模型相关
│   ├── classifier.py               # 分类器（双模式）
│   └── metabci_wrapper.py          # MetaBCI 包装器
├── morse/                             # 摩斯密码处理
│   ├── decoder.py                   # 摩斯解码
│   └── encoder.py                   # 摩斯编码
├── pipeline/                          # 处理流程
│   ├── pipeline.py               # 主流程
│   └── realtime.py              # 实时处理
└── main.py                           # 主入口
```

## 两种处理模式

1. **自定义模型模式 - 使用自定义 EEG 特征 + 分类器
2. **MetaBCI 模式 - 调用 MetaBCI 库

## 安装依赖

```bash
pip install numpy scipy mne matplotlib scikit-learn
# 可选：安装 metabci
pip install metabci
```

## 使用方法

### 基本使用

```bash
python main.py --data ./data/raw --mode custom
# 或
python main.py --data ./data/raw --mode metabci
```

### 实时模式

```bash
python main.py --realtime --mode custom
```
##声明
本项目由seed2.0辅助开发
