# 快速开始指南

## 安装依赖

```bash
cd D:\brainwave-morse
pip install -r requirements.txt
```

可选：安装 MetaBCI
```bash
pip install metabci
```

## 快速测试

### 1. 测试摩斯编解码（无需脑电数据）

```bash
python main.py --test "HELLO WORLD"
```

这会模拟：
- 文字 → 摩斯密码
- 摩斯密码 → 左右手信号
- 信号 → 解码回文字

### 2. 实时模拟模式

```bash
python main.py --realtime --mode custom
```

或使用 MetaBCI 模式：
```bash
python main.py --realtime --mode metabci
```

### 3. 用真实数据训练

```bash
# 自定义模式
python main.py --data ./your_data_folder --mode custom

# MetaBCI 模式
python main.py --data ./your_data_folder --mode metabci
```

## 项目结构说明

```
brainwave-morse/
├── config/              # 配置文件
├── data/                # 数据加载 + 预处理
├── models/              # 分类器（双模式）
│   ├── classifier.py    # 自定义分类器
│   └── metabci_wrapper.py  # MetaBCI 包装
├── morse/               # 摩斯编解码
├── pipeline/            # 主流程 + 实时处理
├── main.py              # 主入口
└── requirements.txt     # 依赖
```

## 两种处理模式

### 模式 1: 自定义模式 (--mode custom)

- 使用自定义特征（功率谱 + CSP + 时域特征）
- 使用 SVM/LDA/RandomForest 分类器
- 轻量级，无需额外库

### 模式 2: MetaBCI 模式 (--mode metabci)

- 调用 MetaBCI 库
- 使用 CSP + LDA 经典组合
- 需要安装: `pip install metabci`

## 数据格式说明

支持的格式:
- `.edf` (欧洲数据格式)
- `.fif` (MNE 格式)
- `.set` (EEGLAB 格式)

事件标记:
- 1 = 左手（点）
- 2 = 右手（划）

## 下一步

1. 用 `--test` 先测试系统是否正常
2. 准备你的脑电数据
3. 用 `--data` 训练模型
4. 用 `--realtime` 进行实时识别
