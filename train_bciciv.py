"""BCICIV 2b 数据集训练脚本"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from data.bciciv_loader import prepare_bciciv_data, create_epochs_from_raws
from data.preprocessing import extract_features
from models.classifier import EEGClassifier
from config.settings import DATA_CONFIG


def main():
    print("=" * 60)
    print("BCICIV 2b 数据集 - 脑电信号分类训练")
    print("=" * 60)
    print()
    
    # 数据目录
    data_dir = r"D:\db\BCICIV_2b_gdf"
    
    # 1. 加载数据
    print("1. 加载数据...")
    raws, events_list = prepare_bciciv_data(data_dir, subjects=['B01'])
    
    if not raws:
        print("未加载到有效数据！")
        return
    
    print(f"\n共加载 {len(raws)} 个文件")
    print()
    
    # 2. 创建 Epochs
    print("2. 创建 Epochs...")
    epochs, labels = create_epochs_from_raws(
        raws, 
        events_list,
        tmin=DATA_CONFIG['epoch_tmin'],
        tmax=DATA_CONFIG['epoch_tmax']
    )
    
    print(f"  总 Epochs 数: {len(epochs)}")
    print(f"  类别分布: 左手={sum(labels==0)}, 右手={sum(labels==1)}")
    print()
    
    # 3. 提取特征
    print("3. 提取特征...")
    features = extract_features(epochs)
    print(f"  特征维度: {features.shape}")
    print()
    
    # 4. 训练分类器
    print("4. 训练分类器...")
    classifier = EEGClassifier(mode='svm')
    classifier.fit(features, labels)
    print()
    
    print("=" * 60)
    print("✓ 训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
