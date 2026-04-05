"""启动脑电数据可视化窗口"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from visualization.eeg_viewer import main

if __name__ == "__main__":
    print("=" * 60)
    print("脑电数据可视化查看器")
    print("=" * 60)
    print("\n启动中...")
    print("\n功能:")
    print("  - 加载 .gdf, .edf, .fif, .set 格式脑电数据")
    print("  - 实时波形显示")
    print("  - 通道选择")
    print("  - 可调节显示窗口大小")
    print("\n按 Ctrl+C 退出\n")
    
    main()
