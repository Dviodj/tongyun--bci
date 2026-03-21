"""脑电数据可视化 - 统一入口"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    print("=" * 60)
    print("脑电数据可视化查看器")
    print("=" * 60)
    print()
    print("选择可视化版本:")
    print("  1) 简单版 (仅 matplotlib，推荐)")
    print("  2) PyQt5 版 (功能更全，需安装 PyQt5)")
    print()
    
    choice = input("请输入选择 (1/2，默认1): ").strip() or "1"
    
    if choice == "2":
        try:
            import PyQt5
            from visualization.eeg_viewer import main as main_pyqt
            print("\n启动 PyQt5 版本...")
            main_pyqt()
        except ImportError:
            print("\nPyQt5 未安装！")
            print("请运行: pip install PyQt5")
            print("\n或者使用简单版 (选择1)")
    else:
        from visualization.simple_viewer import main as main_simple
        print("\n启动简单版...")
        main_simple()


if __name__ == "__main__":
    main()
