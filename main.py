"""主入口文件"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import BASE_DIR
from pipeline.pipeline import BrainwaveMorsePipeline
from pipeline.realtime import RealTimeProcessor, MockEEGStream
from morse.encoder import MorseEncoder


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='脑电信号 → 摩斯密码识别项目'
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        default=None,
        help='训练数据目录或文件路径'
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['custom', 'metabci'],
        default='custom',
        help='处理模式: custom (自定义) 或 metabci (MetaBCI库)'
    )
    
    parser.add_argument(
        '--realtime', '-r',
        action='store_true',
        help='启用实时模式（模拟）'
    )
    
    parser.add_argument(
        '--test', '-t',
        type=str,
        default=None,
        help='测试编码: 输入一段文字，模拟信号流'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 60)
    print("脑电信号 → 摩斯密码识别系统")
    print("=" * 60)
    print(f"模式: {args.mode}")
    print()
    
    # 初始化处理流程
    pipeline = BrainwaveMorsePipeline(mode=args.mode)
    
    # 如果有数据，先训练
    if args.data:
        print("开始训练...")
        try:
            pipeline.train(args.data)
        except Exception as e:
            print(f"训练失败: {e}")
            print("继续使用模拟模式...")
    
    # 测试模式：文字 → 摩斯 → 信号 → 解码
    if args.test:
        print(f"\n测试编码: {args.test}")
        print("-" * 60)
        
        encoder = MorseEncoder()
        pipeline.decoder.reset()
        
        # 模拟信号流
        def callback(signal_type, timestamp):
            if signal_type != -1:
                hand = "左手(点)" if signal_type == 0 else "右手(划)"
                output = pipeline.decoder.process_signal(signal_type, timestamp)
                if output:
                    print(f"解码: {output}")
        
        print("模拟信号流...")
        encoder.simulate_stream(args.test, callback)
        
        # 刷新剩余内容
        final = pipeline.decoder.flush()
        if final:
            print(f"最终: {final}")
        
        print("-" * 60)
        print(f"完整解码文本: {pipeline.get_full_text()}")
    
    # 实时模式
    elif args.realtime:
        print("\n启动实时模式（模拟）...")
        print("按 Ctrl+C 停止")
        print("-" * 60)
        
        processor = RealTimeProcessor(pipeline)
        
        def on_text(output):
            print(f"解码: {output}")
        
        processor.set_on_text_callback(on_text)
        
        # 模拟流
        mock_stream = MockEEGStream()
        
        try:
            def mock_callback(data, timestamp):
                processor.feed_data(data, timestamp)
            
            mock_stream.start_stream(mock_callback, duration=30)
            
        except KeyboardInterrupt:
            print("\n停止实时处理")
        
        print("-" * 60)
        print(f"完整解码文本: {processor.get_current_text()}")
    
    # 如果只有数据，只训练
    elif args.data:
        print("\n训练完成！")
        print("使用 --realtime 启动实时模式，或 --test '文本' 进行测试")
    
    # 帮助信息
    else:
        print("\n使用方法:")
        print("  python main.py --data ./data/raw --mode custom    # 训练模型")
        print("  python main.py --test 'HELLO WORLD'            # 测试编码和解码")
        print("  python main.py --realtime --mode metabci       # 实时模式（MetaBCI）")
        print("\n查看帮助:")
        print("  python main.py --help")


if __name__ == "__main__":
    main()
