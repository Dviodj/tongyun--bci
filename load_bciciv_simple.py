"""简单版 BCICIV 2b 数据加载器"""

import os
import pyedflib
from pyedflib import highlevel
import numpy as np
from pathlib import Path


def load_gdf_with_pyedflib(file_path: str):
    """用 pyedflib 加载 .gdf 文件"""
    print(f"加载: {os.path.basename(file_path)}")
    
    signals, signal_headers, header = highlevel.read_edf(file_path)
    
    # 基本信息
    sfreq = signal_headers[0]['sample_rate']
    n_channels = len(signals)
    n_samples = signals.shape[1]
    
    print(f"  采样率: {sfreq} Hz")
    print(f"  通道数: {n_channels}")
    print(f"  采样点数: {n_samples}")
    print(f"  时长: {n_samples/sfreq:.1f} 秒")
    
    # 打印通道名
    ch_names = [h['label'] for h in signal_headers]
    print(f"  通道: {ch_names[:3]}...")
    
    # 提取事件
    events = []
    if 'annotations' in header:
        for annot in header['annotations']:
            onset = annot['onset']
            duration = annot.get('duration', 0)
            desc = annot.get('description', '')
            
            # 尝试解析事件类型
            event_type = 0
            try:
                if isinstance(desc, str) and desc.isdigit():
                    event_type = int(desc)
                else:
                    event_type = int(desc)
            except (ValueError, TypeError):
                pass
            
            events.append({
                'onset': onset,
                'duration': duration,
                'type': event_type,
                'desc': desc
            })
    
    print(f"  找到 {len(events)} 个事件")
    
    # 显示事件类型分布
    if events:
        types = {}
        for e in events:
            t = e['type']
            types[t] = types.get(t, 0) + 1
        print(f"  事件类型: {types}")
    
    return signals, signal_headers, header, events


def main():
    data_dir = r"D:\db\BCICIV_2b_gdf"
    
    print("=" * 60)
    print("BCICIV 2b 数据集 - 数据探查")
    print("=" * 60)
    print()
    
    data_path = Path(data_dir)
    files = sorted(data_path.glob("*T.gdf"))
    
    print(f"找到 {len(files)} 个训练文件")
    print()
    
    # 先加载第一个文件看看
    if files:
        first_file = files[0]
        print(f"\n探查文件: {first_file.name}")
        print("-" * 60)
        
        signals, sig_headers, header, events = load_gdf_with_pyedflib(str(first_file))
        
        # 显示前几个事件详情
        if events:
            print("\n前 5 个事件:")
            for i, e in enumerate(events[:5]):
                print(f"  {i+1}. t={e['onset']:.2f}s, type={e['type']}, desc={e['desc']}")


if __name__ == "__main__":
    main()
