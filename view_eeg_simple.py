"""EEG Data Visualization - Simple Version"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from visualization.simple_viewer import main

if __name__ == "__main__":
    print("=" * 60)
    print("EEG Data Visualization")
    print("=" * 60)
    print("\nStarting...")
    main()
