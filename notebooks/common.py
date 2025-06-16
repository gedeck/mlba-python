from pathlib import Path
import sys

PROJECT_DIR = Path.cwd().resolve().parent
SRC_DIR = PROJECT_DIR / 'src'
if SRC_DIR not in sys.path:
    sys.path.append(str(SRC_DIR))
