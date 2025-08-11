import sys
import os

from pathlib import Path

project_root = Path(os.getcwd()).resolve().parent.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))