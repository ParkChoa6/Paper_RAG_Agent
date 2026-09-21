import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper_rag.config import PROJECT_ROOT, RAW_DIR

print(PROJECT_ROOT)
print(RAW_DIR)
