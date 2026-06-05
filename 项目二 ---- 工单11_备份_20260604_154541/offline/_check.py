import sys
sys.path.insert(0, ".")
from config import get_kb_paths, PARSED_DIR
print("config OK")
print(f"PARSED_DIR: {PARSED_DIR}")

kb_paths = get_kb_paths("招股说明书2")
print(f"PDF: {kb_paths['pdf_path']}")
