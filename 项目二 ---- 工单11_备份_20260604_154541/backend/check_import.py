import sys
sys.path.insert(0, ".")
try:
    import app.main
    print("import OK")
except Exception as e:
    print(f"import FAILED: {e}")
