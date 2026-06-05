"""
start_server.py — 启动项目二后端服务
在Windows的命令提示符（cmd）中运行：
  python start_server.py
"""
import os
import sys
import subprocess

PROJECT_DIR = r"C:\Users\qjx\Desktop\github\项目二 ---- 工单2\backend"
PYTHON = r"D:\an\envs\project2\python.exe"

os.chdir(PROJECT_DIR)
print("Starting Project2 backend server...")
print(f"Python: {PYTHON}")
print(f"Port: 8001")
print(f"Docs: http://localhost:8001/docs")
print()

subprocess.run([PYTHON, "start_server.py"])
