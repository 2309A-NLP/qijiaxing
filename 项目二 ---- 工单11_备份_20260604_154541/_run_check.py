import subprocess, sys
result = subprocess.run(
    [r"D:\an\envs\project2\python.exe", r"offline\_check.py"],
    capture_output=True, text=True, encoding="utf-8",
    cwd=r"C:\Users\qjx\Desktop\github\项目二 ---- 工单2"
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)
