import pyautogui
import time
import random
import ctypes
import os

# 禁用 pyautogui 的安全暂停功能
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# 源文件路径（使用当前目录下的文件）
src = os.path.join(os.path.dirname(__file__), 'auto_gen_code.py')

# 读取源文件
with open(src, 'r', encoding='utf-8') as f:
    content = f.read()

# 按行分割（不保留换行符）
lines = content.splitlines(keepends=False)

def is_esc_pressed():
    """使用 Windows API 检查 ESC 键状态"""
    try:
        return (ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000) != 0
    except:
        return False

def press_enter():
    """使用 Windows API 直接发送回车键，更可靠"""
    ctypes.windll.user32.keybd_event(0x0D, 0, 0, 0)  # key down
    time.sleep(0.02)
    ctypes.windll.user32.keybd_event(0x0D, 0, 2, 0)  # key up
    time.sleep(random.uniform(0.2, 0.35))

def type_char(ch):
    """输入单个字符"""
    shift_chars = {
        '~': '`', '!': '1', '@': '2', '#': '3', '$': '4',
        '%': '5', '^': '6', '&': '7', '*': '8', '(': '9',
        ')': '0', '_': '-', '+': '=', '{': '[', '}': ']',
        '|': '\\', ':': ';', '"': "'", '<': ',', '>': '.',
        '?': '/'
    }

    if ch == '\t':
        pyautogui.press('tab')
    elif ch == ' ':
        pyautogui.press('space')
    elif ch in shift_chars:
        pyautogui.keyDown('shift')
        pyautogui.press(shift_chars[ch])
        pyautogui.keyUp('shift')
    else:
        pyautogui.press(ch)

print(f'总行数: {len(lines)}')
print('5秒后开始逐行键入...')
print('按住 ESC 可随时停止！')
time.sleep(5)

chars_total = 0
try:
    for line_num, line in enumerate(lines):
        if is_esc_pressed():
            print(f'\n检测到 ESC！已输入 {chars_total} 字符，正在停止...')
            break

        for idx, ch in enumerate(line):
            if is_esc_pressed():
                raise StopIteration

            delay = random.uniform(0.08, 0.15)
            type_char(ch)
            time.sleep(delay)
            chars_total += 1

        if line_num < len(lines) - 1:
            time.sleep(random.uniform(0.05, 0.1))
            press_enter()

        if line_num > 0 and line_num % 20 == 0:
            print(f'进度: {line_num}/{len(lines)} 行 ({line_num*100//len(lines)}%)')

    print(f'\n完成！共输入 {chars_total} 字符')

except StopIteration:
    print(f'\n已停止，共输入 {chars_total} 字符')
except KeyboardInterrupt:
    print(f'\n已中断，共输入 {chars_total} 字符')

time.sleep(5)