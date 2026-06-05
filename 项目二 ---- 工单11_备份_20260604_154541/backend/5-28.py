#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyCharm 逐字符输入工具（真实模拟打字效果）
使用方法：运行脚本，倒计时 5 秒内激活 PyCharm 编辑器窗口，代码将自动输入。
"""

import time
import pyautogui
from typing import List


def stream_print(message: str, delay: float = 0.02, end: str = '\n', flush: bool = True):
    """控制台流式输出，支持自定义结尾字符"""
    for ch in message:
        print(ch, end='', flush=True)
        time.sleep(delay)
    print(end=end, flush=flush)


def type_code_in_pycharm(
        code: str,
        delay: float = 0.04,
        before_start: int = 5,
        debug_mode: bool = True
) -> bool:
    """在 PyCharm 编辑器窗口内逐个输入代码"""
    if not isinstance(code, str) or len(code) == 0:
        if debug_mode:
            print("❌ 错误：代码不能为空")
        return False

    pyautogui.PAUSE = 0.01
    pyautogui.FAILSAFE = True

    if before_start > 0:
        stream_print(f"⏳ 请在 {before_start} 秒内激活 PyCharm 编辑器窗口...")
        for i in range(before_start, 0, -1):
            stream_print(f"{i} ", end='', delay=0)
            time.sleep(1)
        stream_print("\n🚀 开始输入代码...\n")

    if debug_mode:
        stream_print(f"📄 代码总字符数: {len(code)}")
        stream_print(f"⏱️  字符间隔: {delay:.3f} 秒")

    try:
        pyautogui.write(code, interval=delay)
        stream_print(f"\n✅ 完成！共输入 {len(code)} 个字符")
        return True
    except pyautogui.FailSafeException:
        stream_print("\n⚠️ 安全触发（鼠标移至屏幕左上角）")
        return False
    except Exception as e:
        stream_print(f"\n❌ 错误: {str(e)}")
        return False


# ======================= 70 行正确的算法代码 =======================
CORRECT_CODE = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
经典算法集合（约 70 行）
包含：快速排序、归并排序、二分查找、斐波那契、最大子数组和
"""

import random
from typing import List


def quick_sort(arr: List[int]) -> List[int]:
    """快速排序（递归实现）"""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)


def merge_sort(arr: List[int]) -> List[int]:
    """归并排序"""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)


def merge(left: List[int], right: List[int]) -> List[int]:
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def binary_search(arr: List[int], target: int) -> int:
    """二分查找，返回索引，不存在返回 -1"""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


def fibonacci(n: int) -> int:
    """斐波那契数列（动态规划）"""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def max_subarray_sum(nums: List[int]) -> int:
    """最大子数组和（Kadane 算法）"""
    if not nums:
        return 0
    max_ending_here = max_so_far = nums[0]
    for x in nums[1:]:
        max_ending_here = max(x, max_ending_here + x)
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far


def main():
    print("=" * 60)
    print("经典算法演示")
    print("=" * 60)

    random.seed(42)
    data = [random.randint(1, 100) for _ in range(15)]
    print(f"原始数组: {data}")

    sorted_qs = quick_sort(data)
    print(f"快速排序: {sorted_qs}")

    sorted_ms = merge_sort(data)
    print(f"归并排序: {sorted_ms}")

    target = 42
    index = binary_search(sorted_qs, target)
    if index != -1:
        print(f"二分查找 {target} 在索引 {index}")
    else:
        print(f"{target} 不存在于数组中")

    print("斐波那契数列前 10 项:")
    for i in range(10):
        print(f"F({i}) = {fibonacci(i)}")

    nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    max_sum = max_subarray_sum(nums)
    print(f"数组 {nums} 的最大子数组和 = {max_sum}")

    print("\\n" + "=" * 60)
    print("算法演示结束")


if __name__ == "__main__":
    main()
'''

if __name__ == '__main__':
    line_count = len(CORRECT_CODE.splitlines())
    print(f"📊 准备的代码共有 {line_count} 行")
    print("=" * 60)
    print("PyCharm 逐字符输入工具 (真实模拟)")
    print("=" * 60)

    type_code_in_pycharm(
        code=CORRECT_CODE,
        delay=0.035,
        before_start=5,
        debug_mode=True
    )