"""Run a command and report its peak resident memory (RSS).

Usage: python tools/measure_peak_memory.py <command...>
Example: python tools/measure_peak_memory.py python evaluate.py --checkpoint x.pt --split test
"""
import subprocess
import sys
import time

import psutil


def main():
    cmd = sys.argv[1:]
    proc = subprocess.Popen(cmd)
    ps = psutil.Process(proc.pid)
    peak = 0
    while proc.poll() is None:
        try:
            peak = max(peak, ps.memory_info().rss)
        except psutil.NoSuchProcess:
            break
        time.sleep(0.05)
    print(f'peak RSS: {peak / 2**30:.2f} GiB', flush=True)
    sys.exit(proc.returncode)


if __name__ == '__main__':
    main()
