#!/usr/bin/env python3
"""Simple entry point that forwards to src.scripts or src.train"""

import sys
import subprocess

def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py prepare|train|analyze|inference|visualize [...]")
        sys.exit(1)

    cmd = sys.argv[1]
    args = sys.argv[2:]

    if cmd == "prepare":
        subprocess.run([sys.executable, "-m", "src.scripts.prepare_data"] + args)
    elif cmd == "train":
        subprocess.run([sys.executable, "-m", "src.train"] + args)
    elif cmd == "analyze":
        subprocess.run([sys.executable, "-m", "src.scripts.analyze_results"] + args)
    elif cmd == "inference":
        subprocess.run([sys.executable, "-m", "src.scripts.inference"] + args)
    elif cmd == "visualize":
        subprocess.run([sys.executable, "-m", "src.scripts.visualize_contours"] + args)
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

if __name__ == "__main__":
    main()
