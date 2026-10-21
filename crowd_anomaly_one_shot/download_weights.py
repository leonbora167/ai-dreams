"""Convenience root wrapper for downloading pipeline AI weights into 'weights/'."""

import os
import sys

# Delegate to eval/scripts/download_weights.py
script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval", "scripts", "download_weights.py")
if os.path.exists(script_path):
    with open(script_path, "r", encoding="utf-8") as f:
        code = compile(f.read(), script_path, "exec")
        exec(code, globals())
else:
    print(f"Error: Could not locate {script_path}")
    sys.exit(1)

