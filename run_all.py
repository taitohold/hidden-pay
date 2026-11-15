"""Run the whole pipeline in order: raw data -> reports. Usage: .venv/bin/python run_all.py"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for step in sorted(ROOT.glob("0[1-9]_*.py")):
    print(f"\n===== {step.name} =====", flush=True)
    subprocess.run([sys.executable, str(step)], check=True, cwd=ROOT)
