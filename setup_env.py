#!/usr/bin/env python3
"""
Cross-platform setup script for YouTube Shorts Automation.
Creates venv, installs dependencies, prepares folders, and prints OS-specific notes.
"""

import os
import sys
import platform
import subprocess
from pathlib import Path


PROJECT_DIR = Path(__file__).parent
VENV_DIR = PROJECT_DIR / "venv"
REQUIREMENTS = PROJECT_DIR / "requirements.txt"
CREDENTIALS_DIR = PROJECT_DIR / "credentials"


def run(cmd, check=True):
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def get_venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def ensure_venv():
    if not VENV_DIR.exists():
        print("Creating virtual environment...")
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
    else:
        print("Virtual environment already exists.")


def install_requirements():
    venv_python = get_venv_python()
    if not venv_python.exists():
        raise RuntimeError("venv python not found. Did venv creation fail?")
    print("Installing Python dependencies...")
    run([str(venv_python), "-m", "pip", "install", "-r", str(REQUIREMENTS)])


def ensure_folders():
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    print("Ensured credentials/ folder exists.")


def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("FFmpeg: OK")
        return True
    except Exception:
        print("FFmpeg: MISSING")
        return False


def print_next_steps():
    os_name = platform.system()
    print("\nNext steps:")
    if os_name == "Windows":
        print("1) Activate venv (PowerShell): .\\venv\\Scripts\\Activate.ps1")
        print("2) If you hit encoding errors, run: $env:PYTHONUTF8=1")
        print("3) Launch UI: streamlit run app.py")
    else:
        print("1) Activate venv: source venv/bin/activate")
        print("2) Launch UI: streamlit run app.py")
    print("3) Place client_secret.json in credentials/ or upload via UI")
    print("4) Authenticate: python main.py --auth")


def main():
    print("YouTube Shorts Automation Setup")
    print("=" * 40)
    ensure_venv()
    install_requirements()
    ensure_folders()
    ffmpeg_ok = check_ffmpeg()
    if not ffmpeg_ok:
        os_name = platform.system()
        if os_name == "Darwin":
            print("Install FFmpeg: brew install ffmpeg")
        elif os_name == "Windows":
            print("Install FFmpeg: https://ffmpeg.org/download.html")
        else:
            print("Install FFmpeg: sudo apt-get install ffmpeg")
    print_next_steps()


if __name__ == "__main__":
    main()
