import os
import platform
import subprocess
import sys
import time
import venv
from pathlib import Path

VENV_DIR = Path("venv")
PYTHON_BIN = VENV_DIR / "Scripts" / "python.exe" if platform.system() == "Windows" else VENV_DIR / "bin" / "python"
REQUIREMENTS_FILE = Path("requirements.txt")


def create_virtualenv():
    if not VENV_DIR.exists():
        print("📦 Creating virtual environment...")
        venv.create(VENV_DIR, with_pip=True)
    else:
        print("✅ Virtual environment already exists.")


def install_dependencies():
    print("⬇️ Installing dependencies from requirements.txt...")
    subprocess.run([PYTHON_BIN, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([PYTHON_BIN, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)], check=True)
    print("✅ All dependencies installed.")

def start_uvicorn():
    print("🚀 Starting uvicorn server on port 8080...")
    # Use the python executable from the venv to start uvicorn
    # Note: This blocks the script, so it should be last step
    subprocess.run([
        str(PYTHON_BIN),
        "-m",
        "uvicorn",
        "api:app",
        "--port",
        "8080"
    ])

def main():
    if not REQUIREMENTS_FILE.exists():
        print("❌ Missing requirements.txt file.")
        sys.exit(1)

    create_virtualenv()
    install_dependencies()
    start_uvicorn()

if __name__ == "__main__":
    main()
