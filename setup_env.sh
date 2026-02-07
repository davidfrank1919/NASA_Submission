#CoPilot created this file.
#!/usr/bin/env bash
# setup_env.sh
# Cross-platform-ish shell script to create venv and install requirements
set -euo pipefail

VENV_DIR=".venv"

if ! command -v python >/dev/null 2>&1; then
  echo "Python not found. Please install Python 3.10+ and ensure 'python' is on PATH."
  exit 1
fi

echo "Creating virtual environment at $VENV_DIR (if missing)..."
if [ ! -d "$VENV_DIR" ]; then
  python -m venv "$VENV_DIR"
else
  echo "Virtual environment already exists."
fi

echo "To activate: source $VENV_DIR/bin/activate"

PY="$VENV_DIR/bin/python"
if [ -x "$PY" ]; then
  echo "Upgrading pip and installing requirements..."
  "$PY" -m pip install --upgrade pip setuptools wheel
  if [ -f requirements.txt ]; then
    "$PY" -m pip install -r requirements.txt
  else
    echo "No requirements.txt found. Skipping pip install."
  fi
  echo "Setup complete. Activate with: source $VENV_DIR/bin/activate"
else
  echo "Could not find python inside venv. Activate venv and run pip install manually."
fi
