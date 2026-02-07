@echo off
REM setup_env.bat
REM Windows CMD helper to create a venv and install requirements

IF NOT DEFINED PATH (
  echo PATH is not defined.
)

python --version >nul 2>&1
IF ERRORLEVEL 1 (
  echo Python not found. Install Python 3.10+ and ensure 'python' is on PATH.
  goto :eof
)

SET VENV_DIR=.venv
IF NOT EXIST %VENV_DIR% (
  echo Creating virtual environment in %VENV_DIR%
  python -m venv %VENV_DIR%
) ELSE (
  echo Virtual environment already exists at %VENV_DIR%
)

echo To activate the venv in this CMD session run:
echo    %VENV_DIR%\Scripts\activate.bat

SET PY=%VENV_DIR%\Scripts\python.exe
IF EXIST %PY% (
  echo Upgrading pip and installing requirements...
  "%PY%" -m pip install --upgrade pip setuptools wheel
  IF EXIST requirements.txt (
    "%PY%" -m pip install -r requirements.txt
  ) ELSE (
    echo No requirements.txt found. Skipping pip install.
  )
  echo Setup complete. Activate with: %VENV_DIR%\Scripts\activate.bat
) ELSE (
  echo Could not find python.exe inside venv. Activate venv and run pip install manually.
)
