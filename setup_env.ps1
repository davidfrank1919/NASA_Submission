<#
setup_env.ps1

PowerShell helper to create a local virtual environment and install project dependencies.
Run in the project root (PowerShell):

  .\setup_env.ps1
#>

Write-Host "== NASA RAG Project: Setup Environment ==" -ForegroundColor Cyan

function Ensure-Python {
    try {
        $py = & python --version 2>&1
        Write-Host "Found Python: $py"
        return $true
    } catch {
        Write-Host "Python not found on PATH. Please install Python 3.10+ and rerun." -ForegroundColor Red
        return $false
    }
}

if (-not (Ensure-Python)) { exit 1 }

$venvDir = ".\.venv"
if (-not (Test-Path $venvDir)) {
    Write-Host "Creating virtual environment in $venvDir..."
    python -m venv $venvDir
} else {
    Write-Host "Virtual environment already exists at $venvDir"
}

Write-Host "To activate the virtual environment, you have two safe options:" -ForegroundColor Yellow
Write-Host "  - In Command Prompt (cmd.exe): run $venvDir\Scripts\activate.bat" -ForegroundColor Green
Write-Host "  - Or avoid activation entirely and run the venv Python directly:" -ForegroundColor Green
Write-Host "      $venvDir\Scripts\python.exe -m pip install -r requirements.txt" -ForegroundColor Green

# Try to install dependencies into the venv directly (without requiring activation)
$pip = "$venvDir\Scripts\python.exe" -ErrorAction SilentlyContinue
if (Test-Path $pip) {
    Write-Host "Upgrading pip and installing requirements into venv..."
    & $pip -m pip install --upgrade pip setuptools wheel
    if (Test-Path "requirements.txt") {
        & $pip -m pip install -r requirements.txt
    } else {
        Write-Host "No requirements.txt found; skipping pip install." -ForegroundColor Yellow
    }
    Write-Host "Setup complete. Use one of the activation options shown above, or run the venv Python directly as needed." -ForegroundColor Cyan
} else {
    Write-Host "Could not find python executable inside venv. Activate the venv and run pip install manually." -ForegroundColor Red
}
