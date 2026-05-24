# Builds a single-file Windows EXE with PyInstaller.
# Run from the project root:  .\packaging\build_exe.ps1

param(
    [string]$Name = "SupportRole"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Error "Run this from the project root after creating the venv."
}
. .\.venv\Scripts\Activate.ps1

pip install --upgrade pyinstaller | Out-Null

# --windowed  -> uses pythonw.exe, no console window
# --noconfirm -> overwrite build artifacts
# --collect-all faster_whisper / ctranslate2 -> bundle native DLLs
pyinstaller `
    --noconfirm `
    --windowed `
    --name $Name `
    --collect-all faster_whisper `
    --collect-all ctranslate2 `
    --collect-binaries soundcard `
    --collect-data webrtcvad `
    --hidden-import PyQt6.QtCore `
    --hidden-import PyQt6.QtGui `
    --hidden-import PyQt6.QtWidgets `
    run.pyw

Write-Host ""
Write-Host "Build complete -> dist\$Name\$Name.exe" -ForegroundColor Green
Write-Host "Note: Ollama must be installed separately on the target machine."
