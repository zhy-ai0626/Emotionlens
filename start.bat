@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

cd /d "%~dp0"
set "ROOT=%~dp0.."
set "VENV=%ROOT%\.venv"
set "PYTHON=%VENV%\Scripts\python.exe"
set "CF_DIR=%USERPROFILE%\.cloudflared"
set "CF_EXE=%CF_DIR%\cloudflared.exe"

echo.
echo   ╔══════════════════════════════════════════════╗
echo   ║     PokéMood Scanner — Launcher              ║
echo   ╚══════════════════════════════════════════════╝
echo.

REM ─── 1. Check Python ──────────────────────────────────
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found in PATH. Please install Python 3.10+.
    pause
    exit /b 1
)
echo [OK] Python found

REM ─── 2. Create venv if needed ─────────────────────────
if not exist "%VENV%\Scripts\python.exe" (
    echo [VENV] Creating virtual environment at %VENV% ...
    python -m venv "%VENV%"
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
    echo [VENV] Created.
)

REM ─── 3. Install dependencies ──────────────────────────
echo [DEPS] Installing / updating dependencies ...
"%PYTHON%" -m pip install -r "%ROOT%\requirements.txt" fastapi uvicorn websockets --quiet
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] pip install had errors, trying to continue ...
)

REM ─── 4. Check / download cloudflared ──────────────────
if exist "%CF_EXE%" (
    echo [OK] cloudflared found at %CF_EXE%
    goto :cf_done
)

REM Try PATH
where cloudflared >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] cloudflared found in PATH
    goto :cf_done
)

echo.
echo [INFO] cloudflared not found — downloading ...
echo        This is needed for sharing via public HTTPS URL.
echo.

REM Create cloudflared directory
if not exist "%CF_DIR%" mkdir "%CF_DIR%"

REM Download cloudflared from GitHub
set "CF_URL=https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
echo [INFO] Downloading from %CF_URL% ...
powershell -Command "Invoke-WebRequest -Uri '%CF_URL%' -OutFile '%CF_EXE%' -UseBasicParsing" 2>nul

if exist "%CF_EXE%" (
    echo [OK] cloudflared downloaded to %CF_EXE%
) else (
    echo.
    echo [WARN] Could not auto-download cloudflared.
    echo        Please download manually from:
    echo        https://github.com/cloudflare/cloudflared/releases
    echo        Save as: %CF_EXE%
    echo.
    echo        The app will still work locally without it.
    echo.
    choice /C YN /M "Continue without cloudflared (no public sharing)?"
    if !ERRORLEVEL! EQU 2 exit /b 1
)

:cf_done

REM ─── 5. Launch ────────────────────────────────────────
echo.
echo   ┌─────────────────────────────────────────────────┐
echo   │  Starting server …                              │
echo   │  Local:  http://localhost:8000                  │
echo   │  Press Ctrl+C to stop                           │
echo   └─────────────────────────────────────────────────┘
echo.

cd /d "%ROOT%"
"%PYTHON%" emotion-app/main.py

pause
