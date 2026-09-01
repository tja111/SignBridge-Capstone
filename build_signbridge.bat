@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
set "APP_DIR=SignBridge_App"

echo Building SignBridge with %PYTHON%...
%PYTHON% -c "import tkinter as tk; root=tk.Tk(); root.withdraw(); root.destroy()"
if errorlevel 1 (
  echo Tkinter/Tcl is not working in this Python installation. Repair or reinstall Python with Tcl/Tk support, then retry.
  pause
  exit /b 1
)
if not exist "%APP_DIR%" mkdir "%APP_DIR%"
if exist "%APP_DIR%\build" rmdir /s /q "%APP_DIR%\build"
if exist "%APP_DIR%\dist\SignBridge" rmdir /s /q "%APP_DIR%\dist\SignBridge"

%PYTHON% -m PyInstaller --noconfirm --clean --workpath "%APP_DIR%\build" --distpath "%APP_DIR%\dist" SignBridge.spec
if errorlevel 1 (
  echo.
  echo Build failed. Review the output above and logs\signbridge.log if present.
  pause
  exit /b 1
)

echo.
echo Build succeeded: %CD%\%APP_DIR%\dist\SignBridge\SignBridge.exe
echo Share the complete %APP_DIR%\dist\SignBridge folder, not the EXE alone.
pause
