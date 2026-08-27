@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"

REM == This file finds its own folder automatically -- you should NOT need to
REM    edit anything below. It must stay in the same folder as
REM    scratch_stabilizer_bridge.py. ==
set "PROJECT_DIR=%~dp0"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

if not exist "%PROJECT_DIR%\scratch_stabilizer_bridge.py" (
    echo Could not find scratch_stabilizer_bridge.py next to this .bat file.
    echo Make sure both files are in the same folder: %PROJECT_DIR%
    pause
    exit /b 1
)

REM == Locate uv. Prefer it already being on PATH; otherwise check the ==
REM == default install locations uv's own installer uses on Windows.   ==
set "UV_EXE="
where uv.exe >nul 2>&1
if not errorlevel 1 (
    set "UV_EXE=uv.exe"
) else if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"
) else if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\uv.exe" (
    set "UV_EXE=%LOCALAPPDATA%\Microsoft\WinGet\Links\uv.exe"
)

if "%UV_EXE%"=="" (
    echo Could not find uv.exe anywhere on this machine.
    echo Install it from https://docs.astral.sh/uv/getting-started/installation/
    echo then run this Custom Command again.
    pause
    exit /b 1
)

REM == Cache location (any drive with a few GB free space) ==
set "UV_CACHE_BASE=%PROJECT_DIR%"

if not exist "%UV_CACHE_BASE%\.uv-cache" mkdir "%UV_CACHE_BASE%\.uv-cache"
if not exist "%UV_CACHE_BASE%\.uv-tmp" mkdir "%UV_CACHE_BASE%\.uv-tmp"
set "UV_CACHE_DIR=%UV_CACHE_BASE%\.uv-cache"
set "TMP=%UV_CACHE_BASE%\.uv-tmp"
set "TEMP=%UV_CACHE_BASE%\.uv-tmp"
set "UV_LINK_MODE=copy"

%PROJECT_DIR:~0,2%
cd "%PROJECT_DIR%"

"%UV_EXE%" run scratch_stabilizer_bridge.py %*
pause
