@echo off
setlocal EnableDelayedExpansion
title comfyui_o1key Updater

:: Change to the directory where this .bat file lives
cd /d "%~dp0"

echo.
echo ================================================
echo   comfyui_o1key  Auto Updater
echo ================================================
echo.

:: ── 1. Check git is installed ─────────────────────────────────────────
where git >nul 2>&1
if errorlevel 1 (
    set "STATUS=FAILED"
    set "REASON=Git is not installed or not in PATH."
    goto :done
)

:: ── 2. Confirm this is a git repo ─────────────────────────────────────
if not exist ".git" (
    set "STATUS=FAILED"
    set "REASON=.git folder not found. Make sure this .bat is in the plugin root."
    goto :done
)

:: ── 3. Show current version ───────────────────────────────────────────
echo [INFO] Current version:
for /f %%i in ('git rev-parse --short HEAD 2^>nul') do set "OLD_HASH=%%i"
for /f "delims=" %%i in ('git log -1 --format^="%%ci" 2^>nul') do set "OLD_DATE=%%i"
echo        Hash : %OLD_HASH%
echo        Date : %OLD_DATE%
echo.

:: ── 4. Back up .config if it exists ───────────────────────────────────
if exist ".config" (
    copy /y ".config" ".config.bak" >nul 2>&1
    echo [INFO] .config backed up  (API key will be restored after update)
    echo.
)

:: ── 5. Fetch latest from GitHub ───────────────────────────────────────
echo [INFO] Fetching from GitHub...
git fetch origin 2>&1
if errorlevel 1 (
    set "STATUS=FAILED"
    set "REASON=git fetch failed. Check your network / GitHub access."
    goto :restore_config
)
echo.

:: ── 6. Check if already up to date ───────────────────────────────────
for /f %%i in ('git rev-parse HEAD 2^>nul')         do set "LOCAL=%%i"
for /f %%i in ('git rev-parse origin/main 2^>nul')  do set "REMOTE=%%i"
if "%LOCAL%"=="%REMOTE%" (
    set "STATUS=UP-TO-DATE"
    set "REASON=Already on the latest version. No update needed."
    goto :restore_config
)

:: ── 7. Show incoming changes ──────────────────────────────────────────
echo [INFO] Incoming changes:
git log HEAD..origin/main --oneline 2>&1
echo.

:: ── 8. Switch to / create local main branch ───────────────────────────
git branch --list main | findstr "main" >nul 2>&1
if errorlevel 1 (
    git checkout -b main origin/main >nul 2>&1
) else (
    git checkout main >nul 2>&1
)

:: ── 9. Force reset to remote main (handles cross-version gaps) ────────
echo [INFO] Applying update...
git reset --hard origin/main 2>&1
if errorlevel 1 (
    set "STATUS=FAILED"
    set "REASON=git reset --hard failed."
    goto :restore_config
)

:: ── 10. Clean untracked files (keep .config) ──────────────────────────
git clean -fd -e ".config" -e ".config.bak" 2>&1
echo.

set "STATUS=SUCCESS"

:restore_config
:: ── Restore .config ───────────────────────────────────────────────────
if exist ".config.bak" (
    copy /y ".config.bak" ".config" >nul 2>&1
    del /f /q ".config.bak" >nul 2>&1
    if "%STATUS%"=="SUCCESS" (
        echo [INFO] .config restored  (API key unchanged)
        echo.
    )
)

:done
:: ── Final status banner ───────────────────────────────────────────────
echo.
if "%STATUS%"=="SUCCESS" (
    echo ================================================
    echo   RESULT :  UPDATE SUCCESSFUL
    for /f %%i in ('git rev-parse --short HEAD 2^>nul') do set "NEW_HASH=%%i"
    for /f "delims=" %%i in ('git log -1 --format^="%%ci" 2^>nul') do set "NEW_DATE=%%i"
    echo   Old     :  %OLD_HASH%  ^(%OLD_DATE%^)
    echo   New     :  !NEW_HASH!  ^(!NEW_DATE!^)
    echo   Action  :  Please restart ComfyUI to load the new plugin.
    echo ================================================
) else if "%STATUS%"=="UP-TO-DATE" (
    echo ================================================
    echo   RESULT :  ALREADY UP TO DATE
    echo   Hash   :  %LOCAL%
    echo   Info   :  %REASON%
    echo ================================================
) else (
    echo ================================================
    echo   RESULT :  UPDATE FAILED
    echo   Reason :  %REASON%
    echo ================================================
)
echo.
pause
endlocal
