@echo off
setlocal EnableDelayedExpansion
title comfyui_o1key Updater
cd /d "%~dp0"

echo.
echo  [ comfyui_o1key Updater ]
echo.

:: Check git
where git >nul 2>&1
if errorlevel 1 ( set "ERR=Git not found in PATH." & goto :fail )

:: Check repo
if not exist ".git" ( set "ERR=Not a git repo. Place this file in the plugin root." & goto :fail )

:: Save old hash
for /f %%i in ('git rev-parse --short HEAD 2^>nul') do set "OLD=%%i"

:: Backup .config
if exist ".config" copy /y ".config" ".config.bak" >nul 2>&1

:: Fetch
echo  Fetching...
git fetch origin >nul 2>&1
if errorlevel 1 ( set "ERR=Network error. Check GitHub access." & goto :fail )

:: Already up to date?
for /f %%i in ('git rev-parse HEAD 2^>nul')        do set "LOCAL=%%i"
for /f %%i in ('git rev-parse origin/main 2^>nul') do set "REMOTE=%%i"
if "%LOCAL%"=="%REMOTE%" ( goto :uptodate )

:: Switch branch & force reset
git branch --list main | findstr "main" >nul 2>&1
if errorlevel 1 ( git checkout -b main origin/main >nul 2>&1 ) else ( git checkout main >nul 2>&1 )

echo  Updating...
git reset --hard origin/main >nul 2>&1
if errorlevel 1 ( set "ERR=git reset failed." & goto :fail )
git clean -fd -e ".config" -e ".config.bak" >nul 2>&1

:: Restore .config
if exist ".config.bak" ( copy /y ".config.bak" ".config" >nul 2>&1 & del /f /q ".config.bak" >nul 2>&1 )

for /f %%i in ('git rev-parse --short HEAD 2^>nul') do set "NEW=%%i"
echo.
echo  +---------------------------+
echo  ^|  SUCCESS                  ^|
echo  ^|  %OLD%  ->  %NEW%         ^|
echo  ^|  Restart ComfyUI          ^|
echo  +---------------------------+
echo.
pause & exit /b 0

:uptodate
if exist ".config.bak" ( copy /y ".config.bak" ".config" >nul 2>&1 & del /f /q ".config.bak" >nul 2>&1 )
echo.
echo  +---------------------------+
echo  ^|  Already up to date       ^|
echo  ^|  %LOCAL:~0,7%  (no change)       ^|
echo  +---------------------------+
echo.
pause & exit /b 0

:fail
if exist ".config.bak" ( copy /y ".config.bak" ".config" >nul 2>&1 & del /f /q ".config.bak" >nul 2>&1 )
echo.
echo  +---------------------------+
echo  ^|  FAILED                   ^|
echo  ^|  %ERR%
echo  +---------------------------+
echo.
pause & exit /b 1
