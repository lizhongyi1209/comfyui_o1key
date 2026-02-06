@echo off
chcp 65001 > nul
echo ====================================
echo Comfyui_o1key 插件更新工具
echo ====================================
echo.

:: 检查是否在 Git 仓库中
if not exist ".git" (
    echo [错误] 当前目录不是 Git 仓库
    echo 请确保插件是通过 git clone 安装的
    pause
    exit /b 1
)

:: 保存当前版本
if exist "version.txt" (
    set /p OLD_VERSION=<version.txt
    echo 当前版本: %OLD_VERSION%
) else (
    set OLD_VERSION=未知
    echo 当前版本: 未知
)

echo.
echo [1/4] 检查远程更新...
git fetch origin

:: 检查是否有更新
git status -uno | findstr "Your branch is behind" > nul
if %errorlevel% equ 0 (
    echo 发现新版本！
) else (
    echo 已是最新版本
    echo.
    choice /C YN /M "是否继续检查依赖更新？"
    if errorlevel 2 goto :end
)

echo.
echo [2/4] 备份配置文件...
if exist ".config" (
    copy /Y ".config" ".config.backup" > nul
    echo 已备份 .config 到 .config.backup
)

echo.
echo [3/4] 拉取最新代码...
git pull origin main
if %errorlevel% neq 0 (
    echo [错误] 代码更新失败，请检查网络连接或手动解决冲突
    pause
    exit /b 1
)

:: 恢复配置文件
if exist ".config.backup" (
    copy /Y ".config.backup" ".config" > nul
    del ".config.backup"
    echo 已恢复配置文件
)

echo.
echo [4/4] 更新依赖包...
python -m pip install -r requirements.txt --upgrade --quiet
if %errorlevel% neq 0 (
    echo [警告] 依赖包更新失败，请手动运行: pip install -r requirements.txt
)

echo.
echo ====================================
echo 更新完成！
echo ====================================

:: 显示新版本
if exist "version.txt" (
    set /p NEW_VERSION=<version.txt
    echo 新版本: %NEW_VERSION%
)

:: 显示最近更新日志
if exist "CHANGELOG.md" (
    echo.
    echo 最近更新内容:
    echo -----------------------------------
    powershell -Command "Get-Content CHANGELOG.md -TotalCount 20"
    echo -----------------------------------
)

echo.
echo 请重启 ComfyUI 以使更改生效
echo.
pause
goto :end

:end
exit /b 0
