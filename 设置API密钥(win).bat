@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: 检查是否已存在配置文件
if exist .config (
    set /p "overwrite=检测到已存在配置，是否覆盖？(y/n): "
    if /i not "!overwrite!"=="y" (
        echo [取消] 配置未修改
        pause
        exit /b 0
    )
)

:: 提示用户输入 API 密钥
echo.
set /p "api_key=请输入 API 密钥: "

:: 验证输入
if "!api_key!"=="" (
    echo [错误] API 密钥不能为空！
    pause
    exit /b 1
)

:: 创建配置文件
(
echo # Comfyui_o1key 配置文件
echo # 此文件由 setup_api_key.bat 自动生成
echo.
echo # ============ API 密钥配置 ============
echo O1KEY_API_KEY=!api_key!
echo.
echo # ============ API 地址配置 ^(可选^) ============
echo # 默认使用 https://vip.o1key.com
echo # 如需修改，取消下面这行的注释并修改地址
echo # O1KEY_API_BASE_URL=https://vip.o1key.com
) > .config

if errorlevel 1 (
    echo [错误] 配置文件创建失败！
    pause
    exit /b 1
)

echo.
echo ✓ 配置完成！
echo.
echo [提示] 请重启 ComfyUI 以使配置生效
echo.
pause
