#!/bin/bash

# 切换到脚本所在目录，确保所有相对路径操作在项目目录下执行
cd "$(dirname "$0")" || exit 1

# 设置颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "===================================="
echo "Comfyui_o1key 插件更新工具"
echo "===================================="
echo ""

# 检查是否在 Git 仓库中
if [ ! -d ".git" ]; then
    echo -e "${RED}[错误] 当前目录不是 Git 仓库${NC}"
    echo "请确保插件是通过 git clone 安装的"
    exit 1
fi

# 保存当前版本
if [ -f "version.txt" ]; then
    OLD_VERSION=$(cat version.txt)
    echo "当前版本: $OLD_VERSION"
else
    OLD_VERSION="未知"
    echo "当前版本: 未知"
fi

echo ""
echo "[1/4] 检查远程更新..."
git fetch origin

# 检查是否有更新
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse @{u})

if [ $LOCAL != $REMOTE ]; then
    echo -e "${GREEN}发现新版本！${NC}"
else
    echo -e "${GREEN}已是最新版本${NC}"
    read -p "是否继续检查依赖更新？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

echo ""
echo "[2/4] 备份配置文件..."
if [ -f ".config" ]; then
    cp .config .config.backup
    echo "已备份 .config 到 .config.backup"
fi

echo ""
echo "[3/4] 拉取最新代码..."
if git pull origin main --quiet; then
    echo -e "${GREEN}代码更新成功${NC}"
else
    echo -e "${RED}[错误] 代码更新失败，请检查网络连接或手动解决冲突${NC}"
    exit 1
fi

# 恢复配置文件
if [ -f ".config.backup" ]; then
    mv .config.backup .config
    echo "已恢复配置文件"
fi

echo ""
echo "[4/4] 更新依赖包..."
if python3 -m pip install -r requirements.txt --upgrade --quiet; then
    echo -e "${GREEN}依赖包更新成功${NC}"
else
    echo -e "${YELLOW}[警告] 依赖包更新失败，请手动运行: pip install -r requirements.txt${NC}"
fi

echo ""
echo "===================================="
echo -e "${GREEN}✓ 更新完成！${NC}"
echo "===================================="
echo ""
echo "请重启 ComfyUI 以使更改生效"
echo ""
