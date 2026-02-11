#!/bin/bash

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查是否已存在配置文件
if [ -f .config ]; then
    read -p "检测到已存在配置，是否覆盖？(y/n): " overwrite
    if [ "$overwrite" != "y" ] && [ "$overwrite" != "Y" ]; then
        echo -e "${YELLOW}[取消]${NC} 配置未修改"
        exit 0
    fi
fi

# 提示用户输入 API 密钥
echo ""
read -p "请输入 API 密钥: " api_key

# 验证输入
if [ -z "$api_key" ]; then
    echo -e "${RED}[错误]${NC} API 密钥不能为空！"
    exit 1
fi

# 创建配置文件
cat > .config << EOF
# Comfyui_o1key 配置文件
# 此文件由 setup_api_key.sh 自动生成

# ============ API 密钥配置 ============
O1KEY_API_KEY=$api_key

# ============ API 地址配置（可选）============
# 默认使用 https://vip.o1key.com
# 如需修改，取消下面这行的注释并修改地址
# O1KEY_API_BASE_URL=https://vip.o1key.com
EOF

if [ $? -ne 0 ]; then
    echo -e "${RED}[错误]${NC} 配置文件创建失败！"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ 配置完成！${NC}"
echo ""
echo -e "${YELLOW}[提示]${NC} 请重启 ComfyUI 以使配置生效"
echo ""
