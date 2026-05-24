"""
配置管理模块
处理环境变量和 API 密钥管理
"""

import os
from typing import Dict, Optional


# 获取插件根目录
PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(PLUGIN_ROOT, ".config")


# ============ API 基础配置 ============
# 所有 API 客户端的统一基础 URL
# 可通过环境变量 O1KEY_API_BASE_URL 覆盖
DEFAULT_API_BASE_URL = "https://api.o1key.com"

# 异步 API 基础 URL（用于异步提交+轮询模式）
# 可通过环境变量 O1KEY_ASYNC_API_BASE_URL 覆盖
DEFAULT_ASYNC_API_BASE_URL = "https://cf-api.o1key.com"

# ============ 网络线路配置 ============
NETWORK_ROUTES = {
    "全球加速": "https://api.o1key.cn",
    "CF加速": "https://cf-api.o1key.com",
    "美国直连": "https://api.o1key.com",
}
NETWORK_ROUTE_OPTIONS = ["全球加速", "CF加速", "美国直连"]


def load_config(config_path: Optional[str] = None) -> Dict[str, str]:
    """
    从配置文件加载所有配置项
    
    Args:
        config_path: 配置文件路径，默认为插件目录下的 .config
    
    Returns:
        配置字典 {key: value}
    
    Example:
        >>> config = load_config()
        >>> api_key = config.get('O1KEY_API_KEY')
    """
    if config_path is None:
        config_path = CONFIG_FILE
    
    config = {}
    
    if not os.path.exists(config_path):
        return config
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # 跳过空行和注释
                if not line or line.startswith('#'):
                    continue
                
                # 解析 KEY=VALUE 格式
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    
                    if key and value:
                        config[key] = value
    
    except Exception as e:
        print(f"⚠️ 读取配置文件失败: {e}")
    
    return config


def get_api_key(key_name: str = "O1KEY_API_KEY") -> Optional[str]:
    """
    获取 API 密钥
    从 .config 文件读取

    Args:
        key_name: 密钥名称，默认为 O1KEY_API_KEY

    Returns:
        API 密钥字符串，如果未找到则返回 None
    """
    config = load_config()
    return config.get(key_name)


def get_api_key_or_raise(key_name: str = "O1KEY_API_KEY") -> str:
    """
    获取 API 密钥，如果未找到则抛出异常
    
    Args:
        key_name: 密钥名称
    
    Returns:
        API 密钥字符串
    
    Raises:
        ValueError: 如果未找到 API 密钥
    """
    api_key = get_api_key(key_name)
    
    if not api_key:
        raise ValueError("未授权！")
    
    return api_key


def get_api_base_url() -> str:
    """
    获取 API 基础 URL
    从 .config 文件读取，如果未配置则使用默认值

    Returns:
        API 基础 URL 字符串
    """
    config = load_config()
    base_url = config.get("O1KEY_API_BASE_URL")

    if base_url:
        return base_url.rstrip('/')

    return DEFAULT_API_BASE_URL


def get_async_api_base_url() -> str:
    """
    获取异步 API 基础 URL
    从 .config 文件读取，如果未配置则使用默认值

    Returns:
        异步 API 基础 URL 字符串
    """
    config = load_config()
    base_url = config.get("O1KEY_ASYNC_API_BASE_URL")

    if base_url:
        return base_url.rstrip('/')

    return DEFAULT_ASYNC_API_BASE_URL


def get_base_url_by_route(route: str) -> str:
    """根据网络线路选项返回对应域名，未匹配则走 config 垫底"""
    return NETWORK_ROUTES.get(route, get_api_base_url())
