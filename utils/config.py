"""
配置管理模块
处理环境变量和 API 密钥管理
"""

import os
from typing import Dict, Optional


# 获取插件根目录
PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(PLUGIN_ROOT, ".config")


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
    优先级：环境变量（推荐） > .config 文件（向后兼容）
    
    Args:
        key_name: 密钥名称，默认为 O1KEY_API_KEY
    
    Returns:
        API 密钥字符串，如果未找到则返回 None
    
    Raises:
        ValueError: 如果未找到 API 密钥
    
    Example:
        >>> api_key = get_api_key()
        >>> if api_key is None:
        ...     raise ValueError("API key not found")
    """
    # 1. 优先从环境变量读取（推荐方式）
    api_key = os.environ.get(key_name)
    
    if api_key:
        return api_key
    
    # 2. 从 .config 文件读取（向后兼容，已弃用）
    config = load_config()
    api_key = config.get(key_name)
    
    if api_key:
        return api_key
    
    return None


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
