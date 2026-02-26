"""
模型配置中心
用于集中管理所有支持的 Gemini 模型

使用方式:
    1. 添加新模型: 在对应的模型列表中添加新的模型字典
    2. 临时关闭模型: 将模型的 enabled 字段设为 False
    3. 重新启用模型: 将模型的 enabled 字段改回 True

模型类型:
    - GEMINI_MODELS: Nano Banana Pro 图像生成模型
    - GEMINI_FLASH_MODELS: Google Gemini Flash 文本生成模型

示例:
    添加新模型:
    {
        "id": "gemini-新模型名称",
        "description": "模型说明和特点",
        "enabled": True,
        "endpoint_type": "standard",
        "endpoint": "/v1beta/models/gemini-新模型名称:generateContent",
        "thinking_config": {
            "不思考": None,
            "低": "low",
            "中": None,
            "高": "high"
        }
    }
    
    临时关闭模型:
    将对应模型的 "enabled": True 改为 "enabled": False
"""

from typing import List, Dict, Optional


# ============================================================
# 模型配置列表
# ============================================================

# ============================================================
# Nano Banana Pro 图像生成模型
# ============================================================

GEMINI_MODELS = [
    {
        "id": "nano-banana-pro",
        "description": "Nano Banana Pro,根据分辨率自动选择端点 (1K/2K/4K),高性能图像生成模型",
        "enabled": True,
        "endpoint_type": "dynamic",
        "endpoint": None  # 动态端点，由代码根据分辨率选择
    },
    {
        "id": "gemini-3-pro-image-preview",
        "description": "标准模式,固定端点,适用于常规图像生成",
        "enabled": True,
        "endpoint_type": "standard",
        "endpoint": "/v1beta/models/gemini-3-pro-image-preview:generateContent"
    }
]


# ============================================================
# Google Gemini Flash 文本生成模型
# ============================================================

GEMINI_FLASH_MODELS = [
    {
        "id": "gemini-3-flash-preview",
        "description": "Gemini 3 Flash,快速多模态文本生成,通过 thinkingConfig 控制思考等级",
        "enabled": True,
        "endpoint_type": "standard",
        "endpoint": "/v1beta/models/gemini-3-flash-preview:generateContent",
        "thinking_config": {
            "不思考": "minimal",
            "低": "low",
            "中": "medium",
            "高": "high"
        }
    },
    {
        "id": "gemini-3-pro-preview",
        "description": "Gemini 3 Pro,高性能多模态文本生成,通过 thinkingConfig 控制思考等级",
        "enabled": True,
        "endpoint_type": "standard",
        "endpoint": "/v1beta/models/gemini-3-pro-preview:generateContent",
        "thinking_config": {
            "不思考": None,    # 不支持，省略 thinkingConfig，使用模型默认值(high)
            "低": "low",
            "中": None,        # 不支持，省略 thinkingConfig，使用模型默认值(high)
            "高": "high"
        }
    },
    {
        "id": "gemini-3.1-pro-preview",
        "description": "Gemini 3.1 Pro,高性能多模态文本生成,通过 thinkingConfig 控制思考等级",
        "enabled": True,
        "endpoint_type": "standard",
        "endpoint": "/v1beta/models/gemini-3.1-pro-preview:generateContent",
        "thinking_config": {
            "不思考": None,    # 不支持，省略 thinkingConfig，使用模型默认值(high)
            "低": "low",
            "中": None,        # 不支持，省略 thinkingConfig，使用模型默认值(high)
            "高": "high"
        }
    }
]


# ============================================================
# 工具函数
# ============================================================

def get_enabled_models() -> List[str]:
    """
    获取所有启用的模型 ID 列表
    
    Returns:
        启用的模型 ID 列表
    
    Example:
        >>> get_enabled_models()
        ['gemini-3-pro-image-preview-url', 'gemini-3-pro-image-preview', ...]
    """
    return [model["id"] for model in GEMINI_MODELS if model.get("enabled", False)]


def get_all_models() -> List[str]:
    """
    获取所有模型 ID 列表（包括已禁用的）
    
    Returns:
        所有模型 ID 列表
    
    Example:
        >>> get_all_models()
        ['gemini-3-pro-image-preview-url', 'gemini-3-pro-image-preview', ...]
    """
    return [model["id"] for model in GEMINI_MODELS]


def get_model_config(model_id: str) -> Optional[Dict]:
    """
    根据模型 ID 获取完整的模型配置
    
    Args:
        model_id: 模型 ID
    
    Returns:
        模型配置字典，如果未找到则返回 None
    
    Example:
        >>> config = get_model_config("gemini-3-pro-image-preview-url")
        >>> print(config["description"])
        URL 模式,根据分辨率自动选择端点 (1K/2K/4K)
    """
    for model in GEMINI_MODELS:
        if model["id"] == model_id:
            return model
    return None


def is_model_enabled(model_id: str) -> bool:
    """
    检查指定模型是否启用
    
    Args:
        model_id: 模型 ID
    
    Returns:
        True 如果模型启用，False 如果禁用或不存在
    
    Example:
        >>> is_model_enabled("gemini-3-pro-image-preview-url")
        True
    """
    config = get_model_config(model_id)
    if config is None:
        return False
    return config.get("enabled", False)


def get_model_description(model_id: str) -> str:
    """
    获取模型的描述信息
    
    Args:
        model_id: 模型 ID
    
    Returns:
        模型描述，如果未找到则返回空字符串
    
    Example:
        >>> get_model_description("gemini-3-pro-image-preview")
        '标准模式,固定端点,适用于常规图像生成'
    """
    config = get_model_config(model_id)
    if config is None:
        return ""
    return config.get("description", "")


def get_endpoint_type(model_id: str) -> Optional[str]:
    """
    获取模型的端点类型
    
    Args:
        model_id: 模型 ID
    
    Returns:
        端点类型 ("dynamic", "standard", "flatfee")，如果未找到则返回 None
    
    Example:
        >>> get_endpoint_type("gemini-3-pro-image-preview-url")
        'dynamic'
    """
    config = get_model_config(model_id)
    if config is None:
        return None
    return config.get("endpoint_type")


def get_model_endpoint(model_id: str) -> Optional[str]:
    """
    获取模型的 API 端点
    
    Args:
        model_id: 模型 ID
    
    Returns:
        API 端点路径，如果未找到或为动态端点则返回 None
    
    Example:
        >>> get_model_endpoint("gemini-3-pro-image-preview")
        '/v1beta/models/gemini-3-pro-image-preview:generateContent'
        >>> get_model_endpoint("gemini-3-pro-image-preview-url")
        None  # 动态端点
    """
    config = get_model_config(model_id)
    if config is None:
        return None
    return config.get("endpoint")


# ============================================================
# Gemini Flash 模型工具函数
# ============================================================

def get_enabled_flash_models() -> List[str]:
    """
    获取所有启用的 Flash 模型 ID 列表
    
    Returns:
        启用的 Flash 模型 ID 列表
    
    Example:
        >>> get_enabled_flash_models()
        ['gemini-3-flash-preview']
    """
    return [model["id"] for model in GEMINI_FLASH_MODELS if model.get("enabled", False)]


def get_all_flash_models() -> List[str]:
    """
    获取所有 Flash 模型 ID 列表（包括已禁用的）
    
    Returns:
        所有 Flash 模型 ID 列表
    """
    return [model["id"] for model in GEMINI_FLASH_MODELS]


def get_flash_model_config(model_id: str) -> Optional[Dict]:
    """
    根据模型 ID 获取 Flash 模型的完整配置
    
    Args:
        model_id: 模型 ID
    
    Returns:
        模型配置字典，如果未找到则返回 None
    
    Example:
        >>> config = get_flash_model_config("gemini-3-flash-preview")
        >>> print(config["description"])
        'Gemini 3 Flash,快速多模态文本生成,支持图片和视频输入'
    """
    for model in GEMINI_FLASH_MODELS:
        if model["id"] == model_id:
            return model
    return None


def is_flash_model_enabled(model_id: str) -> bool:
    """
    检查指定 Flash 模型是否启用
    
    Args:
        model_id: 模型 ID
    
    Returns:
        True 如果模型启用，False 如果禁用或不存在
    """
    config = get_flash_model_config(model_id)
    if config is None:
        return False
    return config.get("enabled", False)


def get_flash_model_endpoint(model_id: str) -> Optional[str]:
    """
    获取 Flash 模型的 API 端点
    
    Args:
        model_id: 模型 ID
    
    Returns:
        API 端点路径，如果未找到则返回 None
    
    Example:
        >>> get_flash_model_endpoint("gemini-3-flash-preview")
        '/v1beta/models/gemini-3-flash-preview:generateContent'
    """
    config = get_flash_model_config(model_id)
    if config is None:
        return None
    return config.get("endpoint")


def get_flash_model_description(model_id: str) -> str:
    """
    获取 Flash 模型的描述信息
    
    Args:
        model_id: 模型 ID
    
    Returns:
        模型描述，如果未找到则返回空字符串
    """
    config = get_flash_model_config(model_id)
    if config is None:
        return ""
    return config.get("description", "")


def get_flash_model_thinking_level_value(model_id: str, thinking_level: str) -> Optional[str]:
    """
    获取指定模型在给定思考等级下应传入请求体的 thinkingLevel 值。
    
    仅对 endpoint_type="standard" 且配置了 thinking_config 的模型有效。
    返回 None 表示该等级不受支持，请求体中不应包含 thinkingConfig。
    
    Args:
        model_id: 模型 ID
        thinking_level: 思考等级中文名（不思考/低/中/高）
    
    Returns:
        API thinkingLevel 值（如 "low"/"medium"/"high"），或 None（不传参）
    
    Example:
        >>> get_flash_model_thinking_level_value("gemini-3-pro-preview", "低")
        'low'
        >>> get_flash_model_thinking_level_value("gemini-3-pro-preview", "中")
        None  # 不受支持，省略 thinkingConfig
    """
    config = get_flash_model_config(model_id)
    if config is None:
        return None
    thinking_config = config.get("thinking_config")
    if not thinking_config:
        return None
    return thinking_config.get(thinking_level)


# 已弃用：动态端点模式下不再需要这些函数
# def get_flash_model_thinking_levels(model_id: str) -> List[str]:
#     """
#     获取 Flash 模型支持的思考等级列表
#     
#     Args:
#         model_id: 模型 ID
#     
#     Returns:
#         思考等级列表（中文），如果未找到则返回空列表
#     
#     Example:
#         >>> get_flash_model_thinking_levels("gemini-3-flash-preview")
#         ['默认', '最低', '低', '中', '高']
#     """
#     config = get_flash_model_config(model_id)
#     if config is None:
#         return []
#     
#     thinking_levels = config.get("thinking_levels", {})
#     return list(thinking_levels.keys())


# def get_thinking_level_value(model_id: str, thinking_level: str) -> Optional[str]:
#     """
#     获取思考等级对应的 API 参数值
#     
#     Args:
#         model_id: 模型 ID
#         thinking_level: 思考等级（中文）
#     
#     Returns:
#         API 参数值（英文），如果未找到则返回 None
#     
#     Example:
#         >>> get_thinking_level_value("gemini-3-flash-preview", "默认")
#         'high'
#         >>> get_thinking_level_value("gemini-3-flash-preview", "最低")
#         'minimal'
#     """
#     config = get_flash_model_config(model_id)
#     if config is None:
#         return None
#     
#     thinking_levels = config.get("thinking_levels", {})
#     return thinking_levels.get(thinking_level)


# ============================================================
# 向后兼容性检查
# ============================================================

def validate_models_config() -> None:
    """
    验证模型配置的完整性
    
    检查:
    - 每个模型必须有 id, description, enabled, endpoint_type, endpoint 字段
    - 非动态端点模型必须配置有效的 endpoint
    - 至少有一个模型是启用的
    
    Raises:
        ValueError: 如果配置不合法
    """
    if not GEMINI_MODELS:
        raise ValueError("GEMINI_MODELS 列表不能为空")
    
    required_fields = ["id", "description", "enabled", "endpoint_type", "endpoint"]
    valid_endpoint_types = ["dynamic", "standard", "flatfee"]
    
    for i, model in enumerate(GEMINI_MODELS):
        # 检查必需字段
        for field in required_fields:
            if field not in model:
                raise ValueError(f"模型 #{i} 缺少必需字段: {field}")
        
        # 检查 endpoint_type 是否合法
        if model["endpoint_type"] not in valid_endpoint_types:
            raise ValueError(
                f"模型 {model['id']} 的 endpoint_type '{model['endpoint_type']}' 不合法。"
                f"必须是: {', '.join(valid_endpoint_types)}"
            )
        
        # 检查非动态端点模型必须有有效的 endpoint
        if model["endpoint_type"] != "dynamic" and not model.get("endpoint"):
            raise ValueError(
                f"模型 {model['id']} 的 endpoint_type 为 '{model['endpoint_type']}'，"
                f"但未配置有效的 endpoint 字段"
            )
        
        # 检查端点格式（如果配置了）
        endpoint = model.get("endpoint")
        if endpoint and not endpoint.startswith("/v1beta/models/"):
            raise ValueError(
                f"模型 {model['id']} 的 endpoint '{endpoint}' 格式不正确。"
                f"应以 '/v1beta/models/' 开头"
            )
    
    # 检查至少有一个启用的模型
    if not get_enabled_models():
        raise ValueError("至少需要启用一个模型")


def validate_flash_models_config() -> None:
    """
    验证 Flash 模型配置的完整性
    
    检查:
    - 每个模型必须有 id, description, enabled 字段
    - 每个模型必须有 endpoint 字段且格式正确
    - 至少有一个模型是启用的
    
    Raises:
        ValueError: 如果配置不合法
    """
    if not GEMINI_FLASH_MODELS:
        raise ValueError("GEMINI_FLASH_MODELS 列表不能为空")
    
    required_fields = ["id", "description", "enabled"]
    
    for i, model in enumerate(GEMINI_FLASH_MODELS):
        # 检查必需字段
        for field in required_fields:
            if field not in model:
                raise ValueError(f"Flash 模型 #{i} 缺少必需字段: {field}")
        
        # 检查端点配置
        if "endpoint" not in model:
            raise ValueError(f"Flash 模型 {model['id']} 缺少 'endpoint' 字段")
        
        endpoint = model.get("endpoint", "")
        if not endpoint or not endpoint.startswith("/v1beta/models/"):
            raise ValueError(
                f"Flash 模型 {model['id']} 的 endpoint '{endpoint}' 格式不正确。"
                f"应以 '/v1beta/models/' 开头"
            )
    
    # 检查至少有一个启用的模型
    if not get_enabled_flash_models():
        raise ValueError("至少需要启用一个 Flash 模型")


# 在模块加载时验证配置
try:
    validate_models_config()
except ValueError as e:
    print(f"⚠️ 图像模型配置验证失败: {str(e)}")
    print(f"⚠️ 请检查 models_config.py 文件")

try:
    validate_flash_models_config()
except ValueError as e:
    print(f"⚠️ Flash 模型配置验证失败: {str(e)}")
    print(f"⚠️ 请检查 models_config.py 文件")
