"""
更新检查工具
在插件加载时检查是否有新版本
"""

import os
import subprocess
from typing import Optional


def get_current_version() -> Optional[str]:
    """
    获取当前版本号
    
    Returns:
        版本号字符串，如果读取失败返回 None
    """
    version_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "version.txt")
    try:
        with open(version_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception:
        return None


def check_for_updates() -> bool:
    """
    检查是否有更新
    
    Returns:
        True 如果有更新，False 如果已是最新或检查失败
    """
    try:
        # 获取当前目录
        plugin_dir = os.path.dirname(os.path.dirname(__file__))
        
        # 检查是否是 Git 仓库
        git_dir = os.path.join(plugin_dir, '.git')
        if not os.path.exists(git_dir):
            return False
        
        # 执行 git fetch（禁止弹出认证弹框，失败时静默处理）
        env = os.environ.copy()
        env['GIT_TERMINAL_PROMPT'] = '0'
        subprocess.run(
            ['git', 'fetch', 'origin'],
            cwd=plugin_dir,
            capture_output=True,
            timeout=10,
            env=env
        )
        
        # 检查本地和远程版本
        local = subprocess.run(
            ['git', 'rev-parse', '@'],
            cwd=plugin_dir,
            capture_output=True,
            text=True
        ).stdout.strip()
        
        remote = subprocess.run(
            ['git', 'rev-parse', '@{u}'],
            cwd=plugin_dir,
            capture_output=True,
            text=True
        ).stdout.strip()
        
        return local != remote
        
    except Exception:
        return False


def notify_update_available():
    """通知用户有更新可用"""
    current_version = get_current_version()
    version_str = f" (当前版本: {current_version})" if current_version else ""
    
    print("\n" + "="*60)
    print(f"🎉 Comfyui_o1key 有新版本可用{version_str}")
    print("="*60)
    print("更新方法：")
    print("  Windows: 双击运行 update.bat")
    print("  Linux/Mac: 运行 ./update.sh")
    print("或手动执行: git pull origin main")
    print("="*60 + "\n")
