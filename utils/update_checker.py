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


def get_update_changelog() -> list:
    """从远程 CHANGELOG.md 最新版本块中提取更新内容（最多5条）"""
    try:
        plugin_dir = os.path.dirname(os.path.dirname(__file__))
        result = subprocess.run(
            ['git', 'show', 'origin/main:CHANGELOG.md'],
            cwd=plugin_dir,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        lines = result.stdout.splitlines()

        in_block = False
        items = []
        for line in lines:
            if line.startswith('## [') and not line.startswith('## [Unreleased]'):
                if in_block:
                    break
                in_block = True
                continue
            if in_block:
                stripped = line.strip()
                if stripped and not stripped.startswith('#') and not stripped.startswith('---'):
                    text = stripped.lstrip('- ').replace('**', '').strip()
                    if text and len(text) > 3:
                        items.append(text)
                if len(items) >= 5:
                    break

        return items
    except Exception:
        return []


def notify_new_version():
    """检测到新版本时，推送蓝色更新通知弹框"""
    changelog = get_update_changelog()

    try:
        import threading
        from server import PromptServer

        def _send():
            try:
                PromptServer.instance.send_sync(
                    "o1key.new_version",
                    {"changelog": changelog}
                )
            except Exception:
                pass

        threading.Timer(3.0, _send).start()
    except Exception:
        pass


def notify_update_available():
    """通知用户有更新可用（前端弹窗）"""
    try:
        import threading
        from server import PromptServer

        def _send():
            try:
                PromptServer.instance.send_sync(
                    "o1key.update_available",
                    {"message": "欢迎使用o1key工作流，祝您马年，马上有福，马上有钱，马到成功！！！"}
                )
            except Exception:
                pass

        threading.Timer(3.0, _send).start()
    except Exception:
        pass
