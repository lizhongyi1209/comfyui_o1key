"""
Google Gemini 节点
ComfyUI 自定义节点,用于调用 Gemini Flash 模型进行多模态文本生成
"""

import base64
import os
import time
import tempfile
from typing import Dict, List, Optional, Tuple
from io import BytesIO

import torch
from PIL import Image

from ..utils.image_utils import tensor_to_pil, encode_image_to_base64
from ..utils.file_types import FileData
from ..clients.gemini_flash_client import GeminiFlashClient
from ..models_config import get_enabled_flash_models

# 文件大小限制（20MB）
MAX_FILE_SIZE = 20 * 1024 * 1024

# 图片缩放后最大尺寸（1K分辨率 = 1024像素）
MAX_IMAGE_DIMENSION = 1024

# 视频压缩目标大小（1-10MB）
TARGET_VIDEO_SIZE_MIN = 1 * 1024 * 1024
TARGET_VIDEO_SIZE_MAX = 10 * 1024 * 1024


# 支持的视频 MIME 类型映射
VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpg",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".flv": "video/x-flv",
    ".webm": "video/webm",
    ".wmv": "video/x-ms-wmv",
    ".3gp": "video/3gpp",
    ".3gpp": "video/3gpp"
}

# 尝试导入视频处理库
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ Google Gemini: OpenCV (cv2) 不可用，视频压缩功能将受限")

try:
    import subprocess
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False


class GoogleGemini:
    """
    Google Gemini 节点
    
    功能：
    - 支持多个 Gemini Flash 模型
    - 支持图片、视频和文件输入
    - 支持不同思考等级（不思考/低/中/高）- 通过 thinkingConfig.thinkingLevel 控制
    - 输出生成的文本内容（主要内容 + 思考内容）
    """
    
    # 支持的思考等级选项
    THINKING_LEVELS = ["不思考", "低", "中", "高"]
    
    def __init__(self):
        """初始化节点"""
        self.client = None
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        定义输入参数
        """
        # 从配置获取启用的模型列表
        enabled_models = get_enabled_flash_models()
        default_model = enabled_models[0] if enabled_models else "gemini-3-flash-preview"
        
        return {
            "required": {
                "模型": (enabled_models, {
                    "default": default_model
                }),
                "提示词": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "思考等级": (cls.THINKING_LEVELS, {
                    "default": "不思考"
                })
            },
            "optional": {
                "图片": ("IMAGE",),
                "视频": ("VIDEO",),
                "文件": ("FILE",)
            }
        }
    
    # 返回值类型
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("主要内容", "思考内容")
    
    # 执行函数名
    FUNCTION = "generate"
    
    # 节点分类
    CATEGORY = "text/generation"
    
    # 允许输出到 UI
    OUTPUT_NODE = True
    
    def _resize_image_if_needed(self, img: Image.Image) -> Image.Image:
        """
        如果图片过大，缩放到1K分辨率
        
        Args:
            img: PIL Image 对象
            
        Returns:
            缩放后的 PIL Image
        """
        width, height = img.size
        max_dim = max(width, height)
        
        if max_dim > MAX_IMAGE_DIMENSION:
            # 计算缩放比例
            scale = MAX_IMAGE_DIMENSION / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            print(f"Google Gemini: 图片尺寸 {width}x{height} 超过限制，缩放至 {new_width}x{new_height}")
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return img
    
    def _check_and_compress_image(self, img: Image.Image) -> str:
        """
        检查图片大小，如果超过20MB则进行压缩
        
        Args:
            img: PIL Image 对象
            
        Returns:
            base64 编码的字符串
        """
        # 先进行尺寸缩放（如果需要）
        img = self._resize_image_if_needed(img)
        
        # 尝试不同的压缩质量
        qualities = [95, 85, 75, 65, 55, 45]
        
        for quality in qualities:
            buffer = BytesIO()
            # 转换为RGB模式（去除alpha通道）以减小体积
            if img.mode in ('RGBA', 'P'):
                img_rgb = img.convert('RGB')
            else:
                img_rgb = img
            
            img_rgb.save(buffer, format='JPEG', quality=quality, optimize=True)
            buffer.seek(0)
            data = buffer.getvalue()
            
            if len(data) <= MAX_FILE_SIZE:
                print(f"Google Gemini: 图片压缩后大小 {len(data) / 1024 / 1024:.2f}MB (质量{quality})")
                return base64.b64encode(data).decode('utf-8')
        
        # 如果所有质量都无法满足，使用最低质量
        print(f"Google Gemini: 警告 - 即使最低质量仍超过20MB，将使用最低质量发送")
        return base64.b64encode(data).decode('utf-8')
    
    def _prepare_image_data(
        self, 
        images: Optional[torch.Tensor]
    ) -> Optional[List[Dict[str, str]]]:
        """
        准备图片数据
        
        如果图片超过20MB，会自动进行缩放和压缩
        
        Args:
            images: ComfyUI 图片张量 [B, H, W, C]
        
        Returns:
            图片数据列表，每个元素包含 mime_type 和 data
        """
        if images is None:
            return None
        
        image_data = []
        pil_images = tensor_to_pil(images)
        
        for img in pil_images:
            # 检查原始图片大小（通过内存占用估算）
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            original_size = buffer.tell()
            buffer.close()
            
            if original_size > MAX_FILE_SIZE:
                print(f"Google Gemini: 检测到图片过大 ({original_size / 1024 / 1024:.2f}MB)，正在进行压缩...")
                b64_str = self._check_and_compress_image(img)
                mime_type = "image/jpeg"  # 压缩后使用JPEG格式
            else:
                b64_str = encode_image_to_base64(img)
                mime_type = "image/png"
            
            image_data.append({
                "mime_type": mime_type,
                "data": b64_str
            })
        
        return image_data if image_data else None
    
    def _compress_video_with_ffmpeg(self, input_path: str, output_path: str, target_size: int) -> bool:
        """
        使用 FFmpeg 压缩视频到目标大小
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            target_size: 目标文件大小（字节）
            
        Returns:
            是否压缩成功
        """
        try:
            # 获取视频时长（秒）
            probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                        '-of', 'default=noprint_wrappers=1:nokey=1', input_path]
            duration = float(subprocess.check_output(probe_cmd).decode().strip())
            
            # 计算目标比特率（bit/s），预留一些余量
            target_bitrate = int((target_size * 8) / duration * 0.9)
            
            # 使用 FFmpeg 压缩视频
            # -c:v libx264: 使用 H.264 编码器
            # -b:v: 视频比特率
            # -maxrate 和 -bufsize: 控制码率波动
            # -c:a aac: 音频使用 AAC 编码
            # -b:a 128k: 音频比特率 128k
            # -movflags +faststart: 优化网络播放
            cmd = [
                'ffmpeg', '-y', '-i', input_path,
                '-c:v', 'libx264',
                '-b:v', f'{target_bitrate}',
                '-maxrate', f'{int(target_bitrate * 1.5)}',
                '-bufsize', f'{target_bitrate * 2}',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-movflags', '+faststart',
                '-preset', 'fast',
                output_path
            ]
            
            print(f"Google Gemini: 正在压缩视频到 {target_size / 1024 / 1024:.1f}MB...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(output_path):
                final_size = os.path.getsize(output_path)
                print(f"Google Gemini: 视频压缩完成，最终大小 {final_size / 1024 / 1024:.2f}MB")
                return True
            else:
                print(f"Google Gemini: FFmpeg 压缩失败: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Google Gemini: 视频压缩异常: {str(e)}")
            return False
    
    def _compress_video_with_opencv(self, input_path: str, output_path: str, scale: float = 0.5) -> bool:
        """
        使用 OpenCV 压缩视频（备用方案）
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            scale: 尺寸缩放比例
            
        Returns:
            是否压缩成功
        """
        if not CV2_AVAILABLE:
            return False
            
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                return False
            
            # 获取原视频参数
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 计算新尺寸
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
            
            print(f"Google Gemini: 使用 OpenCV 压缩视频，分辨率 {width}x{height} -> {new_width}x{new_height}")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 缩放帧
                resized = cv2.resize(frame, (new_width, new_height))
                out.write(resized)
            
            cap.release()
            out.release()
            
            if os.path.exists(output_path):
                final_size = os.path.getsize(output_path)
                print(f"Google Gemini: 视频压缩完成，最终大小 {final_size / 1024 / 1024:.2f}MB")
                return True
            return False
            
        except Exception as e:
            print(f"Google Gemini: OpenCV 压缩失败: {str(e)}")
            return False
    
    def _compress_video(self, video_path: str) -> str:
        """
        压缩视频到 1-10MB 之间
        
        Args:
            video_path: 原视频路径
            
        Returns:
            压缩后的视频路径（临时文件）
        """
        original_size = os.path.getsize(video_path)
        print(f"Google Gemini: 视频文件过大 ({original_size / 1024 / 1024:.2f}MB)，正在压缩...")
        
        # 创建临时文件
        temp_dir = tempfile.gettempdir()
        _, ext = os.path.splitext(video_path)
        output_path = os.path.join(temp_dir, f"compressed_{int(time.time())}{ext}")
        
        # 确定目标大小（优先尝试 10MB，如果不行再降低）
        target_sizes = [
            TARGET_VIDEO_SIZE_MAX,  # 10MB
            int(TARGET_VIDEO_SIZE_MAX * 0.8),  # 8MB
            int(TARGET_VIDEO_SIZE_MAX * 0.6),  # 6MB
            int(TARGET_VIDEO_SIZE_MAX * 0.5),  # 5MB
            TARGET_VIDEO_SIZE_MIN * 5,  # 5MB
            TARGET_VIDEO_SIZE_MIN * 3,  # 3MB
            TARGET_VIDEO_SIZE_MIN * 2,  # 2MB
        ]
        
        # 优先尝试 FFmpeg
        if FFMPEG_AVAILABLE:
            for target_size in target_sizes:
                if self._compress_video_with_ffmpeg(video_path, output_path, target_size):
                    # 检查最终大小
                    final_size = os.path.getsize(output_path)
                    if TARGET_VIDEO_SIZE_MIN <= final_size <= MAX_FILE_SIZE:
                        return output_path
                    # 如果仍然太大，继续降低目标
                    os.remove(output_path)
        
        # FFmpeg 失败或不可用，尝试 OpenCV
        if CV2_AVAILABLE:
            scales = [0.7, 0.5, 0.4, 0.3, 0.25]
            for scale in scales:
                if self._compress_video_with_opencv(video_path, output_path, scale):
                    final_size = os.path.getsize(output_path)
                    if final_size <= MAX_FILE_SIZE:
                        return output_path
                    # 如果仍然太大，继续降低分辨率
                    os.remove(output_path)
        
        # 所有压缩方法都失败
        raise ValueError(
            f"视频文件过大 ({original_size / 1024 / 1024:.2f}MB) 且无法压缩到 20MB 以下。"
            f"请安装 FFmpeg 以获得更好的压缩效果，或手动压缩视频。"
        )
    
    def _prepare_video_data(
        self, 
        video
    ) -> Optional[Dict[str, str]]:
        """
        准备视频数据
        
        ComfyUI VIDEO 类型包含视频文件路径信息。
        读取视频文件并转换为 base64。
        如果视频超过 20MB，会自动进行压缩。
        
        Args:
            video: ComfyUI VIDEO 类型数据
        
        Returns:
            视频数据字典，包含 mime_type 和 data
        """
        if video is None:
            return None
        
        # VIDEO 类型处理：支持多种格式
        video_path = None
        temp_compressed_path = None
        
        if isinstance(video, dict):
            # 字典格式：尝试常见的键名
            video_path = video.get("video") or video.get("path") or video.get("file") or video.get("filename")
            # 如果还是找不到，遍历所有键找到有效路径
            if not video_path:
                for key, val in video.items():
                    if isinstance(val, str) and os.path.exists(val):
                        video_path = val
                        break
        elif isinstance(video, str):
            # 字符串格式：直接作为路径
            video_path = video
        else:
            # 对象格式：尝试常见属性
            # 1. 尝试 __file 属性（VideoFromFile 对象）
            if hasattr(video, "__file"):
                video_path = video.__file
            # 2. 尝试其他常见属性
            elif hasattr(video, "video"):
                video_path = video.video
            elif hasattr(video, "path"):
                video_path = video.path
            elif hasattr(video, "filename"):
                video_path = video.filename
            # 3. 尝试从 __dict__ 中查找路径（支持私有属性如 _VideoFromFile__file）
            elif hasattr(video, "__dict__"):
                for attr_name, attr_value in video.__dict__.items():
                    # 查找字符串类型的属性，且包含 file 或 path 关键字
                    if isinstance(attr_value, str):
                        if "file" in attr_name.lower() or "path" in attr_name.lower():
                            # 验证路径是否有效
                            if os.path.exists(attr_value):
                                video_path = attr_value
                                break
                        # 如果属性值本身看起来像文件路径，也尝试使用
                        elif os.path.exists(attr_value) and os.path.isfile(attr_value):
                            video_path = attr_value
                            break
        
        if not video_path or not os.path.exists(video_path):
            print(f"Google Gemini: 视频文件不存在或路径无效: {video_path}")
            return None
        
        # 获取文件扩展名和 MIME 类型
        _, ext = os.path.splitext(video_path)
        ext = ext.lower()
        
        mime_type = VIDEO_MIME_TYPES.get(ext, "video/mp4")
        
        try:
            # 检查文件大小
            file_size = os.path.getsize(video_path)
            
            # 如果超过 20MB，进行压缩
            if file_size > MAX_FILE_SIZE:
                video_path = self._compress_video(video_path)
                temp_compressed_path = video_path
                # 压缩后统一使用 mp4 格式
                mime_type = "video/mp4"
            
            # 读取并编码视频
            with open(video_path, "rb") as f:
                video_bytes = f.read()
            
            b64_str = base64.b64encode(video_bytes).decode("utf-8")
            
            # 清理临时文件
            if temp_compressed_path and os.path.exists(temp_compressed_path):
                try:
                    os.remove(temp_compressed_path)
                    print(f"Google Gemini: 临时压缩文件已清理")
                except:
                    pass
            
            return {
                "mime_type": mime_type,
                "data": b64_str
            }
        
        except Exception as e:
            # 清理临时文件
            if temp_compressed_path and os.path.exists(temp_compressed_path):
                try:
                    os.remove(temp_compressed_path)
                except:
                    pass
            
            print(f"Google Gemini: 处理视频文件失败 - {str(e)}")
            return None
    
    def _prepare_file_data(
        self, 
        file: Optional[FileData]
    ) -> Optional[Dict[str, str]]:
        """
        准备文件数据
        
        从 FILE 类型提取文件数据
        
        Args:
            file: FileData 对象（来自 LoadFile 节点）
        
        Returns:
            文件数据字典，包含 mime_type 和 data
        """
        if file is None:
            return None
        
        return {
            "mime_type": file.mime_type,
            "data": file.data
        }
    
    def _parse_dual_output(self, raw_response: Dict) -> Tuple[str, str]:
        """
        解析包含思考内容和主要内容的响应
        
        Args:
            raw_response: API 原始响应字典
        
        Returns:
            (主要内容, 思考内容)
        """
        candidates = raw_response.get("candidates", [])
        if not candidates:
            return ("", "")
        
        parts = candidates[0].get("content", {}).get("parts", [])
        
        thought_text = ""
        main_text = ""
        
        for part in parts:
            if part.get("thought") is True:
                # 思考部分
                thought_text = part.get("text", "")
            elif "thoughtSignature" in part or "text" in part:
                # 主要内容
                main_text = part.get("text", "")
        
        return (main_text, thought_text)
    
    def generate(
        self,
        模型: str,
        提示词: str,
        思考等级: str,
        图片: Optional[torch.Tensor] = None,
        视频=None,
        文件: Optional[FileData] = None
    ) -> Tuple[str]:
        """
        生成文本
        
        Args:
            模型: 使用的模型名称
            提示词: 用户提示词
            思考等级: 思考等级选项
            图片: 输入图片
            视频: 输入视频
            文件: 输入文件（PDF/TXT）
        
        Returns:
            (主要内容, 思考内容)
        """
        start_time = time.time()
        
        try:
            # 初始化 API 客户端
            if self.client is None:
                try:
                    self.client = GeminiFlashClient()
                except ValueError as e:
                    raise ValueError(f"初始化失败: {str(e)}")
            
            # 准备图片数据
            image_data = self._prepare_image_data(图片)
            if image_data:
                print(f"Google Gemini: 输入 {len(image_data)} 张图片")
            
            # 准备视频数据
            video_data = self._prepare_video_data(视频)
            if video_data:
                print(f"Google Gemini: 输入视频 ({video_data['mime_type']})")
            
            # 准备文件数据
            document_data = self._prepare_file_data(文件)
            if document_data:
                file_type = "PDF" if document_data['mime_type'] == "application/pdf" else "TXT"
                print(f"Google Gemini: 输入文件 ({file_type})")
            
            # 构建输入描述
            input_desc = []
            if 提示词:
                input_desc.append("文本")
            if image_data:
                input_desc.append(f"{len(image_data)}张图片")
            if video_data:
                input_desc.append("视频")
            if document_data:
                input_desc.append("文件")
            
            print(f"Google Gemini: 模型 = {模型}")
            print(f"Google Gemini: 多模态输入 ({', '.join(input_desc)})")
            print(f"Google Gemini: 思考等级 = {思考等级}")
            
            # 获取端点和构建请求体
            endpoint = self.client.get_endpoint(model=模型)
            request_body = self.client.build_request_body(
                prompt=提示词,
                model=模型,
                thinking_level=思考等级,
                image_data=image_data,
                video_data=video_data,
                document_data=document_data
            )
            
            print(f"Google Gemini: 发送请求...")
            
            # 调用底层 API 获取原始响应
            async def get_raw_response():
                return await self.client.request_async(
                    endpoint, 
                    request_body, 
                    session=None
                )
            
            # 在独立线程中执行异步请求
            raw_response = self.client.run_async_in_thread(get_raw_response())
            
            # 计算耗时
            elapsed = time.time() - start_time
            
            # 解析响应，分离主要内容和思考内容
            main_text, thought_text = self._parse_dual_output(raw_response)
            
            # 打印响应 token 用量
            usage = raw_response.get("usageMetadata", {})
            prompt_tokens = usage.get("promptTokenCount", 0)
            candidates_tokens = usage.get("candidatesTokenCount", 0)
            thoughts_tokens = usage.get("thoughtsTokenCount", 0)
            total_tokens = usage.get("totalTokenCount", 0)
            finish_reason = ""
            candidates = raw_response.get("candidates", [])
            if candidates:
                finish_reason = candidates[0].get("finishReason", "")
            
            print(f"Google Gemini: 生成完成 (耗时: {elapsed:.2f}s)")
            print(f"Google Gemini: finishReason = {finish_reason}")
            print(f"Google Gemini: Token 用量 — 输入: {prompt_tokens}, 输出: {candidates_tokens}, 思考: {thoughts_tokens}, 合计: {total_tokens}")
            print(f"Google Gemini: 主要内容长度: {len(main_text)} 字符")
            print(f"Google Gemini: 思考内容长度: {len(thought_text)} 字符")
            
            # 输出预览
            if main_text:
                preview = main_text[:100] + "..." if len(main_text) > 100 else main_text
                print(f"Google Gemini: 主要内容预览: {preview}")
            
            return (main_text, thought_text)
        
        except ValueError as e:
            # 检测是否为授权错误
            if str(e) == "未授权！":
                print("请联系作者授权后方可使用！")
                raise ValueError("未授权！") from None
            else:
                # 用户输入错误 - 只显示简洁信息
                error_msg = str(e).split('\n')[0]  # 只取第一行
                print(f"Google Gemini: ❌ {error_msg}")
                raise ValueError(error_msg) from None
        
        except RuntimeError as e:
            # 日志只打第一行；报错框展示完整多行
            error_full = str(e)
            print(f"Google Gemini: ❌ {error_full.split('\n')[0]}")
            raise RuntimeError(error_full) from None
        
        except Exception as e:
            # 其他未知错误 - 只显示简洁信息
            error_msg = str(e).split('\n')[0]
            print(f"Google Gemini: ❌ {error_msg}")
            raise type(e)(error_msg) from None
        
        finally:
            if self.client is not None:
                try:
                    balance_data = self.client.query_balance_sync()
                    balance_info = self.client.format_balance_info(balance_data)
                    print(f"Google Gemini: {balance_info}")
                except Exception:
                    pass
