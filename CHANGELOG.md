# Changelog

本项目的所有重要变更都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

---

## [1.10.1] - 2026-02-06

### Added ⭐
- **文件编码规范化系统** - 彻底解决 GitHub 中文乱码问题
  - 新增 `.gitattributes` - 强制所有文本文件使用 UTF-8 编码和 LF 行结束符
  - 新增 `.editorconfig` - 编辑器自动配置编码和格式
  - 新增 `fix_encoding.ps1` - Windows 自动修复编码脚本
  - 新增 `fix_encoding.sh` - Linux/Mac 自动修复编码脚本
  - 新增 `编码修复指南.md` - 详细的编码问题排查和修复文档

### Changed
- **开发指南更新** (`.cursorrules`)
  - 新增"编码规范"章节（位于"对话原则"之后）
  - 强制要求所有文本文件使用 UTF-8 编码（无 BOM）
  - 强制要求行结束符使用 LF（.bat 文件除外使用 CRLF）
  - 提供常见编辑器（VSCode/Cursor）的配置方法
  - 提供文件编码验证和修复方法
  - 新增提交前编码检查清单

### Fixed 🐛
- **修复 GitHub 中文乱码问题**
  - 根本原因：Windows 系统默认使用 GBK 编码，GitHub 只支持 UTF-8
  - 解决方案：强制项目所有文本文件使用 UTF-8（无 BOM）编码
  - 提供一键修复脚本，自动转换所有文件编码
  - 配置 Git 属性，防止未来再次出现编码问题

### Benefits
- ✅ **彻底解决乱码** - 通过 `.gitattributes` 强制编码，从源头杜绝问题
- ✅ **自动修复** - 一键脚本转换所有文件，无需手动操作
- ✅ **编辑器集成** - `.editorconfig` 确保所有编辑器使用正确配置
- ✅ **详细文档** - 提供完整的问题排查、修复和预防指南
- ✅ **多平台支持** - Windows/Linux/Mac 都有对应的修复脚本
- ✅ **规范化流程** - 提交前检查清单，防止引入新问题

### Technical
- 所有文本文件（.py, .md, .txt, .json, .yaml, .sh）强制 UTF-8 无 BOM
- 行结束符统一为 LF（Unix 风格），.bat 文件除外（CRLF）
- Git 自动标准化行结束符（`text=auto eol=lf`）
- 修复脚本支持批量转换和编码检测
- 排除 `__pycache__`, `.git`, `node_modules`, `venv` 等目录

---

## [1.10.0] - 2026-02-06

### Added ⭐
- **自动更新系统** - 让用户轻松更新插件到最新版本
  - 新增 `update.bat` - Windows 自动更新脚本
  - 新增 `update.sh` - Linux/Mac 自动更新脚本
  - 新增 `version.txt` - 版本号管理文件
  - 新增 `utils/update_checker.py` - 启动时自动检查更新
  - 新增更新检查功能：每次启动 ComfyUI 时自动检测是否有新版本
  
- **更新脚本功能**：
  - ✅ 自动检查远程更新
  - ✅ 自动备份和恢复 `.config` 配置文件
  - ✅ 自动拉取最新代码
  - ✅ 自动更新 Python 依赖包
  - ✅ 显示版本变更信息
  - ✅ 显示最近更新日志（前 20 行）
  - ✅ 友好的彩色终端输出（Linux/Mac）
  - ✅ 完善的错误处理和提示

### Changed
- **插件启动流程** (`__init__.py`)
  - 集成更新检查模块
  - 启动时自动检查是否有新版本
  - 如有更新，终端显示友好的更新提示
  - 静默失败机制，不影响插件正常加载

- **文档更新** (`README.md`)
  - 新增"🔄 更新插件"章节
  - 提供两种更新方法：自动更新（推荐）和手动更新
  - 详细的跨平台更新说明
  - 更新提示和注意事项

### Benefits
- 🎯 **用户友好** - 一键更新，无需手动操作 Git
- 🔒 **配置安全** - 自动备份恢复配置，不会丢失设置
- ⚡ **依赖同步** - 自动更新 Python 包，确保兼容性
- 📋 **信息透明** - 显示版本变更和更新日志
- 🌍 **跨平台** - 支持 Windows/Linux/Mac
- 🛡️ **稳定可靠** - 完善的错误处理，不影响插件运行

---

## [1.9.1] - 2026-02-05

### Security 🔐
- **错误信息伪装 - 极致隐蔽性**
  - 未配置情况下的错误信息伪装成商业授权提示
  - 前端弹框显示：`未授权！`
  - 控制台日志显示：`请联系作者授权后方可使用！`
  - 完全隐藏真实的配置方式（环境变量）
  - 让外部用户以为这是需要购买授权的商业插件

### Changed
- **配置模块** (`utils/config.py`)
  - 将未找到 API 密钥的错误消息改为简洁的"未授权！"
  - 移除所有暴露配置方式的信息

- **节点异常处理** (所有节点)
  - `NanoBananaPro` - 添加授权错误检测，特殊处理授权失败日志
  - `BatchNanoBananaPro` - 添加授权错误检测，特殊处理授权失败日志
  - `GoogleGemini` - 添加授权错误检测，特殊处理授权失败日志
  - 其他 ValueError 类型仍正常显示详细错误信息

### Benefits
- ✅ **极致隐蔽** - 完全看不出需要配置环境变量
- ✅ **伪装商业** - 外部用户以为需要付费授权
- ✅ **信息安全** - 不泄露任何配置相关信息
- ✅ **功能完整** - 其他错误类型仍正常提示
- ✅ **向后兼容** - 已配置用户完全不受影响

---

## [1.9.0] - 2026-02-05

### Security 🔐
- **配置方式优化 - 提升隐蔽性和安全性**
  - 移除所有配置相关文件（`setup_config.ps1`, `.config`, `.config.example`）
  - 改为使用系统环境变量存储 API 密钥
  - 避免在项目中留下配置痕迹，提高隐蔽性
  - 配置方式对外部用户完全不可见

### Changed
- **配置管理模块重构** (`utils/config.py`)
  - 调整读取优先级：环境变量优先 > .config 文件（向后兼容）
  - 简化错误提示：仅提示设置环境变量，不再提及配置文件
  - 更新模块说明：从"处理 .config 文件"改为"处理环境变量"

- **文档更新**
  - `README.md` - 简化配置说明，仅保留环境变量设置方法
  - `批量节点使用指南.md` - 更新常见问题中的配置说明
  - `.gitignore` - 注释配置文件规则（已弃用）

### Benefits
- ✅ **高隐蔽性** - 项目中无任何配置相关文件
- ✅ **高安全性** - 敏感信息存储在系统级别，不在项目目录
- ✅ **简化维护** - 一行命令创建/更新/删除配置
- ✅ **多项目共享** - 环境变量可被其他项目复用
- ✅ **向后兼容** - 仍支持从 .config 文件读取（如果存在）

---

## [1.8.0] - 2026-02-04

### Added
- **集成 ComfyUI 原生进度条** 🎉
  - `NanoBananaPro` 节点现在支持 UI 绿色进度条显示
  - `BatchNanoBananaPro` 节点现在支持 UI 绿色进度条显示
  - 使用 `comfy.utils.ProgressBar` 实现实时进度更新
  - 进度条在节点上方显示，从 0% 平滑更新到 100%
  - 兼容性检查：如果 ProgressBar 不可用，自动降级到终端进度显示

### Improved
- **用户体验提升**
  - 生图过程中可视化进度反馈更直观
  - 节点运行时自动显示绿色边框（ComfyUI 原生）
  - 保留详细的终端进度日志，方便调试
  - 批量提示词模式下进度条总数自动调整

### Technical
- 在 `nodes/nano_banana_pro.py` 中集成 ProgressBar
  - 创建进度条实例：`ProgressBar(生图数量)`
  - 在 `progress_callback` 中调用 `pbar.update(1)`
  - 批量提示词模式重新创建进度条以匹配实际总数
- 在 `nodes/batch_nano_banana_pro.py` 中集成 ProgressBar
  - 将 `pbar` 参数传递给 `_process_batch_async` 方法
  - 每完成一个任务立即更新进度条
  - 支持大批量任务的实时进度显示
- 添加 `PROGRESS_BAR_AVAILABLE` 标志进行兼容性检测

### Impact
- ✅ 所有图像生成节点现在都有 UI 进度条
- ✅ 后续新增的视频生成节点可直接复用此实现
- ✅ 不影响现有功能，完全向后兼容

---

## [1.7.0] - 2026-02-03

### Added
- **新增 Google Gemini 节点** (`GoogleGemini`)
  - 用于调用 Gemini 3 Flash 模型进行多模态文本生成
  - **输入支持**：
    - 提示词（必填）：用户提示词，支持多行
    - 系统指令（可选）：系统级指令，引导模型行为
    - 思考深度：不思考（默认）/ 高
    - 图片（可选）：支持 ComfyUI IMAGE 类型输入
    - 视频（可选）：支持 ComfyUI VIDEO 类型输入
  - **输出**：文本内容（STRING 类型）
  - **支持的视频格式**：mp4, mpeg, mov, avi, flv, webm, wmv, 3gpp
  - **端点映射**：
    - 不思考 → `/v1beta/models/gemini-3-flash-preview-nothinking:generateContent`
    - 高 → `/v1beta/models/gemini-3-flash-preview-high:generateContent`

- **新增 Gemini Flash API 客户端** (`clients/gemini_flash_client.py`)
  - 继承 `BaseAPIClient` 基类
  - 支持系统指令配置
  - 支持图片和视频的 base64 编码发送
  - 智能超时设置（视频请求 5 分钟，其他 3 分钟）

### Technical
- 新增 `GeminiFlashClient` 类处理 Gemini Flash 模型 API 调用
- 新增 `GoogleGemini` 节点类实现多模态文本生成
- 支持自动检测视频 MIME 类型
- 视频文件大小限制 20MB

---

## [1.6.2] - 2026-02-02

### Added
- **模型端点配置化** (`models_config.py`)
  - 在模型配置中新增 `endpoint` 字段，集中管理每个模型的 API 端点
  - 新增 `get_model_endpoint()` 工具函数，用于获取模型端点
  - 添加新模型时只需在配置文件中填写端点，无需修改代码

### Changed
- **简化端点获取逻辑** (`clients/gemini_client.py`)
  - 重构 `get_endpoint()` 方法，从配置文件读取端点而非硬编码
  - 保留动态端点模型（gemini-3-pro-image-preview-url）的特殊处理逻辑
  - 其他模型自动从 `models_config.py` 读取端点配置

### Improved
- **增强配置验证** (`models_config.py`)
  - 验证非动态端点模型必须配置有效的 `endpoint`
  - 验证端点格式是否正确（应以 `/v1beta/models/` 开头）
  - 更新必需字段列表，包含 `endpoint` 字段

- **更新开发文档** (`.cursorrules`)
  - 更新模型配置示例，说明 `endpoint` 字段的使用方法
  - 更新添加新模型的指南，强调配置端点的方式
  - 更新工具函数列表，添加 `get_model_endpoint()` 说明
  - 更新配置验证规则说明

### Benefits
- 降低维护成本：添加新模型只需修改配置文件
- 提高可读性：端点集中管理，一目了然
- 减少错误：配置验证确保端点格式正确
- 保持灵活性：动态端点模型仍可使用代码逻辑

---

## [1.6.1] - 2026-02-02

### Fixed
- **修复 gemini-3-pro-image-preview-flatfee 模型 504 错误**
  - 暂时禁用 `gemini-3-pro-image-preview-flatfee` 模型（端点返回 504 Gateway Timeout）
  - 更新模型描述标注"暂时不可用-504错误"
  - 建议用户使用其他可用模型（如 gemini-3-pro-image-preview 或 gemini-3-pro-image-preview-url）

### Improved
- **增强 HTTP 错误处理** (`clients/base_client.py`)
  - 新增针对 504 Gateway Timeout 的友好错误提示
    - 说明原因：服务器响应超时或端点暂时不可用
    - 提供解决建议：尝试其他模型、稍后重试、降低分辨率等
  - 新增针对 503 Service Unavailable 的错误提示
    - 说明原因：模型服务过载或维护中
    - 提供解决建议：稍后重试或尝试其他模型
  - 新增针对 429 Too Many Requests 的错误提示
    - 说明原因：API 配额用尽或请求过于频繁
    - 提供解决建议：等待后重试或检查配额
  - 新增针对 404 Not Found 的错误提示
    - 说明原因：端点路径错误或模型不存在
    - 提供解决建议：检查模型名称或使用其他模型
  - 统一错误信息格式：错误类型 + 原因 + 建议
  - 同时优化 POST 和 GET 请求的错误处理

### Technical
- 在 `request_async()` 和 `request_get_async()` 中添加状态码判断逻辑
- 提供更详细的错误诊断信息，帮助用户快速定位和解决问题

---

## [1.6.0] - 2026-02-02

### Added
- **模型管理系统** (`models_config.py`)
  - 创建集中式模型配置文件，所有 Nano Banana Pro 支持的模型统一管理
  - 支持快速添加新模型、临时关闭或启用模型
  - 模型配置包含：模型ID、描述、启用状态、端点类型
  - 提供工具函数：
    - `get_enabled_models()` - 获取启用的模型列表
    - `get_all_models()` - 获取所有模型（包括禁用的）
    - `get_model_config()` - 获取指定模型的完整配置
    - `is_model_enabled()` - 检查模型是否启用
    - `get_model_description()` - 获取模型描述
    - `get_endpoint_type()` - 获取端点类型
  - 自动配置验证，确保配置完整性和合法性

### Changed
- **NanoBananaPro 节点重构**
  - 移除硬编码的 `MODELS` 列表
  - 改为从 `models_config.py` 动态加载模型列表
  - 节点在 ComfyUI 中显示的模型列表自动同步配置文件

- **BatchNanoBananaPro 节点重构**
  - 移除硬编码的 `MODELS` 列表
  - 改为从 `models_config.py` 动态加载模型列表
  - 保持与 NanoBananaPro 节点的模型列表一致性

### Improved
- **开发指南更新** (`.cursorrules`)
  - 新增"模型管理系统"章节
  - 详细说明模型配置文件结构和字段含义
  - 提供添加新模型、关闭/启用模型、修改描述等常见操作指南
  - 新增工具函数使用示例和节点集成说明
  - 补充配置验证机制和最佳实践建议

- **目录结构更新**
  - 在开发指南中添加 `models_config.py` 文件说明
  - 标记为模型配置中心 ⭐

### Benefits
- ✅ **集中管理** - 所有模型定义在一个文件，易于维护
- ✅ **易于扩展** - 添加新模型只需在配置文件中添加一个字典
- ✅ **快速开关** - 修改 `enabled` 字段即可临时关闭或启用模型
- ✅ **文档化** - 每个模型都有 `description` 说明特点和适用场景
- ✅ **类型安全** - Python 文件支持代码提示和类型检查
- ✅ **自动同步** - 所有节点自动使用最新的模型配置

---

## [1.5.7] - 2026-02-02

### Performance 🚀
- **极致性能优化：彻底解决异步阻塞问题**
  - 将 `parse_response()` 改为 `parse_response_async()`，实现完全异步的图片下载
  - 使用 `aiohttp` 替代同步的 `requests.get()` 下载图片
  - **问题**：之前在异步事件循环中使用同步 HTTP 请求会阻塞整个事件循环
  - **影响**：虽然 API 请求是并发的，但图片下载变成了串行操作
  - **效果**：现在图片下载也是完全并发的，真正实现端到端的异步性能
  
- **性能提升幅度**：
  - 使用 `gemini-3-pro-image-preview-url` 模型时提升最明显
  - 批量生成 4 张图时，从"串行下载 4 张"变为"并发下载 4 张"
  - 预计性能提升 2-4 倍（取决于网络延迟和图片大小）

### Changed
- `GeminiAPIClient.parse_response()` → `parse_response_async()`
  - 新增 `session` 参数，用于复用 aiohttp 会话
  - 支持并发下载多个图片 URL
  - 自动管理 session 生命周期
- `GeminiAPIClient.generate_single_async()` 现在调用异步解析方法
- 移除 `requests` 依赖，统一使用 `aiohttp`

### Impact
- 所有节点自动受益：
  - ✅ `NanoBananaPro` - 批量生成速度显著提升
  - ✅ `BatchNanoBananaPro` - 大批量任务性能大幅改善
  - ✅ 所有使用 `gemini-3-pro-image-preview-url` 模型的场景

---

## [1.5.6] - 2026-02-02

### Improved
- **优化批量生成实时进度显示** (`Nano Banana Pro`)
  - 使用 `asyncio.as_completed` 替代 `asyncio.gather`，实现真正的实时进度
  - 每完成一个请求立即显示进度，而非等待所有请求完成后批量显示
  - 进度信息包含成功/失败状态：
    - 成功：`✓ [1/4] 第 1 张生成成功`
    - 失败：`✗ [2/4] 生成失败 - 错误原因`
  - 最终统计显示成功和失败数量

### Fixed
- **修复批量生成失败信息丢失问题**
  - 之前：失败的请求被静默忽略，用户不知道哪些请求失败
  - 现在：每个失败的请求都会显示错误原因，方便排查问题

### Technical
- 更新 `generate_batch_async()` 和 `generate_multi_prompts_async()` 方法
- 进度回调签名变更：`(current, total)` → `(current, total, success, error_msg)`
- 错误信息自动截取第一行，避免过长输出

---

## [1.5.5] - 2026-02-02

### Added
- **增强 API 错误处理机制**
  - 新增 `candidatesTokenCount = 0` 检测（最高优先级）
    - 自动检测内容审核拒绝情况
    - 提供明确的拒绝原因和改进建议
  - 新增 `finishReason` 异常检测（次优先级）
    - 支持检测 `PROHIBITED_CONTENT`（违禁内容）
    - 支持检测 `SAFETY`（安全过滤器）
    - 支持检测 `RECITATION`（版权问题）
    - 支持检测 `MAX_TOKENS`（Token 超限）
    - 针对每种错误类型提供具体的解决建议
  - 新增 API 文本响应拒绝检测
    - 当 API 返回文本而非图像时，自动提取拒绝说明
    - 直接展示 API 的拒绝理由给用户

### Improved
- **优化错误提示格式** (`clients/gemini_client.py`)
  - 统一错误信息格式：错误类型 + 原因 + 建议
  - 所有错误以 `RuntimeError` 抛出，便于节点层面捕获
  - 提供清晰的多行格式化错误信息
  - 包含针对性的操作建议，帮助用户快速解决问题

### Technical
- 在 `GeminiAPIClient.parse_response()` 方法中实现三层错误检测
- 错误检测按优先级顺序执行，确保最重要的问题优先报告
- 保持向后兼容，不影响正常的图像生成流程

---

## [1.5.4] - 2026-02-01

### Added
- **新增余额查询功能**
  - 在每次图像生成请求完成后自动查询并显示用户余额
  - 支持查询 API 名称和当前可用余额
  - 余额格式：`当前余额：$XX.XX | API：xxx`
  - 查询失败时显示警告信息，不影响主流程
  - 适用于 `NanoBananaPro` 和 `BatchNanoBananaPro` 节点

### Changed
- **扩展 API 客户端功能** (`clients/base_client.py`)
  - 新增 `request_get_async()` 方法支持 GET 请求
  - 扩展 `get_headers()` 方法支持 Bearer Token 认证
  - 兼容现有的 x-goog-api-key 认证方式

- **增强 Gemini 客户端** (`clients/gemini_client.py`)
  - 新增 `query_balance_async()` 异步查询余额方法
  - 新增 `query_balance_sync()` 同步查询余额方法（用于节点）
  - 新增 `format_balance_info()` 格式化余额信息方法
  - 余额转换公式：实际显示 = total_available / 500000

---

## [1.5.3] - 2026-02-01

### Changed
- **重大更新新手使用指南** (`GUIDE.md`)
  - 新增批量提示词功能详细说明（节点详细说明部分）
  - 新增场景 5：批量提示词生成（多提示词并发）
  - 新增场景 6：批量提示词 + 图生图（共享参考图）
  - 新增场景 7：批量提示词精准匹配规则详解
    - 单提示词模式与批量提示词模式对比
    - 详细的行为示例（纯文生图、图生图、多参考图）
    - 常见误区与正确用法对照
    - 使用决策树帮助用户选择合适的节点和模式
  - 新增节点功能对比表和提示词模式对比表
  - 新增 Q6-Q7：批量提示词相关常见问题
  - 新增技巧 7：批量提示词最佳实践（4个子技巧）
  - 更新场景编号（原场景 6-8 → 新场景 8）
  - 修正批量提示词的描述（所有提示词共享输入图像，而非 1:1 匹配）
  - 强化对 `---` 分隔符格式要求的说明

---

## [1.5.2] - 2026-02-01

### Added
- **新增新手使用指南** (`GUIDE.md`)
  - 详细的安装配置步骤
  - 三个核心节点的完整文档和参数说明
  - 六种常见使用场景的工作流示例
  - 常见问题解答和解决方案
  - 六个进阶使用技巧
  - 面向初次使用插件的用户，提供从零到一的完整指导

### Changed
- **更新开发规范** (`.cursorrules`)
  - 新增"新手指南维护规则"章节
  - 规定每次新增或变更节点时必须同步更新 `GUIDE.md`
  - 提供节点文档和使用场景的标准模板
  - 明确文档维护的五大原则（用户视角、实用性、完整性、同步性、可读性）

---

## [1.5.1] - 2026-02-01

### Fixed
- **修复批量 Nano Banana Pro 节点事件循环冲突**
  - 修复 `RuntimeError: Cannot run the event loop while another loop is running` 错误
  - 使用 `ThreadPoolExecutor` 在独立线程中运行异步事件循环
  - 避免与 ComfyUI 主事件循环冲突
  - 确保批量处理任务稳定执行

---

## [1.5.0] - 2026-02-01

### Changed
- **批量 Nano Banana Pro 节点重构** (`BatchNanoBananaPro`)
  - 参数重命名：
    - `手动参考图` → `加载参考图`
    - `预览图像` → `输出图像`
    - `配对模式` → `图片配对模式`
  - 配对模式选项值重命名：
    - `1:1索引配对` → `1:1`
    - `笛卡尔积` → `1*N`
  - 参数顺序调整：`文件夹2-4` 移到 `文件夹1` 下方（从 optional 移至 required）
  - 固定并发控制：移除 `最大并发数` 参数，默认自动管理（每批最多 100 并发）
  - 固定生成数量：移除 `每组生成数量` 参数，每组配对固定生成 1 张图

### Added
- **批量 Nano Banana Pro 节点新增像素缩放功能**
  - 新增 `像素缩放` 参数（BOOLEAN，默认 True）
  - 新增 `分辨率像素` 参数（FLOAT，默认 1.0，范围 0.1-100.0）
  - 支持对文件夹加载的图片和手动参考图进行缩放
  - 使用 Lanczos 重采样算法保持图片质量
  - 缩放在发送 API 前应用

### Removed
- **批量 Nano Banana Pro 节点移除处理报告功能**
  - 移除返回值中的 `处理报告` 字符串输出
  - 简化返回类型为单一 `IMAGE` 输出
  - 统计信息仍通过控制台打印输出

### Improved
- **输出图像优化**
  - `输出图像` 现在返回所有生成的图片（而非仅预览最后几张）
  - 提供完整的批处理结果输出

---

## [1.4.0] - 2026-02-01

### Added
- **批量 Nano Banana Pro 节点** (`BatchNanoBananaPro`)
  - 支持从 1-4 个文件夹批量加载图片
  - 两种配对模式：1:1 索引配对 / 笛卡尔积
  - 支持手动参考图输入（可与文件夹图片混合使用）
  - 智能命名保存（保留原始文件名 + 自动后缀）
  - 并发控制（默认最大 100，超过自动分批）
  - 完整的处理报告输出
  - 预览最后生成的图片

- **文件处理工具模块** (`utils/file_utils.py`)
  - `load_images_from_folder()` - 从文件夹加载图片
  - `pair_images_indexed()` - 1:1 索引配对
  - `pair_images_cartesian()` - 笛卡尔积配对
  - `generate_output_filename()` - 智能输出文件名生成
  - `save_image()` - 保存图片到指定路径

### Technical
- 新增 `ImageInfo` 命名元组，携带图片元数据
- 支持 jpg/jpeg/png/webp/bmp/gif 图片格式
- 文件名按字母顺序排序，确保配对顺序一致

---

## [1.3.0] - 2026-02-01

### Added
- **批量提示词功能（Nano Banana Pro）**
  - 支持使用单行 `---` 分隔符同时提交多个不同提示词
  - 所有提示词并发生成，提高效率
  - 每个提示词可生成指定数量的图像（提示词数量 × 生图数量）
  - 示例：3个提示词 × 2张/提示词 = 6张图
  - 触发条件：`---` 必须单独占据一行
  - 自动过滤空提示词

### Changed
- 优化 Nano Banana Pro 节点日志输出
  - 批量提示词模式显示 "X个提示词 × Y张/提示词 = Z张图"
  - 单提示词模式保持原有输出格式

### Technical
- 新增 `parse_batch_prompts()` 工具函数（utils/image_utils.py）
- 新增 `generate_multi_prompts_async()` 方法（clients/gemini_client.py）
- 新增 `generate_multi_prompts_sync()` 方法（clients/gemini_client.py）

---

## [1.2.1] - 2026-02-01

### Changed
- **加载批次图像（Nano Banana Pro）** 节点优化
  - 节点显示名称改为 "加载批次图像（Nano Banana Pro）"
  - 移除 "参考图数量" 参数，改为自动检测所有有效输入图像数量
  - "开启像素缩放" 改名为 "像素缩放"，类型改为 BOOLEAN 开关（默认开启）
  - "目标像素数_百万" 改名为 "分辨率像素"

### Improved
- 简化用户操作流程，无需手动设置图像数量
- 支持灵活的图像输入方式（1-14 张任意数量）

---

## [1.2.0] - 2026-02-01

### Added
- **加载批次图像** 节点 (`BatchImageLoader`)
  - 支持动态输入数量（1-14 张图像）
  - 可选的像素缩放功能（保持纵横比）
  - 使用 Lanczos 重采样算法进行高质量缩放
  - 支持设置目标像素数（0.1-100 百万像素）
  - 可与原生"加载图像"节点连接使用
  - 输出批次张量供其他节点使用

---

## [1.1.1] - 2026-02-01

### Changed
- 将 Nano Banana Pro 节点的随机种子参数改为 ComfyUI 原生格式
  - `随机种子` → `seed` (参数名符合 ComfyUI 标准)
  - 移除 `-1` 自动随机逻辑
  - 种子默认值改为 `0`，取值范围 `[0, 2^64-1]`

### Removed
- 移除 `control_after_generate` 参数（简化节点参数）

---

## [1.1.0] - 2026-02-01

### Changed
- 重构项目结构，模块化设计
  - 新增 `nodes/` 目录存放节点实现
  - 新增 `utils/` 目录存放工具函数
  - 新增 `clients/` 目录存放 API 客户端
- 封装图像转换工具到 `utils/image_utils.py`
- 封装配置管理到 `utils/config.py`
- 重构 API 客户端，拆分为基类和具体实现
- 精简文档结构，整合为 README.md + CHANGELOG.md + .cursorrules

### Added
- `BaseAPIClient` 抽象基类，支持快速开发新 API 客户端
- `.cursorrules` 开发指导文档，统一开发规范
- `get_api_key_or_raise()` 函数，简化密钥获取逻辑

---

## [1.0.0] - 2026-02-01

### Added
- 初始版本发布
- Nano Banana Pro 节点
  - 文生图功能
  - 图生图功能（最多 14 张输入图像）
  - 批量并发生成（最多 1000 张）
  - 多分辨率支持（1K / 2K / 4K）
  - 10 种宽高比选项
  - 可控随机种子
- 通过 api.o1key.com 调用 Gemini 3 Pro 模型
- 支持 .config 文件和环境变量配置 API 密钥
- 完善的错误处理和日志输出
