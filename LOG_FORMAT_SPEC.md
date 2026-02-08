# 日志格式规范

本文档定义了 Comfyui_o1key 项目中日志打印的统一格式规范。

## 设计原则

- **极简**：删除冗余信息，只保留用户关心的内容
- **完整**：展示所有关键阶段和状态
- **可读**：使用清晰的层级结构和符号
- **一致**：所有节点遵循统一格式

## 日志格式

### 基本结构

```
节点名: 模式信息 | 参数概览 | 数量信息
[阶段/总数] 阶段名 状态符号 耗时 | 附加信息
  ├─ 子阶段1: 详情
  └─ 子阶段2: 详情
[阶段/总数] 阶段名 状态符号 耗时 | 附加信息
[阶段/总数] 完成！总耗时 XX.XXs | 结果统计
```

### 符号规范

- **状态符号**
  - `✓` - 成功
  - `✗` - 失败
  - `⏳` - 进行中（可选，用于长时间操作）

- **层级符号**
  - `├─` - 中间项
  - `└─` - 最后一项
  - 使用两个空格缩进

### 阶段定义

标准流程分为 4 个阶段：

1. **[1/4] 请求构建** - 构造请求体、编码图像
2. **[2/4] API 请求** - 发送请求、等待响应
3. **[3/4] 响应解析** - 解析响应、下载图像
4. **[4/4] 完成** - 任务完成总结

### 数据格式化规则

#### 1. 时间显示

```python
if time_seconds < 1:
    f"{time_seconds:.3f}s"  # 例: 0.234s
else:
    f"{time_seconds:.2f}s"  # 例: 45.67s
```

#### 2. 数据大小显示

```python
if size_bytes < 1024 * 1024:  # < 1MB
    f"{size_bytes / 1024:.2f}KB"  # 例: 256.45KB
else:
    f"{size_bytes / (1024 * 1024):.2f}MB"  # 例: 3.45MB
```

#### 3. 速度显示

```python
f"{speed_mbps:.2f}MB/s"  # 例: 1.63MB/s
```

## 场景示例

### 场景 1: 文生图（Base64 返回）

```log
Nano Banana Pro: 文生图模式 | 2K 1:1 | 1张
[1/4] 请求构建 ✓ 0.003s | 0.15MB
[2/4] API 请求 ✓ 45.91s
  ├─ 连接: 0.23s
  └─ 响应: 45.68s (5.1MB)
[3/4] 响应解析 ✓ 0.46s | Base64 3.39MB → 2048x2048
[4/4] 完成！总耗时 46.37s | 成功 1张
```

**要点说明：**
- 首行显示：模式 + 分辨率 + 宽高比 + 数量
- API 请求阶段显示连接和响应时间
- 响应解析阶段显示返回格式（Base64）、大小、图像尺寸
- 最后汇总总耗时和成功数量

### 场景 2: 图生图（URL 返回）

```log
Nano Banana Pro: 图生图模式 (输入2张) | 2K 1:1 | 1张
[1/4] 请求构建 ✓ 0.24s | 8.73MB
[2/4] API 请求 ✓ 42.77s
  ├─ 连接: 0.20s
  └─ 响应: 42.57s (2.1KB)
[3/4] 响应解析 ✓ 2.23s | URL → 下载 3.45MB (1.63MB/s) → 2048x2048
[4/4] 完成！总耗时 45.00s | 成功 1张
```

**要点说明：**
- 图生图模式需标注输入图像数量
- 请求构建时间和大小会增加（因为包含输入图像）
- URL 模式需显示下载大小和速度

### 场景 3: 批量生成

```log
BatchNanoBananaPro: 批量任务 | 1:1 配对模式 | 共 10 任务
[1/10] 请求构建 ✓ 0.15s | 4.2MB
[1/10] API 请求 ✓ 38.5s
  ├─ 连接: 0.18s
  └─ 响应: 38.3s (1.9KB)
[1/10] 响应解析 ✓ 1.2s | URL → 2048x2048
[2/10] 请求构建 ✓ 0.14s | 4.1MB
[2/10] API 请求 ✓ 37.8s
  ├─ 连接: 0.19s
  └─ 响应: 37.6s (2.0KB)
[2/10] 响应解析 ✓ 1.3s | URL → 2048x2048
...
[10/10] 响应解析 ✓ 1.1s | URL → 2048x2048
============================================================
完成！总耗时 125.3s | 成功: 10/10 | 生成 10 张
保存路径: D:\output\
```

**要点说明：**
- 显示任务序号 [当前/总数]
- 每个任务显示完整的 3 个阶段
- 最后用分隔线汇总结果

### 场景 4: 多张批量生成（单个提示词）

```log
Nano Banana Pro: 文生图模式 | 2K 1:1 | 4张
[1/4] 请求构建 ✓ 0.003s | 0.15MB
[2/4] API 请求（并发4个）
  [1/4] API 请求 ✓ 45.2s
    ├─ 连接: 0.23s
    └─ 响应: 44.97s (5.1MB)
  [2/4] API 请求 ✓ 46.1s
    ├─ 连接: 0.25s
    └─ 响应: 45.85s (5.2MB)
  [3/4] API 请求 ✓ 44.8s
    ├─ 连接: 0.21s
    └─ 响应: 44.59s (5.0MB)
  [4/4] API 请求 ✓ 45.5s
    ├─ 连接: 0.24s
    └─ 响应: 45.26s (5.1MB)
[3/4] 响应解析 ✓ 1.8s | Base64 → 4 张 2048x2048
[4/4] 完成！总耗时 47.0s | 成功 4张
```

### 场景 5: 请求失败

```log
Nano Banana Pro: 文生图模式 | 2K 1:1 | 1张
[1/4] 请求构建 ✓ 0.003s | 0.15MB
[2/4] API 请求 ✗ 45.2s
  └─ 错误: 内容审核拒绝 (candidatesTokenCount=0)
     建议: 检查提示词是否包含敏感内容
```

**要点说明：**
- 使用 `✗` 标识失败
- 显示具体错误原因
- 提供解决建议

### 场景 6: 部分失败（批量）

```log
BatchNanoBananaPro: 批量任务 | 1:1 配对模式 | 共 10 任务
...
============================================================
完成！总耗时 125.3s | 成功: 8/10 | 生成 8 张
保存路径: D:\output\
失败 2个: 任务3,7 - 内容审核拒绝
```

## 删除的冗余信息清单

以下信息已从日志中移除：

- ❌ API 端点完整 URL（开发调试用）
- ❌ 超时设置值（配置项）
- ❌ Base64 数据预览字符串
- ❌ 图片下载的完整 URL
- ❌ 过多的表情符号
- ❌ 阶段之间的多余空行
- ❌ "发送请求"等冗余状态提示

## 实现指南

### 节点层（nodes/*.py）

节点负责打印：
1. 首行概览
2. 各阶段的汇总（调用 client 后打印）
3. 最终完成统计

```python
# 首行概览
mode = f"图生图模式 (输入{len(input_images)}张)" if input_images else "文生图模式"
print(f"Nano Banana Pro: {mode} | {分辨率} {宽高比} | {生图数量}张")

# 阶段1
build_start = time.time()
# ... 构建请求
build_time = time.time() - build_start
size_mb = len(json.dumps(request_body)) / (1024 * 1024)
print(f"[1/4] 请求构建 ✓ {format_time(build_time)} | {format_size(size_mb)}")

# 阶段2+3（由 client 内部打印）
images, format_info = client.generate_sync(...)

# 阶段4
total_time = time.time() - start_time
print(f"[4/4] 完成！总耗时 {format_time(total_time)} | 成功 {len(images)}张")
```

### 客户端层（clients/*.py）

客户端负责打印：
1. API 请求阶段的详细信息（连接、响应）
2. 响应解析阶段的格式信息（Base64/URL、大小、尺寸）

```python
# 阶段2: API 请求
print(f"[2/4] API 请求 ✓ {total_time:.2f}s")
print(f"  ├─ 连接: {connect_time:.3f}s")
print(f"  └─ 响应: {response_time:.2f}s ({format_size(response_size)})")

# 阶段3: 响应解析
if format_type == "base64":
    print(f"[3/4] 响应解析 ✓ {parse_time:.2f}s | Base64 {format_size(data_size)} → {width}x{height}")
else:  # URL
    print(f"[3/4] 响应解析 ✓ {parse_time:.2f}s | URL → 下载 {format_size(download_size)} ({download_speed:.2f}MB/s) → {width}x{height}")
```

## 工具函数

建议在 `utils/` 中添加统一的格式化函数：

```python
def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 1:
        return f"{seconds:.3f}s"
    else:
        return f"{seconds:.2f}s"

def format_size(bytes_value: float) -> str:
    """格式化数据大小"""
    if bytes_value < 1024 * 1024:
        return f"{bytes_value / 1024:.2f}KB"
    else:
        return f"{bytes_value / (1024 * 1024):.2f}MB"

def format_speed(bytes_per_second: float) -> str:
    """格式化速度"""
    return f"{bytes_per_second / (1024 * 1024):.2f}MB/s"
```

## 注意事项

1. **保持错误信息的详细性**：失败时仍需提供完整的错误原因和建议
2. **批量任务显示进度**：避免长时间无反馈
3. **多语言考虑**：日志使用中文，但符号保持统一
4. **终端兼容性**：确保特殊字符（✓ ✗ ├ └）在 Windows/Linux/Mac 都能正确显示

## 版本历史

- **v1.0** (2026-02-09): 初始规范，实现极简日志格式
