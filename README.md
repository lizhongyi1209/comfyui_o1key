# Comfyui_o1key

通过 `api.o1key.com` 调用 AI 模型的 ComfyUI 自定义节点集合。

## 功能特性

- 🎨 文生图 / 图生图
- 🔄 批量并发生成（最多 1000 张）
- 📐 10 种宽高比
- 🎯 3 种分辨率（1K / 2K / 4K）
- 🌱 可控随机种子

---

## 📦 安装

### 方法一：通过 ComfyUI Manager（推荐）

1. 在 ComfyUI 中打开 Manager
2. 搜索 `Comfyui_o1key`
3. 点击安装
4. 重启 ComfyUI

### 方法二：手动安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lizhongyi1209/comfyui_o1key.git
cd comfyui_o1key
pip install -r requirements.txt
```

然后重启 ComfyUI。

---

## ⚙️ 配置

### 获取 API 密钥

1. 访问 [vip.o1key.com](https://vip.o1key.com)
2. 注册并获取 API 密钥

### 配置方式

#### 配置 API 密钥（必需）

**方法一：环境变量（推荐）**

**Windows 用户：**
1. 右键 "此电脑" → 属性 → 高级系统设置 → 环境变量
2. 在"用户变量"中新建：
   - 变量名：`O1KEY_API_KEY`
   - 变量值：你的 API 密钥
3. 重启 ComfyUI

**Linux/Mac 用户：**

在 `~/.bashrc` 或 `~/.zshrc` 中添加：
```bash
export O1KEY_API_KEY="你的API密钥"
```

然后执行 `source ~/.bashrc` 并重启 ComfyUI。

**方法二：配置文件**

在插件目录下创建 `.config` 文件（参考 `.config.example`）：
```
O1KEY_API_KEY=你的API密钥
```

#### 配置 API 地址（可选）

默认使用 `https://vip.o1key.com`，通常无需修改。

如需自定义 API 地址，可通过以下方式：

1. **环境变量**（推荐）：
   ```bash
   # Windows
   set O1KEY_API_BASE_URL=https://your-api-domain.com
   
   # Linux/Mac
   export O1KEY_API_BASE_URL=https://your-api-domain.com
   ```

2. **配置文件**：在 `.config` 中添加：
   ```
   O1KEY_API_BASE_URL=https://your-api-domain.com
   ```

3. **修改默认值**：编辑 `utils/config.py` 中的 `DEFAULT_API_BASE_URL` 常量

---

## 🔄 更新插件

### 方法一：自动更新（推荐）⭐

**Windows 用户：**
1. 进入插件目录：`ComfyUI\custom_nodes\comfyui_o1key`
2. 双击运行 `update.bat`
3. 等待更新完成
4. 重启 ComfyUI

**Linux/Mac 用户：**
```bash
cd ComfyUI/custom_nodes/comfyui_o1key
chmod +x update.sh  # 首次运行需要添加执行权限
./update.sh
```

### 方法二：手动更新

```bash
cd ComfyUI/custom_nodes/comfyui_o1key
git pull origin main
pip install -r requirements.txt --upgrade
```

**💡 提示：**
- 自动更新脚本会自动备份和恢复你的 `.config` 配置文件
- 更新会保留环境变量中配置的 API 密钥
- 更新检查在每次启动 ComfyUI 时自动进行（不会影响性能）
- 如果发现新版本，终端会显示更新提示

---

## 📚 节点说明

### Nano Banana Pro

高性能图像生成节点，支持文生图和图生图。

**参数：**
- **提示词**：描述你想生成的图像
- **模型**：选择使用的 AI 模型
- **分辨率**：1K / 2K / 4K
- **宽高比**：1:1, 16:9, 9:16, 4:3, 3:4, 21:9, 9:21, 3:2, 2:3, 16:10
- **批次大小**：单次生成的图像数量（1-1000）
- **随机种子**：控制生成的随机性（-1 为随机）
- **输入图像**（可选）：用于图生图模式

### Batch Nano Banana Pro

批量并发生成节点，适合大量图像生成。

### Google Gemini

Google Gemini 模型节点，支持更多模型选择。

---

## 📝 更新日志

查看 [CHANGELOG.md](./CHANGELOG.md) 了解详细的版本更新记录。

---

## 📄 许可证

本项目采用 Apache License 2.0 许可证。

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## ⚠️ 开发者注意事项

### 文件编码要求

**所有文本文件必须使用 UTF-8 编码（无 BOM）！**

如果你在 GitHub 上看到中文乱码，说明文件编码有问题。请使用以下方法修复：

**Windows 用户：**

```powershell
.\fix_encoding.ps1
```

**Linux/Mac 用户：**

```bash
chmod +x fix_encoding.sh
./fix_encoding.sh
```

详细说明请查看 [编码修复指南.md](./编码修复指南.md)

---

## 📮 联系方式

- GitHub: [@lizhongyi1209](https://github.com/lizhongyi1209)
- 项目地址: https://github.com/lizhongyi1209/comfyui_o1key

---

**当前版本：v1.10.1**
