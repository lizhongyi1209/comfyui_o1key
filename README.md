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

### 国内用户安装（GitHub 拉取慢或失败时）

使用 Gitee 镜像安装与更新，避免网络问题：

```bash
cd ComfyUI/custom_nodes
git clone https://gitee.com/resonLzy/comfyui_o1key.git
cd comfyui_o1key
pip install -r requirements.txt
```

自动更新脚本（见下方「更新插件」）已改为从 Gitee 拉取，国内用户可直接使用。

---

## ⚙️ 配置

### 获取 API 密钥

1. 访问 [vip.o1key.com](https://vip.o1key.com)
2. 注册并获取 API 密钥

### 配置方式

#### 配置 API 密钥（必需）

**方法一：快捷脚本配置（最简单）⭐**

我们提供了一键配置脚本，自动创建配置文件：

**Windows 用户：**
双击运行 `设置API密钥(win).bat`，按提示输入 API 密钥即可。

**Linux/Mac 用户：**
```bash
# 添加执行权限（仅首次需要）
chmod +x 设置API密钥(mac).sh

# 运行配置脚本
./设置API密钥(mac).sh
```

按提示输入 API 密钥，配置完成后重启 ComfyUI。

**方法二：环境变量（推荐）**

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

**方法三：手动创建配置文件**

在插件目录下创建 `.config` 文件（参考 `.config.example`）：
```
O1KEY_API_KEY=你的API密钥
```

> **⚠️ 安全提示**
> 
> `.config` 文件包含敏感信息，已添加到 `.gitignore` 中，不会被提交到版本控制。
> 请妥善保管你的 API 密钥，不要分享给他人。

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

### 界面更新

在 ComfyUI 左侧功能栏点击「更新」（位于「重启」下方）。按钮会从当前 Git 仓库的 `origin/main` 拉取最新版本。完成后点击「重启」使新版本生效。

界面更新需要通过 Git 安装、处于 `main` 分支，且节点包文件没有本地修改。更新仅允许快进，不会覆盖本地修改或删除配置。ZIP 安装、分支分叉或网络连接失败时，界面会显示原因，需要手动处理。

如果提示依赖列表已变化，请在 ComfyUI 使用的 Python 环境中执行：

```bash
cd ComfyUI/custom_nodes/comfyui_o1key
python -m pip install -r requirements.txt
```

### 手动更新

```bash
cd ComfyUI/custom_nodes/comfyui_o1key
git pull --ff-only origin main
python -m pip install -r requirements.txt
```

更新保留环境变量中配置的 API 密钥。启动时仍会检查是否有新版本。

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

### 维护者：发布流程与镜像同步

代码**先提交并推送到 GitHub**，再**同步到 Gitee 镜像**，国内用户通过 Gitee 拉取以解决网络问题。

**首次配置**（仅需一次）：
```bash
git remote add gitee https://gitee.com/resonLzy/comfyui_o1key.git
```

**每次发布**：
```bash
git push origin main    # 先更新 GitHub
git push gitee main     # 再同步到 Gitee 镜像
```

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
