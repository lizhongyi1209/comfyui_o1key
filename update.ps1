# ============================================================
#  comfyui_o1key 插件自动更新脚本
#  使用方式：在本目录 Shift+右键 > 在此处打开 PowerShell 窗口，直接运行
#  作用：从 GitHub 拉取最新版本，强制覆盖本地（含跨版本更新）
# ============================================================

$Host.UI.RawUI.WindowTitle = "comfyui_o1key 更新工具"

function Write-Title {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "   comfyui_o1key 插件更新工具" -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Step {
    param([string]$msg)
    Write-Host ">> $msg" -ForegroundColor Yellow
}

function Write-OK {
    param([string]$msg)
    Write-Host "   [OK] $msg" -ForegroundColor Green
}

function Write-Fail {
    param([string]$msg)
    Write-Host "   [ERROR] $msg" -ForegroundColor Red
}

function Pause-Exit {
    param([int]$code = 0)
    Write-Host ""
    Write-Host "按任意键退出..." -ForegroundColor DarkGray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit $code
}

# ── 切换到脚本所在目录（Shift+右键打开时目录可能不对）──────────────────
Set-Location -Path $PSScriptRoot

Write-Title

# ── 1. 检查 git 是否安装 ────────────────────────────────────────────────
Write-Step "检查 Git 环境..."
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Fail "未找到 git 命令，请先安装 Git for Windows：https://git-scm.com"
    Pause-Exit 1
}
Write-OK "Git 已安装：$(git --version)"

# ── 2. 确认当前目录是 git 仓库 ─────────────────────────────────────────
Write-Step "检查插件目录..."
if (-not (Test-Path ".git")) {
    Write-Fail "当前目录不是 Git 仓库，请检查脚本是否放在插件根目录"
    Pause-Exit 1
}
Write-OK "目录正常：$PSScriptRoot"

# ── 3. 显示当前版本 ────────────────────────────────────────────────────
Write-Step "当前版本信息..."
$currentCommit = git rev-parse --short HEAD 2>&1
$currentDate   = git log -1 --format="%ci" 2>&1
Write-Host "   提交 Hash : $currentCommit" -ForegroundColor Gray
Write-Host "   提交时间 : $currentDate" -ForegroundColor Gray

# ── 4. 备份 .config（如果存在，防止意外覆盖） ─────────────────────────
$configPath   = Join-Path $PSScriptRoot ".config"
$configBakPath = Join-Path $PSScriptRoot ".config.bak"
if (Test-Path $configPath) {
    Copy-Item $configPath $configBakPath -Force
    Write-OK "已备份 .config -> .config.bak（更新后自动还原）"
}

# ── 5. 获取远程更新 ────────────────────────────────────────────────────
Write-Step "正在从 GitHub 获取最新代码..."
git fetch origin 2>&1 | ForEach-Object { Write-Host "   $_" -ForegroundColor DarkGray }
if ($LASTEXITCODE -ne 0) {
    Write-Fail "fetch 失败，请检查网络连接或 GitHub 访问是否正常"
    # 还原备份
    if (Test-Path $configBakPath) { Copy-Item $configBakPath $configPath -Force }
    Pause-Exit 1
}
Write-OK "远程代码获取成功"

# ── 6. 检查是否已经是最新 ─────────────────────────────────────────────
$localHash  = git rev-parse HEAD 2>&1
$remoteHash = git rev-parse origin/main 2>&1
if ($localHash -eq $remoteHash) {
    Write-Host ""
    Write-Host "   已经是最新版本，无需更新。" -ForegroundColor Green
    if (Test-Path $configBakPath) { Remove-Item $configBakPath -Force }
    Pause-Exit 0
}

# ── 7. 显示即将更新的变更 ──────────────────────────────────────────────
Write-Step "即将更新的内容："
git log HEAD..origin/main --oneline 2>&1 | ForEach-Object {
    Write-Host "   $_" -ForegroundColor DarkCyan
}

# ── 8. 强制重置到远程 main（处理跨版本 / 历史不一致情况） ──────────────
Write-Step "正在强制更新到最新版本..."

# 确保本地有 main 分支
$hasBranchMain = git branch --list main
if (-not $hasBranchMain) {
    git checkout -b main origin/main 2>&1 | Out-Null
} else {
    git checkout main 2>&1 | Out-Null
}

# 强制重置（无论历史是否一致，直接对齐远程）
git reset --hard origin/main 2>&1 | ForEach-Object { Write-Host "   $_" -ForegroundColor DarkGray }
if ($LASTEXITCODE -ne 0) {
    Write-Fail "更新失败（reset 出错）"
    if (Test-Path $configBakPath) { Copy-Item $configBakPath $configPath -Force }
    Pause-Exit 1
}

# 清理多余的未跟踪文件（可选，保留 .config）
git clean -fd --exclude=".config" --exclude=".config.bak" 2>&1 | ForEach-Object {
    Write-Host "   $_" -ForegroundColor DarkGray
}

Write-OK "代码已更新成功"

# ── 9. 还原 .config ────────────────────────────────────────────────────
if (Test-Path $configBakPath) {
    Copy-Item $configBakPath $configPath -Force
    Remove-Item $configBakPath -Force
    Write-OK ".config 已还原（API Key 保持不变）"
}

# ── 10. 显示更新后版本 ─────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "   更新完成！" -ForegroundColor Green
$newCommit = git rev-parse --short HEAD 2>&1
$newDate   = git log -1 --format="%ci" 2>&1
Write-Host "   新版本 Hash : $newCommit" -ForegroundColor Green
Write-Host "   提交时间   : $newDate" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "   请重启 ComfyUI 以加载最新插件。" -ForegroundColor Cyan
Write-Host ""

Pause-Exit 0
