$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

python -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw '当前 Python 环境没有 PyInstaller。请先运行：python -m pip install -r requirements-build.txt'
}

python -m PyInstaller --noconfirm --clean DatasetTool.spec
if ($LASTEXITCODE -ne 0) {
    throw 'PyInstaller 构建失败，请检查上方日志。'
}

$exe = Join-Path $PSScriptRoot 'dist\DatasetTool.exe'
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw "构建命令结束，但没有找到 exe：$exe"
}
Write-Host "构建完成：$exe"
