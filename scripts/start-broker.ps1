param(
    [string]$Config = (Join-Path $PSScriptRoot "..\config.toml")
)
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $python) { throw "No broker Python found. Create .venv and install the broker there, or install the broker into the system Python (python -m pip install -e .) before starting it." }
}
& $python -m local_llm_broker.cli --config $Config serve

