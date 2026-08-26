function claude {
    [CmdletBinding(PositionalBinding=$false)]
    param(
        [Parameter(Mandatory=$false)]
        [string]$Model, 
        
        [Parameter(Mandatory=$false, ValueFromRemainingArguments=$true)]
        [string[]]$RemainingArgs
    )

    # 1. Define paths
    $bridgeDir = "C:\Users\Satgu\Documents\VS Code\nvidia-claude-code-bridge"
    $yamlPath = "$bridgeDir\litellm_config.yaml"
    $venvDir = "$bridgeDir\.venv"
    $pythonExe = "$venvDir\Scripts\python.exe"
    $litellmExe = "$venvDir\Scripts\litellm.exe"
    $envFile = "$bridgeDir\.env"

    # Determine provider from .env if Model not specified
    if (-not $Model) {
        $provider = "kaggle"
        if (Test-Path $envFile) {
            $line = Get-Content $envFile | Where-Object { $_ -match "^\s*PRIMARY_PROVIDER\s*=" } | Select-Object -First 1
            if ($line -match '=\s*"?([a-zA-Z0-9_-]+)"?') {
                $provider = $matches[1].ToLower()
            }
        }
        if ($provider -eq "nvidia") {
            $Model = "nvidia-agent"
        } else {
            $Model = "kaggle-agent"
        }
    }

    # Ensure Kaggle credentials are valid before launching if using Kaggle
    if ($Model -like "kaggle*" -or -not (Test-Path "$bridgeDir\.kaggle_proxy.env")) {
        Write-Host "Verifying Kaggle Model Proxy authentication..." -ForegroundColor Cyan
        & $pythonExe "$bridgeDir\kaggle_auth.py" --status | Out-Null
    }

    # Load Kaggle proxy environment variables into environment before starting LiteLLM
    if (Test-Path "$bridgeDir\.kaggle_proxy.env") {
        Get-Content "$bridgeDir\.kaggle_proxy.env" | ForEach-Object {
            if ($_ -match '^\s*([A-Za-z0-9_]+)\s*=\s*(.*)$') {
                [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim())
            }
        }
    }

    # 2. Check if a LiteLLM proxy is already running on port 4000
    $port = 4000
    $connection = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue

    if (-not $connection) {
        $logOutPath = "$bridgeDir\litellm_out.log"
        $logErrPath = "$bridgeDir\litellm_err.log"

        Write-Host "Starting LiteLLM proxy background process on port $port..." -ForegroundColor Gray
        $proxyProcess = Start-Process -FilePath $litellmExe `
            -ArgumentList "--config `"$yamlPath`" --port $port --detailed_debug" `
            -WorkingDirectory $bridgeDir `
            -RedirectStandardOutput $logOutPath `
            -RedirectStandardError $logErrPath `
            -WindowStyle Hidden -PassThru

        # Give the proxy 4 seconds to spin up on Windows
        Start-Sleep -Seconds 4
    } else {
        Write-Host "LiteLLM proxy is already active on port $port. Reusing running server." -ForegroundColor Gray
    }

    # 3. Set routing environment variables pointing to proxy port
    $env:ANTHROPIC_BASE_URL = "http://127.0.0.1:$port"
    $env:ANTHROPIC_AUTH_TOKEN = "sk-dummy"
    $env:ANTHROPIC_MODEL = $Model

    # Timeouts and stability fixes
    $env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = "1"
    $env:DISABLE_NON_ESSENTIAL_MODEL_CALLS = "1"
    $env:CLAUDE_CODE_CONNECT_TIMEOUT_MS = "0"
    $env:API_TIMEOUT_MS = "1200000"
    $env:CLAUDE_CODE_SKIP_MODEL_CHECK = "1"

    # --- CRITICAL BASH SWITCHES ---
    # Disable PowerShell primary tool rollout
    $env:CLAUDE_CODE_USE_POWERSHELL_TOOL = "0" 

    Write-Host "Launching Claude Code via Git Bash using model: $Model (Proxy Port: $port)" -ForegroundColor Green

    # 4. Execute claude CLI (server stays running in background)
    & claude.exe $RemainingArgs
}
