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

    # 2. Check if LiteLLM proxy is already running on port 4000
    $logOutPath = "$bridgeDir\litellm_out.log"
    $logErrPath = "$bridgeDir\litellm_err.log"
    $portActive = Get-NetTCPConnection -LocalPort 4000 -State Listen -ErrorAction SilentlyContinue

    $proxyProcess = $null
    if (-not $portActive) {
        Write-Host "Starting LiteLLM proxy background process on port 4000..." -ForegroundColor Gray
        $proxyProcess = Start-Process -FilePath $litellmExe `
            -ArgumentList "--config `"$yamlPath`" --port 4000 --detailed_debug" `
            -WorkingDirectory $bridgeDir `
            -RedirectStandardOutput $logOutPath `
            -RedirectStandardError $logErrPath `
            -WindowStyle Hidden -PassThru

        # Give the proxy 5 seconds to spin up on Windows
        Start-Sleep -Seconds 5
    } else {
        Write-Host "LiteLLM proxy is already running on port 4000. Reusing active server." -ForegroundColor Gray
    }

    # 3. Set routing environment variables
    $env:ANTHROPIC_BASE_URL = "http://127.0.0.1:4000"
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

    Write-Host "Launching Claude Code via Git Bash using model: $Model" -ForegroundColor Green

    # 4. Execute inside Try/Finally block to ensure proxy cleanup if started by this session
    try {
        & claude.exe $RemainingArgs
    }
    finally {
        if ($proxyProcess) {
            Write-Host "Stopping LiteLLM proxy background process..." -ForegroundColor Gray
            Stop-Process -Id $proxyProcess.Id -Force -ErrorAction SilentlyContinue
        }
    }
}
