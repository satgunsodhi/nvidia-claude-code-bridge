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

    # Load environment variables from .env and .kaggle_proxy.env before starting LiteLLM
    foreach ($ef in @("$bridgeDir\.env", "$bridgeDir\.kaggle_proxy.env")) {
        if (Test-Path $ef) {
            Get-Content $ef | ForEach-Object {
                if ($_ -match '^\s*([A-Za-z0-9_]+)\s*=\s*"?([^"#]+)"?') {
                    [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim())
                }
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

    # Resolve model native context window and output limits dynamically from litellm config
    $maxInputTokens = 200000
    $maxOutputTokens = 8192
    $pyCmd = 'import sys, litellm, custom_callbacks, json; info = litellm.get_model_info(sys.argv[1]); print(json.dumps(info))'
    $rawSpecs = & $pythonExe -c $pyCmd $Model 2>$null
    $jsonStr = ($rawSpecs | Where-Object { $_ -match '^\{.*\}$' }) | Select-Object -Last 1
    if ($jsonStr) {
        try {
            $parsedInfo = $jsonStr | ConvertFrom-Json
            if ($parsedInfo.max_input_tokens) { $maxInputTokens = [int]$parsedInfo.max_input_tokens }
            if ($parsedInfo.max_output_tokens) { $maxOutputTokens = [int]$parsedInfo.max_output_tokens }
        } catch {}
    }

    # Set auto-compact token limits and context window enforcement for Claude Code CLI based on selected model's native capacity
    $autoCompactLimit = [math]::Floor($maxInputTokens * 0.90)
    
    [System.Environment]::SetEnvironmentVariable("CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT", "1", "Process")
    [System.Environment]::SetEnvironmentVariable("CLAUDE_CODE_MAX_CONTEXT_TOKENS", "$maxInputTokens", "Process")
    [System.Environment]::SetEnvironmentVariable("CLAUDE_CODE_AUTO_COMPACT_WINDOW", "$maxInputTokens", "Process")
    [System.Environment]::SetEnvironmentVariable("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE", "90", "Process")
    [System.Environment]::SetEnvironmentVariable("CLAUDE_AUTO_COMPACT_TOKEN_LIMIT", "$autoCompactLimit", "Process")
    [System.Environment]::SetEnvironmentVariable("CLAUDE_CODE_AUTO_COMPACT_TOKEN_LIMIT", "$autoCompactLimit", "Process")
    [System.Environment]::SetEnvironmentVariable("CLAUDE_CODE_WORKFLOW_SIZE_WARNING_TOKENS", "$autoCompactLimit", "Process")
    [System.Environment]::SetEnvironmentVariable("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "$maxOutputTokens", "Process")

    $env:CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT = "1"
    $env:CLAUDE_CODE_MAX_CONTEXT_TOKENS = "$maxInputTokens"
    $env:CLAUDE_CODE_AUTO_COMPACT_WINDOW = "$maxInputTokens"
    $env:CLAUDE_AUTOCOMPACT_PCT_OVERRIDE = "90"
    $env:CLAUDE_AUTO_COMPACT_TOKEN_LIMIT = "$autoCompactLimit"
    $env:CLAUDE_CODE_AUTO_COMPACT_TOKEN_LIMIT = "$autoCompactLimit"
    $env:CLAUDE_CODE_WORKFLOW_SIZE_WARNING_TOKENS = "$autoCompactLimit"
    $env:CLAUDE_CODE_MAX_OUTPUT_TOKENS = "$maxOutputTokens"

    # Timeouts and stability fixes
    $env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = "1"
    $env:DISABLE_NON_ESSENTIAL_MODEL_CALLS = "1"
    $env:CLAUDE_CODE_CONNECT_TIMEOUT_MS = "0"
    $env:API_TIMEOUT_MS = "1200000"
    $env:CLAUDE_CODE_SKIP_MODEL_CHECK = "1"

    # --- CRITICAL BASH SWITCHES ---
    # Disable PowerShell primary tool rollout
    $env:CLAUDE_CODE_USE_POWERSHELL_TOOL = "0" 

    # If model is 1M capacity, append [1m] tag if not present so Claude Code recognizes 1M context
    $effectiveModel = $Model
    if ($maxInputTokens -ge 1000000 -and $Model -notlike "*[1m]*") {
        $effectiveModel = "${Model}[1m]"
    }

    Write-Host "Launching Claude Code via Git Bash using model: $effectiveModel (Proxy Port: $port | Context: ${maxInputTokens} tokens | AutoCompact Limit: ${autoCompactLimit} tokens)" -ForegroundColor Green

    # 4. Execute claude CLI with --model flag
    $cliArgs = @()
    if ($effectiveModel -and ($RemainingArgs -notcontains "--model")) {
        $cliArgs += @("--model", $effectiveModel)
    }
    if ($RemainingArgs) {
        $cliArgs += $RemainingArgs
    }

    & claude.exe @cliArgs
}

