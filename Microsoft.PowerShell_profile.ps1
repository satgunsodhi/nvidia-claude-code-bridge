function claude {
    [CmdletBinding(PositionalBinding=$false)]
    param(
        [Parameter(Mandatory=$false)]
        [string]$Model = "nvidia-opus-agent", 
        
        [Parameter(Mandatory=$false, ValueFromRemainingArguments=$true)]
        [string[]]$RemainingArgs
    )

    # 1. Define paths
    $yamlPath = "C:\Users\Satgu\Documents\VS Code\nvidia-claude-code-bridge\litellm_config.yaml"
    $venvDir = "C:\Users\Satgu\Documents\VS Code\nvidia-claude-code-bridge\.venv"
    $litellmExe = "$venvDir\Scripts\litellm.exe"

    # 2. Start LiteLLM with explicit WorkingDirectory and log redirection
    $logOutPath = "C:\Users\Satgu\Documents\VS Code\nvidia-claude-code-bridge\litellm_out.log"
    $logErrPath = "C:\Users\Satgu\Documents\VS Code\nvidia-claude-code-bridge\litellm_err.log"
    $proxyProcess = Start-Process -FilePath $litellmExe `
        -ArgumentList "--config `"$yamlPath`" --detailed_debug" `
        -WorkingDirectory "C:\Users\Satgu\Documents\VS Code\nvidia-claude-code-bridge" `
        -RedirectStandardOutput $logOutPath `
        -RedirectStandardError $logErrPath `
        -WindowStyle Hidden -PassThru

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

    # Force Claude to find and run via Git Bash executable (if not already on path)
    # $env:CLAUDE_CODE_GIT_BASH_PATH = "C:\Program Files\Git\bin\bash.exe"

    # Give the proxy 5 seconds to spin up on Windows
    Start-Sleep -Seconds 5

    Write-Host "Launching Claude Code via Git Bash using model: $Model" -ForegroundColor Green

    # 4. Execute inside Try/Finally block to ensure proxy cleanup on Ctrl+C
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
