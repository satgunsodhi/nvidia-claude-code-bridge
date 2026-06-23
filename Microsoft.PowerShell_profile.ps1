function claude {
    [CmdletBinding(PositionalBinding=$false)]
    param(
        [Parameter(Mandatory=$false)]
        [string]$Model = "nvidia-agent", # Default model if none specified
        
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
    
    # Dynamically inject the model chosen at runtime
    $env:ANTHROPIC_MODEL = $Model

    # Timeouts and stability fixes
    $env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = "1"
    $env:DISABLE_NON_ESSENTIAL_MODEL_CALLS = "1"
    $env:CLAUDE_CODE_CONNECT_TIMEOUT_MS = "0"
    $env:API_TIMEOUT_MS = "1200000"
    $env:CLAUDE_CODE_USE_POWERSHELL_TOOL=1

    # Give the proxy 5 seconds to spin up on Windows
    Start-Sleep -Seconds 5

    Write-Host "Launching Claude Code using model: $Model" -ForegroundColor Green

    # 4. Run the actual Claude executable, passing remaining flags (like --debug)
    & claude.exe $RemainingArgs

    # 5. Clean up background proxy on exit
    if ($proxyProcess) {
        Stop-Process -Id $proxyProcess.Id -Force -ErrorAction SilentlyContinue
    }
}