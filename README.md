# Claude Code Bridge (Kaggle Model Proxy & NVIDIA NIM)

Use Kaggle Model Proxy ($10/day quota) and NVIDIA NIM models behind Claude Code by running LiteLLM as an Anthropic-compatible proxy.

This repository provides a complete proxy setup that maps Claude/Anthropic model names requested by Claude Code to upstream model groups with **automated short-lived credential rotation**, multi-tier fallbacks, and custom safety callbacks.

## What is included

- `kaggle_auth.py` - Dynamic authentication manager that automatically refreshes Kaggle's short-lived (~1 hour) Model Proxy API keys before expiration.
- `litellm_config.yaml` - LiteLLM proxy configuration with Anthropic model aliases, Kaggle and NVIDIA NIM model definitions, and multi-tier fallback chains.
- `custom_callbacks.py` - LiteLLM callback for dynamic Kaggle token injection, 401/403 recovery, `max_tokens` clamping, and rate limit logging.
- `Microsoft.PowerShell_profile.ps1` - PowerShell automation function to start the proxy and launch Claude Code seamlessly.
- `README.md` - Setup and usage guide.

## How it works

Claude Code sends requests for Anthropic model names such as `claude-3-5-sonnet-20241022` or `claude-3-opus-20240229`.

LiteLLM receives those requests and routes them to primary Kaggle model groups:

| Claude Code Request | Primary Model Group | Upstream Model (Kaggle) | Fallback Models |
| :--- | :--- | :--- | :--- |
| `claude-3-opus-*` | `kaggle-opus-agent` | `deepseek-ai/deepseek-r1-0528` | `openai/gpt-5.4-nano`, `nvidia-opus-agent` (DeepSeek v4 Pro), `kimi-k2.6` |
| `claude-3-5-sonnet-*` / `claude-sonnet-*` | `kaggle-agent` | `anthropic/claude-sonnet-5@default` | `google/gemini-3-flash-preview`, `nvidia-agent` (Nemotron 550B), `mistral-large-2` |
| `claude-3-5-haiku-*` / `claude-haiku-*` | `kaggle-fast-agent` | `google/gemini-3-flash-preview` | `google/gemini-3.1-flash-lite-preview`, `nvidia-fast-agent` (Step 3.7 Flash), `llama-3.1-8b` |

If a Kaggle model hits a rate limit (429) or quota restriction, LiteLLM automatically fails over across other authorized Kaggle models and NVIDIA NIM models.

## Requirements

- Python 3.10+
- Kaggle CLI (`kaggle>=2.0.0`)
- Kaggle API token (in `~/.kaggle/access_token` or `~/.kaggle/kaggle.json`)
- NVIDIA API key (for NVIDIA NIM fallbacks)
- Claude Code CLI

## Setup

1. **Activate the Python virtual environment**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Verify Kaggle Authentication**:
   ```powershell
   python kaggle_auth.py --status
   ```
   If needed, force an immediate token refresh:
   ```powershell
   python kaggle_auth.py --refresh
   ```

3. **Start the LiteLLM Proxy manually**:
   ```powershell
   litellm --config .\litellm_config.yaml --port 4000
   ```

4. **Launch Claude Code**:
   ```powershell
   $env:ANTHROPIC_BASE_URL = "http://127.0.0.1:4000"
   $env:ANTHROPIC_API_KEY = "sk-dummy"
   claude
   ```

## PowerShell Profile Integration

You can copy the `claude` function from `Microsoft.PowerShell_profile.ps1` into your `$PROFILE`.
Then, simply run:

```powershell
# Launch Claude Code with default Kaggle Sonnet 5
claude

# Or choose a specific agent/model:
claude -Model kaggle-opus-agent
claude -Model nvidia-opus-agent
```

## Dynamic Authentication Details

Kaggle Model Proxy tokens expire after 1 hour. `kaggle_auth.py` and `custom_callbacks.py` manage this automatically:
- **Pre-Call Check**: Before each request, `async_pre_call_hook` verifies if the current token has at least 5 minutes remaining. If expired or expiring, it runs `kaggle benchmarks auth` in the background and updates the in-memory authorization header.
- **401/403 Recovery**: If upstream responds with 401 or 403, the callback invalidates the cached key and force-refreshes immediately.
- **No Proxy Restart Required**: Tokens are injected dynamically on every request without requiring LiteLLM to be restarted.
