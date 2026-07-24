# NVIDIA Claude Code Bridge

Use NVIDIA NIM models behind Claude Code by running LiteLLM as an Anthropic-compatible proxy.

This repository contains a ready-to-edit LiteLLM configuration that maps Claude/Anthropic model names requested by Claude Code to NVIDIA NIM model groups, with retries, fallbacks, context-window fallbacks, and a small custom callback for safer request handling.

## What is included

- `litellm_config.yaml` - LiteLLM proxy configuration for NVIDIA NIM.
- `custom_callbacks.py` - LiteLLM callback that clamps overly large `max_tokens` requests to `4096` and logs rate-limit details.
- `README.md` - setup and usage notes for this bridge.

## How it works

Claude Code sends requests for Anthropic model names such as `claude-3-5-sonnet-20241022` or `claude-3-5-haiku-20241022`.

LiteLLM receives those requests and uses `router_settings.model_group_alias` to route them to NVIDIA-backed model groups:

- Sonnet-style requests route to `nvidia-agent`.
- Haiku-style requests route to `nvidia-fast-agent`.
- Fallback models are used when the primary model is unavailable or the request exceeds the primary context window.

The primary multimodal models are configured with 32k token context windows. Text fallback models provide larger context windows where available.

## Requirements

- Python 3.10 or newer
- LiteLLM
- An NVIDIA API key with access to the configured NIM models
- Claude Code configured to use a custom Anthropic-compatible base URL

## Setup

Install LiteLLM:

```powershell
pip install "litellm[proxy]"
```

Set your NVIDIA API key:

```powershell
$env:NVIDIA_API_KEY = "your-nvidia-api-key"
```

Start the LiteLLM proxy from the repository root:

```powershell
litellm --config .\litellm_config.yaml --port 4000
```

The proxy will listen at:

```text
http://localhost:4000
```

## Claude Code configuration

Point Claude Code at the local LiteLLM proxy using Anthropic-compatible settings.

For a PowerShell session:

```powershell
$env:ANTHROPIC_BASE_URL = "http://localhost:4000"
$env:ANTHROPIC_API_KEY = "anything"
claude
```

`ANTHROPIC_API_KEY` is still required by Anthropic-compatible clients, but LiteLLM uses `NVIDIA_API_KEY` from `litellm_config.yaml` when calling NVIDIA NIM.

## Model routing

The config currently aliases these Claude model families:

| Claude Code request | LiteLLM model group |
| --- | --- |
| `claude-3-5-sonnet-*` | `nvidia-agent` |
| `claude-3-sonnet-*` | `nvidia-agent` |
| `claude-sonnet-4-6` | `nvidia-agent` |
| `anthropic_sonnet` | `nvidia-agent` |
| `claude-3-5-haiku-*` | `nvidia-fast-agent` |
| `claude-haiku-4-6` | `nvidia-fast-agent` |
| `anthropic_haiku` | `nvidia-fast-agent` |

Primary model groups:

- `nvidia-opus-agent` -> `nvidia_nim/deepseek-ai/deepseek-v4-pro`
- `nvidia-agent` -> `nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b`
- `nvidia-fast-agent` -> `nvidia_nim/stepfun-ai/step-3.7-flash`

The configuration includes all 118 models freely available on the NVIDIA NIM catalog, covering DeepSeek, Llama 3.3/3.2, Mistral, Qwen, Gemma, Phi, Minimax, StepFun, GLM 5.2, and specialized vision, safety, and embedding models. Fallback groups automatically route requests across equivalent classes if a primary model is unavailable.

## Custom callback behavior

`custom_callbacks.py` registers `proxy_handler_instance` with LiteLLM.

It currently does two things:

- Clamps `max_tokens` values above `4096` before the request is sent.
- Prints detailed failure information, especially for `429` rate-limit responses.

This is useful when Claude Code or another client asks for a large completion budget that would make NVIDIA NIM requests fail or behave unpredictably.

## Troubleshooting

If the proxy fails to start, check that:

- `NVIDIA_API_KEY` is set in the same shell where you start LiteLLM.
- LiteLLM can import `custom_callbacks.py` from the repository root.
- The model names in `litellm_config.yaml` are available for your NVIDIA account.

If Claude Code cannot connect, check that:

- LiteLLM is still running on `http://localhost:4000`.
- `ANTHROPIC_BASE_URL` points to the LiteLLM proxy.
- Your terminal session has the expected environment variables.

If requests fail with rate limits, inspect the callback output in the LiteLLM terminal. It prints status codes, exception details, response headers, and retry-related fields when available.

## Editing the bridge

To change routing, update `router_settings.model_group_alias` in `litellm_config.yaml`.

To add or replace NVIDIA models, edit `model_list` and keep the corresponding fallback groups in sync.
