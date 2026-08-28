import litellm
from litellm.integrations.custom_logger import CustomLogger
import sys
import json
import asyncio
import time
import threading
import os
from collections import deque
from kaggle_auth import kaggle_auth_manager

import re

DEFAULT_CONTEXT_WINDOW = 200000
DEFAULT_MAX_OUTPUT_TOKENS = 8192

MODEL_CONTEXT_WINDOWS = {
    # Gemini 1.5/2.0/3.0/3.1 series: 1M - 2M tokens
    "google/gemini-3-flash-preview": 1000000,
    "google/gemini-3.1-flash-lite-preview": 1000000,
    "gemini-1.5-pro": 2000000,
    "gemini-1.5-flash": 1000000,
    "gemini-2.0-flash": 1000000,
    "google/gemma-3-12b-it": 128000,
    "google/gemma-3n-e4b-it": 128000,
    "google/gemma-3n-e2b-it": 128000,
    "google/gemma-4-31b-it": 128000,
    "google/gemma-2-2b-it": 8192,
    "google/diffusiongemma-26b-a4b-it": 32768,

    # Claude 3/3.5/3.7/Sonnet 5 series: 200k tokens
    "anthropic/claude-sonnet-5@default": 200000,
    "claude-3-5-sonnet": 200000,
    "claude-3-7-sonnet": 200000,
    "claude-3-opus": 200000,
    "claude-3-haiku": 200000,
    "claude-3-5-haiku": 200000,

    # DeepSeek R1 & v3/v4 series: 128k - 256k tokens
    "deepseek-ai/deepseek-r1-0528": 128000,
    "deepseek-ai/deepseek-r1": 128000,
    "deepseek-ai/deepseek-v3": 128000,
    "deepseek-ai/deepseek-v4-pro-0813": 256000,
    "deepseek-ai/deepseek-v4-flash-0731": 128000,
    "moonshotai/kimi-k3": 200000,
    "meta/muse-glimmer-30b": 128000,
    "meta/llama-3.1-405b-instruct": 128000,
    "meta/llama-3.1-70b-instruct": 128000,
    "meta/llama-3.1-8b-instruct": 128000,
    "mistralai/mistral-large-2-instruct": 128000,
    "mistralai/mixtral-8x22b-v0.1": 64000,
    "nv-mistralai/mistral-nemo-12b-instruct": 128000,
    "nvidia/nemotron-4-340b-instruct": 128000,
    "nvidia/llama-3.1-nemotron-70b-instruct": 128000,
    "nvidia/nemotron-3.5-lightning-30b-a3b": 128000,
    "nvidia/nemotron-3-super-120b-a12b": 128000,
    "nvidia/nemotron-3-ultra-550b-a55b": 32768,
    "nvidia/llama-3.3-nemotron-super-49b-v1": 128000,
    "nvidia/cosmos-reason2-8b": 32768,
    "nvidia/nemotron-mini-4b-instruct": 4096,

    # Primary Agents
    "kaggle-fast-agent": 1000000,   # Gemini 3 Flash backing -> 1M context
    "kaggle-agent": 200000,        # Claude Sonnet 5 backing -> 200k context
    "kaggle-opus-agent": 128000,    # DeepSeek R1 backing -> 128k context
    "nvidia-opus-agent": 256000,    # DeepSeek v4 Pro backing -> 256k context
    "nvidia-agent": 128000,         # Nemotron Super 49B / 550B fallback -> 128k context
    "nvidia-fast-agent": 256000,    # Step 3.7 Flash backing -> 256k context
}

MODEL_OUTPUT_TOKENS = {
    "claude-3-7-sonnet-20240219": 64000,
    "claude-3-7-sonnet-latest": 64000,
    "claude-3-5-sonnet-20241022": 8192,
    "claude-3-5-sonnet-latest": 8192,
    "claude-sonnet-5": 8192,
    "anthropic/claude-sonnet-5@default": 8192,
    "claude-3-opus-20240229": 4096,
    "claude-3-opus-latest": 4096,
    "claude-3-haiku-20240307": 4096,
    "claude-3-5-haiku-20241022": 8192,
    "deepseek-ai/deepseek-v4-pro": 16384,
    "stepfun-ai/step-3.7-flash": 16384,
    "stepfun-ai/step-3.5-flash": 16384,
    "nvidia-opus-agent": 16384,
    "nvidia-fast-agent": 16384,
    "moonshotai/kimi-k2.6": 4096,
    "nvidia/nemotron-mini-4b-instruct": 4096,
    "google/gemma-2-2b-it": 4096,
}

def resolve_context_window(model_name: str) -> int:
    if not model_name:
        return DEFAULT_CONTEXT_WINDOW
    if model_name in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[model_name]
    clean = re.sub(r'^(openai/|nvidia_nim/|anthropic/|google/|deepseek/|mistral/|meta/)', '', model_name)
    if clean in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[clean]
    if model_name in litellm.model_cost and litellm.model_cost[model_name].get("max_input_tokens"):
        return litellm.model_cost[model_name]["max_input_tokens"]
    if clean in litellm.model_cost and litellm.model_cost[clean].get("max_input_tokens"):
        return litellm.model_cost[clean]["max_input_tokens"]
    m_lower = model_name.lower()
    if "gemini" in m_lower or "minimax" in m_lower:
        return 1000000
    if "step-3" in m_lower or "deepseek-v4" in m_lower:
        return 256000
    if any(k in m_lower for k in ["llama-3", "qwen", "deepseek", "mistral", "gemma-3", "gemma-4"]):
        return 128000
    return DEFAULT_CONTEXT_WINDOW

def resolve_max_output_tokens(model_name: str) -> int:
    if not model_name:
        return DEFAULT_MAX_OUTPUT_TOKENS
    if model_name in MODEL_OUTPUT_TOKENS:
        return MODEL_OUTPUT_TOKENS[model_name]
    clean = re.sub(r'^(openai/|nvidia_nim/|anthropic/|google/|deepseek/|mistral/|meta/)', '', model_name)
    if clean in MODEL_OUTPUT_TOKENS:
        return MODEL_OUTPUT_TOKENS[clean]
    if model_name in litellm.model_cost and litellm.model_cost[model_name].get("max_output_tokens"):
        return litellm.model_cost[model_name]["max_output_tokens"]
    if clean in litellm.model_cost and litellm.model_cost[clean].get("max_output_tokens"):
        return litellm.model_cost[clean]["max_output_tokens"]
    m_lower = model_name.lower()
    if "claude-3-7-sonnet" in m_lower or "sonnet-3-7" in m_lower:
        return 64000
    if "step-3" in m_lower or "deepseek-v4" in m_lower:
        return 16384
    if re.search(r'-(mini|micro|nano)\b', m_lower) or any(k in m_lower for k in ["-2b-", "-1b-", "-3b-", "opus-20240229"]):
        return 4096
    return DEFAULT_MAX_OUTPUT_TOKENS

def ensure_model_context_window(model_name: str):
    if not model_name:
        return
    max_in = resolve_context_window(model_name)
    max_out = resolve_max_output_tokens(model_name)
    if model_name not in litellm.model_cost:
        litellm.model_cost[model_name] = {
            "max_input_tokens": max_in,
            "max_output_tokens": max_out,
            "max_tokens": max_out,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "litellm_provider": "openai",
            "mode": "chat"
        }
    else:
        litellm.model_cost[model_name]["max_input_tokens"] = max_in
        litellm.model_cost[model_name]["max_output_tokens"] = max_out
        litellm.model_cost[model_name]["max_tokens"] = max_out


# Pre-populate litellm.model_cost for common agents, aliases, and catalog models
for name in [
    "kaggle-agent", "kaggle-opus-agent", "kaggle-fast-agent",
    "nvidia-agent", "nvidia-opus-agent", "nvidia-fast-agent",
    "claude-3-5-sonnet-20241022", "claude-3-5-sonnet-latest", "claude-3-5-sonnet-20240620",
    "claude-3-sonnet-20240229", "claude-sonnet-4-6", "claude-sonnet-5", "anthropic_sonnet",
    "claude-3-opus-20240229", "claude-3-opus-latest", "claude-opus-4-6", "anthropic_opus",
    "claude-3-5-haiku-20241022", "claude-3-5-haiku-latest", "claude-3-haiku-20240307",
    "claude-haiku-4-6", "anthropic_haiku", "claude-3-7-sonnet-20240219", "claude-3-7-sonnet-latest",
    "google/gemini-3-flash-preview", "google/gemini-3.1-flash-lite-preview",
    "deepseek-ai/deepseek-r1-0528", "deepseek-ai/deepseek-v4-pro", "stepfun-ai/step-3.7-flash"
]:
    ensure_model_context_window(name)

try:
    config_path = os.path.join(os.path.dirname(__file__), "litellm_config.yaml")
    if os.path.exists(config_path):
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            _cfg = yaml.safe_load(f)
            for _m in _cfg.get("model_list", []):
                _name = _m.get("model_name")
                if _name:
                    ensure_model_context_window(_name)
except Exception:
    pass


class AsyncRateLimiter:
    def __init__(self, rpm: int = 40):
        self.rpm = rpm
        self.lock = asyncio.Lock()
        self.history = deque()

    async def acquire(self, model_name: str):
        async with self.lock:
            while True:
                now = time.time()
                # Clear history older than 60 seconds
                while self.history and self.history[0] < now - 60:
                    self.history.popleft()
                
                if len(self.history) < self.rpm:
                    self.history.append(now)
                    return
                
                # Calculate sleep time
                sleep_time = self.history[0] + 60.1 - now
                if sleep_time > 0:
                    print(f"\n\033[1;33m[RateLimiter] Rate limit (40 RPM) reached for '{model_name}'. Sleeping for {sleep_time:.2f} seconds to prevent failure...\033[0m")
                    sys.stdout.flush()
                    await asyncio.sleep(sleep_time)

class ModelRateLimiter:
    def __init__(self, rpm: int = 40):
        self.rpm = rpm
        self.limiters = {}
        self.global_lock = asyncio.Lock()

    async def acquire(self, model_name: str):
        async with self.global_lock:
            if model_name not in self.limiters:
                self.limiters[model_name] = AsyncRateLimiter(self.rpm)
            limiter = self.limiters[model_name]
        
        await limiter.acquire(model_name)

# Instantiate the rate limiter
rate_limiter = ModelRateLimiter(rpm=40)

def is_vision_model(model_name: str) -> bool:
    if not model_name:
        return False
    model_lower = model_name.lower()
    vision_keywords = ["vision", "-vl", "paligemma", "multimodal"]
    return any(kw in model_lower for kw in vision_keywords)

class CustomRateLimitLogger(CustomLogger):
    def __init__(self):
        super().__init__()
        self.last_request_time = time.time()
        self._start_inactivity_checker()

    def touch_activity(self):
        self.last_request_time = time.time()

    def _start_inactivity_checker(self):
        def _checker():
            while True:
                time.sleep(5)
                try:
                    timeout_str = os.getenv("PROXY_INACTIVITY_TIMEOUT", "300").strip()
                    timeout = float(timeout_str) if timeout_str else 300.0
                    if timeout <= 0:
                        continue
                    
                    idle_seconds = time.time() - self.last_request_time
                    if idle_seconds >= timeout:
                        print(f"\n\033[1;33m[AutoShutdown] Inactivity threshold reached ({int(idle_seconds)}s >= {int(timeout)}s). Shutting down LiteLLM proxy server...\033[0m")
                        sys.stdout.flush()
                        os._exit(0)
                except Exception:
                    pass

        t = threading.Thread(target=_checker, daemon=True)
        t.start()

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type, **kwargs):
        self.touch_activity()
        try:
            model_name = data.get("model", "unknown")
            if model_name and "[1m]" in model_name:
                model_name = model_name.replace("[1m]", "")
                data["model"] = model_name

            raw_model = kwargs.get("model") or data.get("model")
            if raw_model:
                ensure_model_context_window(raw_model)
            if model_name:
                ensure_model_context_window(model_name)

            # 1. Dynamic max_tokens clamping according to model's native output capacity
            max_tokens = data.get("max_tokens")
            native_max_out = resolve_max_output_tokens(model_name)
            if max_tokens is not None and max_tokens > native_max_out:
                print(f"\n[CustomLogger] Clamping max_tokens for '{model_name}' from {max_tokens} to native limit {native_max_out}")
                data["max_tokens"] = native_max_out
                sys.stdout.flush()

            # --- Dynamic Provider Selection (Configurable via .env PRIMARY_PROVIDER) ---
            primary_provider = os.getenv("PRIMARY_PROVIDER", "kaggle").strip().lower()
            if primary_provider == "nvidia":
                if model_name in ("kaggle-agent", "claude-3-5-sonnet-20241022", "claude-3-5-sonnet-latest", "claude-3-5-sonnet-20240620", "claude-3-sonnet-20240229", "claude-sonnet-4-6", "claude-sonnet-5", "anthropic_sonnet"):
                    data["model"] = "nvidia-agent"
                    model_name = "nvidia-agent"
                elif model_name in ("kaggle-opus-agent", "claude-3-opus-20240229", "claude-3-opus-latest", "claude-opus-4-6", "anthropic_opus"):
                    data["model"] = "nvidia-opus-agent"
                    model_name = "nvidia-opus-agent"
                elif model_name in ("kaggle-fast-agent", "claude-3-5-haiku-20241022", "claude-3-5-haiku-latest", "claude-3-haiku-20240307", "claude-haiku-4-6", "anthropic_haiku"):
                    data["model"] = "nvidia-fast-agent"
                    model_name = "nvidia-fast-agent"

            # --- Message Role & Content Sanitization (Fixing "unexpected role: " 400 errors) ---
            messages = data.get("messages", [])
            if messages and isinstance(messages, list):
                valid_non_system_roles = {"user", "assistant", "tool", "function"}
                system_prompts = []
                non_system_messages = []

                for msg in messages:
                    if not isinstance(msg, dict):
                        if hasattr(msg, "model_dump"):
                            msg = msg.model_dump(exclude_unset=True)
                        elif hasattr(msg, "dict"):
                            msg = msg.dict(exclude_unset=True)
                        else:
                            try:
                                msg = dict(msg)
                            except Exception:
                                continue
                    
                    role = msg.get("role")
                    
                    # Normalize developer role (OpenAI format) to system
                    if role == "developer":
                        role = "system"
                    
                    # Extract system prompts from anywhere in the message list
                    if role == "system":
                        content = msg.get("content")
                        if content:
                            if isinstance(content, str):
                                system_prompts.append(content)
                            elif isinstance(content, list):
                                text_parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                                if text_parts:
                                    system_prompts.append("\n".join(text_parts))
                                else:
                                    system_prompts.append(str(content))
                        continue
                    
                    content = msg.get("content")
                    # Handle Anthropic-style blocks that LiteLLM might have failed to translate
                    if isinstance(content, list):
                        if role == "assistant":
                            new_content = ""
                            tool_calls = []
                            for block in content:
                                if not isinstance(block, dict):
                                    continue
                                btype = block.get("type")
                                if btype == "text":
                                    new_content += block.get("text", "")
                                elif btype == "thinking":
                                    new_content += f"<thinking>\n{block.get('thinking', '')}\n</thinking>\n"
                                elif btype == "tool_use":
                                    tool_calls.append({
                                        "id": block.get("id"),
                                        "type": "function",
                                        "function": {
                                            "name": block.get("name"),
                                            "arguments": json.dumps(block.get("input", {})) if isinstance(block.get("input"), dict) else str(block.get("input", "{}"))
                                        }
                                    })
                                elif btype == "image_url":
                                    pass # Handled normally if we just keep the list, but since we modify it, we might break multimodal.
                            
                            # Only overwrite if we actually found Anthropic specific blocks
                            has_anthropic_blocks = any(isinstance(b, dict) and b.get("type") in ("thinking", "tool_use") for b in content)
                            if has_anthropic_blocks:
                                msg["content"] = new_content.strip() or None
                                if tool_calls:
                                    msg["tool_calls"] = msg.get("tool_calls", []) + tool_calls
                        
                        elif role == "user":
                            has_tool_result = any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
                            if has_tool_result:
                                text_parts = []
                                tool_msgs = []
                                for block in content:
                                    if not isinstance(block, dict):
                                        continue
                                    btype = block.get("type")
                                    if btype == "text":
                                        text_parts.append(block.get("text", ""))
                                    elif btype == "tool_result":
                                        tm_content = block.get("content", "")
                                        if isinstance(tm_content, list):
                                            # extract text from nested content
                                            tm_content = "".join([b.get("text", "") for b in tm_content if isinstance(b, dict) and b.get("type") == "text"])
                                        tool_msgs.append({
                                            "role": "tool",
                                            "tool_call_id": block.get("tool_use_id"),
                                            "content": str(tm_content)
                                        })
                                
                                # Add the text part as a user message if it exists
                                if text_parts:
                                    msg["content"] = "\n".join(text_parts)
                                    msg["role"] = "user"
                                    non_system_messages.append(msg)
                                
                                # Add the tool messages
                                for tm in tool_msgs:
                                    non_system_messages.append(tm)
                                continue # Skip the default append below

                    # Fix invalid, empty, missing, or unsupported roles for non-system messages
                    role = msg.get("role")
                    if not role or not isinstance(role, str) or role not in valid_non_system_roles:
                        role = "user"
                        msg["role"] = role
                    
                    # Ensure content is non-empty
                    content = msg.get("content")
                    if content is None or content == "" or (isinstance(content, list) and len(content) == 0):
                        if not msg.get("tool_calls"):
                            msg["content"] = " "
                    
                    non_system_messages.append(msg)
                
                cleaned_messages = list(non_system_messages)
                
                # Consolidate system prompts into top-level system/instructions and prepend to first user message
                if system_prompts:
                    combined_system_text = "\n\n".join(system_prompts)
                    data["system"] = combined_system_text
                    data["instructions"] = combined_system_text
                    if "litellm_params" in data and isinstance(data["litellm_params"], dict):
                        data["litellm_params"]["system"] = combined_system_text
                        data["litellm_params"]["instructions"] = combined_system_text
                    
                    # Prepend system prompt context to the first user message
                    if cleaned_messages:
                        first_msg = cleaned_messages[0]
                        if first_msg.get("role") == "user":
                            orig_content = first_msg.get("content", "")
                            sys_prefix = f"<system_instructions>\n{combined_system_text}\n</system_instructions>\n\n"
                            if isinstance(orig_content, str):
                                first_msg["content"] = sys_prefix + orig_content
                            elif isinstance(orig_content, list):
                                first_msg["content"] = [{"type": "text", "text": sys_prefix}] + orig_content
                        else:
                            cleaned_messages.insert(0, {"role": "user", "content": f"<system_instructions>\n{combined_system_text}\n</system_instructions>"})
                    else:
                        cleaned_messages.append({"role": "user", "content": f"<system_instructions>\n{combined_system_text}\n</system_instructions>"})
                
                data["messages"] = cleaned_messages
                if "litellm_params" in data and isinstance(data["litellm_params"], dict):
                    data["litellm_params"]["messages"] = cleaned_messages
                
                import json
                try:
                    with open(r"c:\Users\Satgu\Documents\VS Code\nvidia-claude-code-bridge\kaggle_payload.txt", "w") as f:
                        for idx, m in enumerate(data['messages']):
                            # Convert m to dict if needed
                            if hasattr(m, 'model_dump'):
                                m = m.model_dump()
                            elif hasattr(m, 'dict'):
                                m = m.dict()
                            elif not isinstance(m, dict):
                                m = dict(m)
                            f.write(f"Message {idx}: Role='{m.get('role')}' | Content={str(m.get('content'))[:100]}\n")
                except Exception as e:
                    pass


            # --- Target Provider Auth Injection ---
            api_base = data.get("api_base") or (data.get("litellm_params", {}).get("api_base") if isinstance(data.get("litellm_params"), dict) else "") or ""
            
            # List of model prefixes or exact names known to be hosted on NVIDIA NIM
            nvidia_catalog_prefixes = (
                "nvidia", "nvidia_nim", "nv-mistralai/", "mistralai/", "meta/",
                "google/gemma", "google/codegemma", "google/deplot", "google/diffusiongemma", "google/recurrentgemma",
                "deepseek-ai/", "qwen/", "stepfun-ai/", "moonshotai/", "minimaxai/", "z-ai/",
                "abacusai/", "01-ai/", "adept/", "ai21labs/", "aisingapore/", "baai/",
                "bigcode/", "bytedance/", "databricks/", "ibm/", "microsoft/", "poolside/",
                "sarvamai/", "snowflake/", "upstage/", "writer/", "zyphra/"
            )

            is_gemini = "gemini" in model_name.lower()

            is_nvidia_target = not is_gemini and (
                any(model_name.lower().startswith(p) for p in nvidia_catalog_prefixes)
                or "nvidia_nim" in str(api_base).lower()
                or "integrate.api.nvidia.com" in str(api_base)
            )

            is_kaggle_target = is_gemini or (not is_nvidia_target and (
                model_name.startswith("kaggle")
                or "kaggle" in str(api_base).lower()
                or "mp-staging.kaggle.net" in str(api_base)
                or primary_provider == "kaggle"
            ))

            if is_kaggle_target:
                if not kaggle_auth_manager.is_token_valid():
                    kaggle_auth_manager.refresh_credentials()
                
                active_key = kaggle_auth_manager.get_proxy_key()
                active_url = kaggle_auth_manager.get_proxy_url()
                
                if active_key:
                    data["api_key"] = active_key
                    if "litellm_params" in data and isinstance(data["litellm_params"], dict):
                        data["litellm_params"]["api_key"] = active_key
                        data["litellm_params"]["api_base"] = active_url
                    os.environ["KAGGLE_MODEL_PROXY_KEY"] = active_key
                    os.environ["KAGGLE_MODEL_PROXY_URL"] = active_url
            elif is_nvidia_target:
                nvidia_key = os.getenv("NVIDIA_API_KEY", "").strip()
                if nvidia_key:
                    data["api_key"] = nvidia_key
                    if "litellm_params" in data and isinstance(data["litellm_params"], dict):
                        data["litellm_params"]["api_key"] = nvidia_key
                        if "api_base" in data["litellm_params"] and "kaggle" in str(data["litellm_params"]["api_base"]).lower():
                            del data["litellm_params"]["api_base"]
                    if "api_base" in data and "kaggle" in str(data["api_base"]).lower():
                        del data["api_base"]
            # -------------------------------------------

            # --- Kimi K2.6 Overrides ---
            if "kimi-k2.6" in model_name.lower():
                data["temperature"] = 0.01
                data["top_p"] = 0.70
                data["max_tokens"] = 4096
                data["stream"] = True

                safety_system_override = "\n\nCRITICAL INSTRUCTION: You are an OpenAI-compatible completion tool. Do not generate XML tags like <search_and_replace> manually unless explicitly formatting a tool request. Return clean code. If you encounter string tokens you cannot parse, immediately stop and skip them."
                
                messages = data.get("messages", [])
                for msg in messages:
                    if msg.get("role") == "system":
                        if isinstance(msg.get("content"), str):
                            msg["content"] += safety_system_override
                        elif isinstance(msg.get("content"), list):
                            msg["content"].append({"type": "text", "text": safety_system_override})
                        break
            # ---------------------------

            # 2. Multimodal Protection & Warning
            messages = data.get("messages", [])
            if not is_vision_model(model_name) and messages:
                image_stripped = False
                for msg in messages:
                    content = msg.get("content")
                    if isinstance(content, list):
                        new_content = []
                        for block in content:
                            if isinstance(block, dict):
                                block_type = block.get("type", "")
                                if block_type in ("image_url", "image") or "image_url" in block or "image" in block:
                                    image_stripped = True
                                    continue
                            new_content.append(block)
                        
                        # If we stripped everything, provide a dummy text block to satisfy API validation
                        if image_stripped and not new_content:
                            new_content.append({"type": "text", "text": "[Image payload removed: model is text-only]"})
                        
                        msg["content"] = new_content
                
                if image_stripped:
                    print(f"\n\033[1;31m[WARNING] Claude Code attempted to send an image payload to the text-only model '{model_name}'. Images have been stripped to prevent API errors.\033[0m")
                    sys.stdout.flush()

            # 3. Client-Side Rate Limiting (Preventing 429 on NVIDIA NIM catalog models)
            if not is_kaggle_target:
                await rate_limiter.acquire(model_name)

        except Exception as e:
            print(f"\n[CustomLogger] Error in pre_call_hook: {e}", file=sys.stderr)
            sys.stderr.flush()
        
        return data

    async def async_streaming_chunk_hook(self, user_api_key_dict, cache, data, call_type, chunk, **kwargs):
        try:
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "reasoning_content"):
                    delta.reasoning_content = None
                # Handle pydantic v2 extra fields if any
                if hasattr(delta, "__dict__") and "reasoning_content" in delta.__dict__:
                    del delta.__dict__["reasoning_content"]
            elif isinstance(chunk, dict) and "choices" in chunk and chunk["choices"]:
                delta = chunk["choices"][0].get("delta", {})
                if "reasoning_content" in delta:
                    del delta["reasoning_content"]
        except Exception:
            pass
        return chunk


    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.touch_activity()
        try:
            model = kwargs.get("model", "unknown")
            actual_model = "unknown"
            if hasattr(response_obj, "model") and response_obj.model:
                actual_model = response_obj.model
            elif kwargs.get("litellm_params", {}).get("model"):
                actual_model = kwargs.get("litellm_params").get("model")
                
            print(f"\n[CustomLogger] SUCCESS: Mapped request '{model}' -> Actual downstream model used: '{actual_model}'")
            sys.stdout.flush()
        except Exception:
            pass

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.touch_activity()
        try:
            model = kwargs.get("model", "unknown")
            actual_model = "unknown"
            if hasattr(response_obj, "model") and response_obj.model:
                actual_model = response_obj.model
            elif kwargs.get("litellm_params", {}).get("model"):
                actual_model = kwargs.get("litellm_params").get("model")
                
            print(f"\n[CustomLogger] SUCCESS: Mapped request '{model}' -> Actual downstream model used: '{actual_model}'")
            sys.stdout.flush()
        except Exception:
            pass


    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self.touch_activity()
        try:
            self._log_failure(kwargs, response_obj)
        except Exception as e:
            print(f"[CustomLogger] Error in sync callback: {e}", file=sys.stderr)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self.touch_activity()
        try:
            self._log_failure(kwargs, response_obj)
        except Exception as e:
            print(f"[CustomLogger] Error in async callback: {e}", file=sys.stderr)

    def _log_failure(self, kwargs, response_obj):
        exception = kwargs.get("exception")
        if not exception:
            return

        status_code = getattr(exception, "status_code", None)
        if status_code is None and hasattr(exception, "response") and hasattr(exception.response, "status_code"):
            status_code = exception.response.status_code

        model = kwargs.get("model", "unknown")
        print("\n--- [CustomLogger] API Call Failure ---")
        print(f"Model: {model}")
        print(f"Status Code: {status_code}")
        print(f"Exception Type: {type(exception).__name__}")
        # Check for 401/403 Unauthorized on Kaggle to trigger immediate re-auth
        if status_code in (401, 403) and ("kaggle" in model.lower() or "mp-staging.kaggle.net" in str(kwargs)):
            print("\n\033[1;31m[KaggleAuth] 401/403 Detected: Upstream Kaggle token expired or rejected. Triggering credential refresh...\033[0m")
            try:
                kaggle_auth_manager.refresh_credentials(force=True)
            except Exception as auth_err:
                print(f"[KaggleAuth] Re-auth attempt failed: {auth_err}", file=sys.stderr)

        # Check for rate limit status (429) or rate limit strings in message
        is_rate_limit = (status_code == 429) or ("rate" in type(exception).__name__.lower()) or ("rate" in str(exception).lower())
        
        if is_rate_limit:
            print(">>> RATE LIMIT DETECTED <<<")
            
            # Print all non-private attributes of the exception object to find try again info
            print("--- Exception Object Attributes ---")
            for attr in dir(exception):
                if not attr.startswith('_'):
                    try:
                        val = getattr(exception, attr)
                        if val is not None and not callable(val):
                            if isinstance(val, dict):
                                print(f"  {attr}: {json.dumps(val)}")
                            else:
                                print(f"  {attr}: {val}")
                    except Exception:
                        pass

            # Inspect exception response headers if available
            headers = None
            if hasattr(exception, "headers") and exception.headers:
                headers = exception.headers
            elif hasattr(exception, "response") and hasattr(exception.response, "headers"):
                headers = exception.response.headers

            if headers:
                print(f"Response Headers: {dict(headers)}")
                for key, val in headers.items():
                    if "retry-after" in key.lower() or "limit" in key.lower():
                        print(f"Rate Limit Header -> {key}: {val}")
            
            # Try to get response text/content
            response_text = None
            if hasattr(exception, "response") and hasattr(exception.response, "text"):
                response_text = exception.response.text
            elif hasattr(exception, "response") and hasattr(exception.response, "content"):
                try:
                    response_text = exception.response.content.decode("utf-8")
                except Exception:
                    pass

            if response_text:
                print(f"Raw Response Body: {response_text}")
                try:
                    body_json = json.loads(response_text)
                    print(f"Parsed Response JSON: {json.dumps(body_json, indent=2)}")
                except Exception:
                    pass

            if response_obj:
                print(f"Response Object Type: {type(response_obj)}")
                print("--- Response Object Attributes ---")
                for attr in dir(response_obj):
                    if not attr.startswith('_'):
                        try:
                            val = getattr(response_obj, attr)
                            if val is not None and not callable(val):
                                print(f"  {attr}: {val}")
                        except Exception:
                            pass

        print("-----------------------------------------\n")
        sys.stdout.flush()

proxy_handler_instance = CustomRateLimitLogger()
