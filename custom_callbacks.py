from litellm.integrations.custom_logger import CustomLogger
import sys
import json
import asyncio
import time
from collections import deque
import os
from kaggle_auth import kaggle_auth_manager

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
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type, **kwargs):
        try:
            # 1. Clamping max_tokens to 8192 to prevent context window explosion
            max_tokens = data.get("max_tokens")
            if max_tokens is not None and max_tokens > 8192:
                print(f"\n[CustomLogger] Clamping max_tokens from {max_tokens} to 8192")
                data["max_tokens"] = 8192
                sys.stdout.flush()

            model_name = data.get("model", "unknown")

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

            # --- Dynamic Kaggle Credential Injection ---
            api_base = data.get("api_base") or (data.get("litellm_params", {}).get("api_base") if isinstance(data.get("litellm_params"), dict) else "") or ""
            is_kaggle_target = "kaggle" in model_name.lower() or "kaggle" in str(api_base).lower() or "mp-staging.kaggle.net" in str(api_base)
            
            if is_kaggle_target:
                if not kaggle_auth_manager.is_token_valid():
                    kaggle_auth_manager.refresh_credentials()
                
                active_key = kaggle_auth_manager.get_proxy_key()
                active_url = kaggle_auth_manager.get_proxy_url()
                
                if active_key:
                    data["api_key"] = active_key
                    if "litellm_params" in data and isinstance(data["litellm_params"], dict):
                        data["litellm_params"]["api_key"] = active_key
                        if not data["litellm_params"].get("api_base"):
                            data["litellm_params"]["api_base"] = active_url
                    os.environ["KAGGLE_MODEL_PROXY_KEY"] = active_key
                    os.environ["KAGGLE_MODEL_PROXY_URL"] = active_url
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

            # 3. Client-Side Rate Limiting (Preventing 429)
            # Enforce client-side rate limits before sending request to NIM catalog
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
        try:
            self._log_failure(kwargs, response_obj)
        except Exception as e:
            print(f"[CustomLogger] Error in sync callback: {e}", file=sys.stderr)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
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
