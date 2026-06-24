from litellm.integrations.custom_logger import CustomLogger
import sys
import json
import asyncio
import time
from collections import deque

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
            # 1. Clamping max_tokens to 4096 to prevent context window explosion
            max_tokens = data.get("max_tokens")
            if max_tokens is not None and max_tokens > 4096:
                print(f"\n[CustomLogger] Clamping max_tokens from {max_tokens} to 4096")
                data["max_tokens"] = 4096
                sys.stdout.flush()

            model_name = data.get("model", "unknown")

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
        print(f"\n--- [CustomLogger] API Call Failure ---")
        print(f"Model: {model}")
        print(f"Status Code: {status_code}")
        print(f"Exception Type: {type(exception).__name__}")
        print(f"Exception Message: {str(exception)}")

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

        print(f"-----------------------------------------\n")
        sys.stdout.flush()

proxy_handler_instance = CustomRateLimitLogger()
