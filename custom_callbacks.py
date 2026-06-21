from litellm.integrations.custom_logger import CustomLogger
import sys
import json

class CustomRateLimitLogger(CustomLogger):
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
                            # Format large dicts or display simply
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

            # Check response_obj if provided
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
