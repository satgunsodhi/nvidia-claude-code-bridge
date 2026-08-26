"""
Kaggle Model Proxy Dynamic Authentication Manager

Handles automatic rotation, expiry checking, and caching of Kaggle Model Proxy
credentials for LiteLLM and Claude Code.
"""

import os
import sys
import time
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from dotenv import dotenv_values


class KaggleAuthManager:
    """Manages short-lived Kaggle Model Proxy credentials with automatic rotation."""

    def __init__(self, env_file_path: Optional[str] = None):
        self.workspace_dir = Path(__file__).parent.resolve()
        self.env_file = Path(env_file_path) if env_file_path else self.workspace_dir / ".kaggle_proxy.env"
        self._lock = threading.Lock()
        
        self.proxy_url: Optional[str] = None
        self.proxy_key: Optional[str] = None
        self.expiry_timestamp: Optional[float] = None
        self.expiry_str: Optional[str] = None
        
        # Initial load from existing env file if present
        self._load_from_env_file()

    def _load_from_env_file(self) -> bool:
        """Loads cached credentials from .kaggle_proxy.env file if available."""
        if not self.env_file.exists():
            return False
        
        try:
            values = dotenv_values(str(self.env_file))
            url = values.get("MODEL_PROXY_URL")
            key = values.get("MODEL_PROXY_API_KEY")
            expiry = values.get("MODEL_PROXY_EXPIRY_TIME")
            
            if url and key:
                # Ensure OpenAPI format
                if not url.endswith("/openapi") and not url.endswith("/genai"):
                    self.proxy_url = f"{url.rstrip('/')}/openapi"
                else:
                    self.proxy_url = url
                    
                self.proxy_key = key
                self.expiry_str = expiry
                
                if expiry:
                    try:
                        # ISO 8601 parsing with timezone
                        dt = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
                        self.expiry_timestamp = dt.timestamp()
                    except Exception:
                        self.expiry_timestamp = None
                
                # Sync with environment
                os.environ["KAGGLE_MODEL_PROXY_URL"] = self.proxy_url
                os.environ["KAGGLE_MODEL_PROXY_KEY"] = self.proxy_key
                return True
        except Exception as e:
            print(f"[KaggleAuth] Warning loading env file: {e}", file=sys.stderr)
        return False

    def is_token_valid(self, buffer_seconds: int = 300) -> bool:
        """Checks if current token exists and has at least `buffer_seconds` remaining."""
        if not self.proxy_key or not self.proxy_url:
            return False
        if self.expiry_timestamp is None:
            return True  # If no expiry info, assume valid until rejected
        
        now = time.time()
        return (self.expiry_timestamp - now) > buffer_seconds

    def refresh_credentials(self, force: bool = False, buffer_seconds: int = 300) -> Tuple[str, str]:
        """
        Refreshes credentials if expired or if force=True.
        Thread-safe to prevent multiple simultaneous subprocess calls.
        """
        with self._lock:
            # Double check after acquiring lock
            if not force and self.is_token_valid(buffer_seconds=buffer_seconds):
                return self.proxy_url, self.proxy_key

            print("\n\033[1;36m[KaggleAuth] Refreshing short-lived Kaggle Model Proxy token...\033[0m")
            sys.stdout.flush()

            # Ensure access_token exists if kaggle.json is present
            kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
            access_token = Path.home() / ".kaggle" / "access_token"
            if kaggle_json.exists() and not access_token.exists():
                try:
                    import json
                    with open(kaggle_json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if "key" in data:
                        with open(access_token, "w", encoding="utf-8") as f:
                            f.write(data["key"].strip())
                except Exception:
                    pass

            # Execute Kaggle CLI benchmarks auth
            kaggle_exe = sys.executable.replace("python.exe", "kaggle.exe")
            if not Path(kaggle_exe).exists():
                kaggle_exe = "kaggle"

            cmd = [kaggle_exe, "benchmarks", "auth", "-y", "--env-file", str(self.env_file)]

            try:
                subprocess.run(
                    cmd,
                    cwd=str(self.workspace_dir),
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30
                )
                print("\033[1;32m[KaggleAuth] Successfully refreshed Kaggle credentials.\033[0m")
            except subprocess.CalledProcessError as e:
                print(f"\033[1;31m[KaggleAuth] Error refreshing Kaggle token: {e.stderr or e.stdout}\033[0m", file=sys.stderr)
                # If we still have an existing token, keep using it as fallback
                if self.proxy_url and self.proxy_key:
                    return self.proxy_url, self.proxy_key
                raise RuntimeError(f"Kaggle CLI auth failed: {e.stderr or e.stdout}") from e
            except Exception as e:
                print(f"\033[1;31m[KaggleAuth] Subprocess execution failed: {e}\033[0m", file=sys.stderr)
                if self.proxy_url and self.proxy_key:
                    return self.proxy_url, self.proxy_key
                raise

            # Reload values
            if not self._load_from_env_file():
                raise RuntimeError(f"Failed to parse updated credentials from {self.env_file}")

            if self.expiry_timestamp:
                remaining_mins = max(0, int((self.expiry_timestamp - time.time()) / 60))
                print(f"\033[1;32m[KaggleAuth] Token valid for ~{remaining_mins} minutes.\033[0m\n")
            sys.stdout.flush()

            return self.proxy_url, self.proxy_key

    def get_proxy_key(self) -> str:
        """Returns valid proxy API key, refreshing if necessary."""
        _, key = self.refresh_credentials()
        return key

    def get_proxy_url(self) -> str:
        """Returns valid proxy base URL, refreshing if necessary."""
        url, _ = self.refresh_credentials()
        return url

    def get_status_info(self) -> dict:
        """Returns status dictionary for diagnostics."""
        now = time.time()
        remaining_secs = (self.expiry_timestamp - now) if self.expiry_timestamp else None
        return {
            "has_token": bool(self.proxy_key),
            "proxy_url": self.proxy_url,
            "expiry_time": self.expiry_str,
            "expires_in_seconds": int(remaining_secs) if remaining_secs is not None else None,
            "is_valid": self.is_token_valid(buffer_seconds=0),
            "env_file": str(self.env_file),
        }


# Global singleton instance
kaggle_auth_manager = KaggleAuthManager()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kaggle Model Proxy Auth Utility")
    parser.add_argument("--refresh", action="store_true", help="Force refresh credentials immediately")
    parser.add_argument("--status", action="store_true", help="Display current authentication status")
    args = parser.parse_args()

    if args.refresh:
        url, key = kaggle_auth_manager.refresh_credentials(force=True)
        print(f"URL: {url}")
        print(f"Key Prefix: {key[:15]}...")
    elif args.status or len(sys.argv) == 1:
        status = kaggle_auth_manager.get_status_info()
        print("\n--- Kaggle Model Proxy Status ---")
        for k, v in status.items():
            print(f"  {k}: {v}")
        print("---------------------------------\n")
