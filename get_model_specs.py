import sys
import json
import litellm
import custom_callbacks

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "kaggle-agent"
    try:
        info = litellm.get_model_info(model_name)
        max_in = info.get("max_input_tokens") or 200000
        max_out = info.get("max_output_tokens") or 8192
    except Exception:
        max_in = 200000
        max_out = 8192
    print(json.dumps({"max_input_tokens": max_in, "max_output_tokens": max_out}))

if __name__ == "__main__":
    main()
