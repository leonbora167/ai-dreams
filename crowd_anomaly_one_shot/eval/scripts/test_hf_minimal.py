"""Minimal Hugging Face diagnostic script to isolate model loading and SSL certificates."""

import os
import sys
import traceback

print("=" * 60)
print(f"Python Executable : {sys.executable}")
print(f"Python Version    : {sys.version.split()[0]}")
print("=" * 60)

# Check relevant environment variables
env_keys = ["HF_ENDPOINT", "HF_HUB_DISABLE_SSL_VERIFICATION", "REQUESTS_CA_BUNDLE",
            "CURL_CA_BUNDLE", "SSL_CERT_FILE", "HTTP_PROXY", "HTTPS_PROXY"]
found_envs = {k: os.environ[k] for k in env_keys if k in os.environ}
print(f"Active Network Envs: {found_envs if found_envs else 'None (default system settings)'}\n")

# Model to test (SigLIP SO400M default)
test_model_id = sys.argv[1] if len(sys.argv) > 1 else "google/siglip-so400m-patch14-384"
print(f"Testing model ID: {test_model_id}\n")

print("--- 1. Testing AutoModel.from_pretrained ---")
try:
    from transformers import AutoModel
    print("Calling AutoModel.from_pretrained()...")
    model = AutoModel.from_pretrained(test_model_id)
    print(f"SUCCESS: AutoModel loaded -> {type(model).__name__}")
except Exception as e:
    print(f"\nFAILED on AutoModel: {e}\n")
    print("Full Traceback:")
    traceback.print_exc()

print("\n--- 2. Testing AutoProcessor.from_pretrained ---")
try:
    from transformers import AutoProcessor
    print("Calling AutoProcessor.from_pretrained()...")
    proc = AutoProcessor.from_pretrained(test_model_id)
    print(f"SUCCESS: AutoProcessor loaded -> {type(proc).__name__}")
except Exception as e:
    print(f"\nFAILED on AutoProcessor: {e}\n")
    print("Full Traceback:")
    traceback.print_exc()

print("\n" + "=" * 60)
print("Diagnostic test finished.")
print("=" * 60)

