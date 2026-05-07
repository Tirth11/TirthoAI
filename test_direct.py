from openai import OpenAI
import requests
import json

key_pro = "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"
key_flash = "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"

def test_requests(name, model, key):
    print(f"Testing {name} ({model}) via requests...")
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=5)
        print(f"  Status: {response.status_code}")
        print(f"  Result: {response.json().get('choices')[0]['message']['content']}")
    except Exception as e:
        print(f"  Error: {str(e)}")

test_requests("DeepSeek Pro", "deepseek-ai/deepseek-v4-pro", key_pro)
test_requests("DeepSeek Flash", "deepseek-ai/deepseek-v4-flash", key_flash)
