from openai import OpenAI
import sys

models = [
    ("DeepSeek Pro", "deepseek-ai/deepseek-v4-pro", "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"),
    ("DeepSeek Flash (v4)", "deepseek-ai/deepseek-v4-flash", "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"),
    ("DeepSeek Flash (v3)", "deepseek-ai/deepseek-v3", "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"),
    ("GLM 4.7", "z-ai/glm4.7", "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")
]

for name, model_id, key in models:
    print(f"Testing {name} ({model_id})...")
    try:
        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=key)
        completion = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
            timeout=10
        )
        print(f"  Result: {completion.choices[0].message.content}")
    except Exception as e:
        print(f"  Error: {str(e)}")
