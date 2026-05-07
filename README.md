# DeepSeek Agent

A minimal local CLI agent for DeepSeek's OpenAI-compatible chat API.

## Setup

1. Set your API key in an environment variable:

```powershell
$env:DEEPSEEK_API_KEY = "your-key-here"
```

2. Run the agent:

```powershell
python .\deepseek_agent.py
```

## Optional settings

- `DEEPSEEK_MODEL`: defaults to `deepseek-chat`
- `DEEPSEEK_BASE_URL`: defaults to `https://api.deepseek.com`

Example:

```powershell
$env:DEEPSEEK_MODEL = "deepseek-reasoner"
python .\deepseek_agent.py
```

## Notes

- The API key is never written into the repository.
- If you want, this can be extended into a VS Code task, a web UI, or an MCP-style agent next.
