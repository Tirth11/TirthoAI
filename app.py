import os
import sys

# Monkey patch for gevent if needed, but we'll try gthread first
if os.environ.get("USE_GEVENT", "False").lower() == "true":
    try:
        import gevent.monkey
        gevent.monkey.patch_all()
    except ImportError:
        pass

import json
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import google.generativeai as genai
import warnings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Suppress the deprecation warning for google-generativeai
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

from openai import OpenAI
import uuid
from datetime import datetime
import json
from werkzeug.utils import secure_filename
import base64
import ollama

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Model Configurations
# -------------------------------------------------------------------
# CATEGORY LABELS:  reasoning | vision | coding | creative | general
MODEL_CONFIGS = {
    # ── FREE FRONTIER MODELS ────────────────────────────────────────
    "Gemini 1.5 Flash (Free)": {
        "model": "gemini-1.5-flash",
        "api_key": os.getenv("GEMINI_API_KEY"),
        "category": "general",
        "badge": "♊",
        "description": "Google's fastest model — free via AI Studio",
        "provider": "google"
    },
    "Gemini 1.5 Pro (Free)": {
        "model": "gemini-1.5-pro",
        "api_key": os.getenv("GEMINI_API_KEY"),
        "category": "reasoning",
        "badge": "🌌",
        "description": "Google's most capable model — free tier available",
        "provider": "google"
    },
    "Gemini 3 Flash": {
        "model": "gemini-3-flash-preview",
        "api_key": "local",
        "category": "general",
        "badge": "⚡",
        "description": "Google's next-gen Flash model — high speed & efficiency",
        "provider": "ollama"
    },
    "Gemini 3 Flash (Cloud)": {
        "model": "gemini-3-flash-preview:cloud",
        "api_key": "local",
        "category": "general",
        "badge": "☁️",
        "description": "Cloud-optimized Gemini 3 Flash via Ollama",
        "provider": "ollama"
    },
    "Gemma 4 Distill": {
        "model": "cleex/gemma-4-31B-it-Claude-Opus-Distill-GGUF",
        "api_key": "local",
        "category": "reasoning",
        "badge": "💎",
        "description": "Gemma 4 31B distilled from Claude Opus",
        "provider": "ollama"
    },
    # ── OLLAMA PREMIUM MODELS ────────────────────────────────────────
    "DeepSeek R1 (Ollama)": {
        "model": "deepseek-r1",
        "api_key": "local",
        "category": "reasoning",
        "badge": "🧠",
        "description": "DeepSeek's advanced reasoning model with CoT",
        "provider": "ollama"
    },
    "Qwen 3.5 122B": {
        "model": "qwen3.5:122b",
        "api_key": "local",
        "category": "general",
        "badge": "🏮",
        "description": "Alibaba's flagship 122B model for frontier performance",
        "provider": "ollama"
    },
    "Gemma 4 26B": {
        "model": "gemma4:26b",
        "api_key": "local",
        "category": "general",
        "badge": "💎",
        "description": "Google's flagship open model for reasoning & agents",
        "provider": "ollama"
    },
    "Mistral Large 3 (Ollama)": {
        "model": "mistral-large:latest",
        "api_key": "local",
        "category": "general",
        "badge": "🌊",
        "description": "Mistral's most capable flagship model locally",
        "provider": "ollama"
    },
    "Qwen 2.5 Coder 32B": {
        "model": "qwen2.5-coder:32b",
        "api_key": "local",
        "category": "coding",
        "badge": "🛠️",
        "description": "King of local coding — 32B parameters",
        "provider": "ollama"
    },
    "Llama 3.3 70B (Ollama)": {
        "model": "llama3.3:70b",
        "api_key": "local",
        "category": "general",
        "badge": "🦙",
        "description": "Meta's reliable 70B general assistant locally",
        "provider": "ollama"
    },
    "GLM 4.7 Flash": {
        "model": "glm-4.7-flash",
        "api_key": "local",
        "category": "coding",
        "badge": "💻",
        "description": "ZhipuAI's GLM 4.7 Flash — optimized for speed and code",
        "provider": "ollama"
    },
    "Llama 3.3 70B (Free)": {
        "model": "meta/llama-3.3-70b-instruct",
        "api_key": os.getenv("KIMI_26_KEY"),
        "category": "general",
        "badge": "🦙",
        "description": "Meta's reliable 70B model via NVIDIA NIM",
        "provider": "nvidia"
    },
    # ── REASONING ───────────────────────────────────────────────────
    "DeepSeek V4 Pro": {
        "model": "deepseek-ai/deepseek-v4-pro",
        "api_key": os.getenv("NVIDIA_API_KEY_PRO"),
        "temperature": 0.6,
        "max_tokens": 8192,
        "supports_vision": True,
        "category": "reasoning",
        "badge": "🌌",
        "description": "DeepSeek's flagship V4 model — Enterprise Cloud",
        "extra_body": {"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}},
        "provider": "nvidia"
    },
    "DeepSeek R1": {
        "model": "meta/llama-3.3-70b-instruct",
        "api_key": os.getenv("KIMI_26_KEY"),
        "temperature": 1,
        "max_tokens": 16384,
        "supports_vision": False,
        "category": "reasoning",
        "badge": "🧠",
        "description": "DeepSeek's flagship reasoning model",
        "extra_body": {"chat_template_kwargs": {"thinking": True}}
    },
    "Dracarys Llama 3.1 70B": {
        "model": "abacusai/dracarys-llama-3.1-70b-instruct",
        "api_key": os.getenv("NVIDIA_API_KEY_DRACARYS"),
        "temperature": 0.5,
        "max_tokens": 1024,
        "category": "reasoning",
        "badge": "🔥",
        "description": "AbacusAI's Dracarys — fine-tuned Llama 3.1 70B for superior instruction following",
        "provider": "nvidia"
    },
    "Nemotron 340B": {
        "model": "nvidia/nemotron-4-340b-instruct",
        "api_key": os.getenv("KIMI_26_KEY"),
        "temperature": 0.7,
        "max_tokens": 4096,
        "supports_vision": False,
        "category": "reasoning",
        "badge": "🏛️",
        "description": "NVIDIA's massive 340B reasoning model"
    },
    "Kimi K2": {
        "model": "meta/llama-3.3-70b-instruct",
        "api_key": os.getenv("KIMI_26_KEY"),
        "temperature": 1,
        "max_tokens": 16384,
        "supports_vision": False,
        "category": "reasoning",
        "badge": "🧠",
        "description": "Deep math & logical reasoning",
        "extra_body": {"chat_template_kwargs": {"thinking": True}}
    },

    # ── VISION / MULTIMODAL ─────────────────────────────────────────
    "Llama 4 Maverick": {
        "model": "meta/llama-4-maverick-17b-128e-instruct",
        "api_key": os.getenv("KIMI_26_KEY"),
        "temperature": 0.7,
        "max_tokens": 8192,
        "supports_vision": True,
        "category": "vision",
        "badge": "👁️",
        "description": "Meta's latest multimodal powerhouse"
    },
    "Llama Vision 90B": {
        "model": "meta/llama-3.2-90b-vision-instruct",
        "api_key": os.getenv("NVIDIA_API_KEY_VISION"),
        "temperature": 0.7,
        "max_tokens": 8192,
        "supports_vision": True,
        "category": "vision",
        "badge": "🖼️",
        "description": "90B vision model for image analysis"
    },
    "Phi-4 Multimodal": {
        "model": "microsoft/phi-4-multimodal-instruct",
        "api_key": os.getenv("KIMI_26_KEY"),
        "temperature": 0.7,
        "max_tokens": 4096,
        "supports_vision": True,
        "category": "vision",
        "badge": "🔭",
        "description": "Microsoft multimodal — fast & efficient"
    },

    # ── CODING ──────────────────────────────────────────────────────
    "Qwen 2.5 Coder": {
        "model": "qwen/qwen2.5-coder-32b-instruct",
        "api_key": os.getenv("KIMI_26_KEY"),
        "temperature": 0.7,
        "max_tokens": 16384,
        "supports_vision": False,
        "category": "coding",
        "badge": "🚀",
        "description": "Powerful Qwen 2.5 Coder — best for complex code tasks"
    },
    "GLM 4.7": {
        "model": "meta/llama-3.3-70b-instruct",
        "api_key": os.getenv("KIMI_26_KEY"),
        "temperature": 1,
        "max_tokens": 16384,
        "supports_vision": False,
        "category": "coding",
        "badge": "💻",
        "description": "Excellent for code generation & debugging",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
    },
    "Devstral 2": {
        "model": "meta/llama-3.3-70b-instruct",
        "api_key": os.getenv("KIMI_26_KEY"),
        "temperature": 0.7,
        "max_tokens": 16384,
        "supports_vision": False,
        "category": "coding",
        "badge": "🛠️",
        "description": "Mistral's dev-focused 123B coding model"
    },

    # ── CREATIVE ────────────────────────────────────────────────────
    "Kimi 2.6": {
        "model": "meta/llama-3.3-70b-instruct",
        "api_key": os.getenv("KIMI_26_KEY"),
        "temperature": 1,
        "max_tokens": 16384,
        "supports_vision": False,
        "category": "creative",
        "badge": "✍️",
        "description": "Creative writing & storytelling",
        "extra_body": {"chat_template_kwargs": {"thinking": True}}
    },
    "Gemma 2 27B": {
        "model": "google/gemma-2-27b-it",
        "api_key": os.getenv("KIMI_26_KEY"),
        "temperature": 1,
        "max_tokens": 8192,
        "supports_vision": False,
        "category": "creative",
        "badge": "💎",
        "description": "Google's lightweight creative assistant"
    },

    # ── GENERAL / FAST ───────────────────────────────────────────────
    "Qwen 2.5 72B": {
        "model": "qwen/qwen2.5-72b-instruct",
        "api_key": os.getenv("KIMI_26_KEY"),
        "temperature": 0.7,
        "max_tokens": 16384,
        "supports_vision": False,
        "category": "general",
        "badge": "🏮",
        "description": "Qwen 2.5 72B — massive general purpose model"
    },
    "DeepSeek Flash": {
        "model": "meta/llama-3.3-70b-instruct",
        "api_key": os.getenv("KIMI_26_KEY"),
        "temperature": 1,
        "max_tokens": 16384,
        "supports_vision": False,
        "category": "general",
        "badge": "⚡",
        "description": "Ultra-fast DeepSeek",
        "extra_body": {"chat_template_kwargs": {"thinking": False}}
    },
    "Llama 3.3 70B": {
        "model": "meta/llama-3.3-70b-instruct",
        "api_key": os.getenv("KIMI_26_KEY"),
        "temperature": 0.7,
        "max_tokens": 8192,
        "supports_vision": False,
        "category": "general",
        "badge": "🦙",
        "description": "Meta's reliable 70B general assistant"
    },
    "Sarvam M": {
        "model": "sarvamai/sarvam-m",
        "api_key": os.getenv("NVIDIA_API_KEY_SARVAM"),
        "temperature": 0.5,
        "max_tokens": 16384,
        "category": "general",
        "badge": "🇮🇳",
        "description": "Sarvam AI's multilingual model — optimized for Indian languages",
        "provider": "nvidia"
    },
    "Minimax M2.5": {
        "model": "minimaxai/minimax-m2.5",
        "api_key": os.getenv("NVIDIA_API_KEY_VISION"),
        "temperature": 1,
        "max_tokens": 8192,
        "category": "general",
        "badge": "🚀",
        "description": "MiniMax's high-performance M2.5 model",
        "provider": "nvidia"
    },
    "Mistral Large 3": {
        "model": "mistralai/mistral-large-3-675b-instruct-2512",
        "api_key": os.getenv("KIMI_26_KEY"),
        "temperature": 0.7,
        "max_tokens": 8192,
        "supports_vision": False,
        "category": "general",
        "badge": "🌊",
        "description": "Mistral's 675B flagship instruction model"
    },
    # ── IMAGE GENERATION ────────────────────────────────────────────
    "Qwen Image": {
        "model": "nvidia/qwen-image",
        "api_key": os.getenv("NVIDIA_API_KEY_QWEN_IMAGE"),
        "category": "image_generation",
        "badge": "🎨",
        "description": "NVIDIA Text-to-Image foundation model"
    },
    "Stable Diffusion XL": {
        "model": "stabilityai/stable-diffusion-xl",
        "api_key": os.getenv("KIMI_26_KEY"),
        "category": "image_generation",
        "badge": "🖼️",
        "description": "High-quality photographic image generation"
    },
    "SDXL Turbo": {
        "model": "stabilityai/sdxl-turbo",
        "api_key": os.getenv("KIMI_26_KEY"),
        "category": "image_generation",
        "badge": "⚡",
        "description": "Ultra-fast real-time image generation"
    },
}

# Store conversations in memory
conversations = {}
current_conversation_id = None

def get_client(actual_label):
    config = MODEL_CONFIGS.get(actual_label, MODEL_CONFIGS["DeepSeek V4 Pro"])
    api_key = config.get('api_key')
    if not api_key:
        return None
    return OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def auto_select_model(text, files_data=None):
    """
    Intelligently select the best model based on prompt content.
    Returns (model_label, reason_string).
    """
    t = text.lower().strip()
    word_count = len(t.split())

    # --- Image/vision input -> Llama 4 Maverick (best multimodal) ---
    if files_data and any(f.get('type') == 'image' for f in files_data):
        return "Llama 4 Maverick", "👁️ Llama 4 Maverick selected for image analysis"

    # --- Simple greetings -> DeepSeek Flash ---
    greet_patterns = ["hi", "hello", "hey", "how are you", "what's up", "whats up",
                      "good morning", "good night", "thanks", "thank you", "ok", "okay", "bye"]
    if word_count <= 5 and any(t.startswith(g) or t == g for g in greet_patterns):
        return "DeepSeek Flash", "⚡ DeepSeek Flash selected for quick conversation"

    # --- UI / frontend / design -> Qwen 2.5 Coder ---
    ui_keywords = [
        "ui", "ux", "design", "html", "css", "tailwind", "frontend", "interface",
        "component", "layout", "responsive", "mobile", "web app", "website",
        "button", "form", "modal", "navbar", "sidebar", "dashboard", "landing page"
    ]
    if any(kw in t for kw in ui_keywords):
        return "Qwen 2.5 Coder", "🚀 Qwen 2.5 Coder selected for UI/frontend tasks"

    # --- Complex code -> Qwen 2.5 Coder, else GLM 4.7 ---
    hard_code_keywords = [
        "architecture", "microservice", "system design", "refactor", "optimize",
        "performance", "scalable", "kubernetes", "docker", "ci/cd", "devops"
    ]
    if any(kw in t for kw in hard_code_keywords):
        return "Devstral 2", "🛠️ Devstral 2 selected for complex dev tasks"

    code_keywords = [
        "code", "program", "script", "function", "class", "debug", "error", "bug",
        "python", "javascript", "java", "c++", "c#", "typescript", "react", "sql",
        "api", "endpoint", "database", "flask", "django", "fastapi",
        "implement", "write a", "create a function", "unit test", "deploy",
        "git", "pull request"
    ]
    if any(kw in t for kw in code_keywords):
        return "GLM 4.7 Flash", "💻 GLM 4.7 Flash selected for coding tasks"

    # --- Math / logic / deep reasoning -> Kimi K2 or Nemotron 340B ---
    math_keywords = [
        "prove", "proof", "theorem", "calculate", "integral", "derivative", "equation",
        "algebra", "geometry", "calculus", "statistics", "probability", "matrix",
        "algorithm", "complexity", "big o", "dynamic programming", "recursion",
        "logic", "reasoning", "deduce", "infer", "step by step", "step-by-step",
        "solve", "evaluate", "explain why"
    ]
    if any(kw in t for kw in math_keywords):
        if word_count > 50:
            return "Nemotron 340B", "🏛️ Nemotron 340B selected for complex reasoning"
        return "Kimi K2", "🧠 Kimi K2 selected for deep reasoning & math"

    # --- Creative / writing -> Kimi 2.6 ---
    creative_keywords = [
        "write a story", "poem", "creative", "essay", "draft", "letter",
        "blog post", "article", "narrative", "fiction", "character", "plot",
        "song", "lyrics", "dialogue", "brainstorm", "idea", "slogan",
        "caption", "tweet", "social media", "marketing copy"
    ]
    if any(kw in t for kw in creative_keywords):
        return "Kimi 2.6", "✍️ Kimi 2.6 selected for creative writing"

    # --- Summarization / translation -> Qwen 2.5 72B ---
    summary_keywords = [
        "summarize", "summary", "translate", "translation", "paraphrase",
        "tldr", "key points", "main points", "extract", "overview", "brief"
    ]
    if any(kw in t for kw in summary_keywords):
        return "Qwen 2.5 72B", "🏮 Qwen 2.5 72B selected for general tasks"

    # --- General knowledge / Google stuff -> Gemini ---
    google_keywords = [
        "google", "gemini", "alphabet", "youtube", "android", "search the web",
        "current events", "news", "latest", "what is happening"
    ]
    if any(kw in t for kw in google_keywords):
        return "Gemini 1.5 Flash (Free)", "♊ Gemini 1.5 Flash selected for Google-related or general tasks"

    # --- Gemini 3 / Next gen -> Gemini 3 Flash ---
    if "gemini 3" in t or "next gen" in t:
        if "cloud" in t:
            return "Gemini 3 Flash (Cloud)", "☁️ Gemini 3 Flash (Cloud) selected"
        return "Gemini 3 Flash", "⚡ Gemini 3 Flash selected for next-gen tasks"

    # --- Distilled / Claude-like reasoning -> Gemma 4 Distill ---
    if "distill" in t or "claude opus" in t:
        return "Gemma 4 Distill", "💎 Gemma 4 Distill selected for specialized reasoning"

    # --- Multilingual / Indian languages -> Sarvam M ---
    indian_language_keywords = [
        "hindi", "bengali", "marathi", "telugu", "tamil", "gujarati", "urdu",
        "kannada", "odia", "punjabi", "malayalam", "assamese", "maithili", "santhali",
        "sanskrit", "konkani", "nepali", "bodo", "dogri", "manipuri", "kashmiri", "sindhi",
        "indian language", "translation to", "translate to"
    ]
    if any(kw in t for kw in indian_language_keywords):
        return "Sarvam M", "🇮🇳 Sarvam M selected for multilingual/Indian language task"

    # --- Chinese language / Roleplay -> Minimax M2.5 ---
    minimax_keywords = ["chinese", "mandarin", "roleplay", "creative story", "large context"]
    if any(kw in t for kw in minimax_keywords):
        return "Minimax M2.5", "🚀 Minimax M2.5 selected for creative or Chinese language task"

    # --- Image Generation -> Stable Diffusion XL ---
    image_keywords = [
        "generate an image", "draw a", "create an image", "make a picture",
        "generate a picture", "visualize a", "render a", "sketch a"
    ]
    if any(kw in t for kw in image_keywords):
        return "Stable Diffusion XL", "🎨 Stable Diffusion XL selected for image generation"

    # --- Analysis / research -> DeepSeek V4 Pro ---
    analysis_keywords = [
        "analyze", "analysis", "compare", "comparison", "research", "explain",
        "difference between", "pros and cons", "advantages", "disadvantages",
        "review", "assess", "what is", "who is", "how to",
        "best way", "recommend", "suggest", "plan", "strategy"
    ]
    if any(kw in t for kw in analysis_keywords):
        return "DeepSeek V4 Pro", "🔬 DeepSeek V4 Pro selected for analysis & research"

    # --- Short simple prompts -> Llama 3.3 70B (fast & reliable) ---
    if word_count <= 20:
        return "Llama 3.3 70B", "🦙 Llama 3.3 70B selected for reliable performance"

    # --- Default: Llama 3.3 70B for stability ---
    return "Llama 3.3 70B", "🦙 Llama 3.3 70B selected for general stability"


def create_conversation():
    conv_id = str(uuid.uuid4())[:8]
    conversations[conv_id] = {
        "id": conv_id,
        "title": "New Chat",
        "created_at": datetime.now().isoformat(),
        "messages": [
            {
                "role": "system",
                "content": "You are TirthoAI, a powerful AI assistant powered by multiple advanced models. You provide accurate, helpful, and insightful responses across various domains including coding, analysis, and general chat. You can also analyze uploaded files and images."
            }
        ]
    }
    return conv_id

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

@app.route('/')
def index():
    return render_template('deepseek_enhanced.html')

@app.route('/models', methods=['GET'])
def get_models():
    """Return list of all models with metadata"""
    models_list = []
    for label, cfg in MODEL_CONFIGS.items():
        models_list.append({
            "label": label,
            "model": cfg["model"],
            "category": cfg.get("category", "general"),
            "badge": cfg.get("badge", "🤖"),
            "description": cfg.get("description", ""),
            "supports_vision": cfg.get("supports_vision", False)
        })
    return jsonify(models_list)

@app.route('/conversations', methods=['GET'])
def get_conversations():
    """Get list of all conversations"""
    conv_list = []
    for conv_id, conv_data in conversations.items():
        conv_list.append({
            "id": conv_id,
            "title": conv_data.get("title", "Chat"),
            "created_at": conv_data.get("created_at", "")
        })
    return jsonify(sorted(conv_list, key=lambda x: x['created_at'], reverse=True))

@app.route('/conversation/new', methods=['POST'])
def new_conversation():
    """Create a new conversation"""
    global current_conversation_id
    current_conversation_id = create_conversation()
    return jsonify({'id': current_conversation_id})

@app.route('/conversation/<conv_id>', methods=['GET'])
def get_conversation(conv_id):
    """Get a specific conversation"""
    global current_conversation_id
    current_conversation_id = conv_id
    if conv_id not in conversations:
        return jsonify({'error': 'Conversation not found'}), 404
    
    conv = conversations[conv_id]
    # Don't send system message to frontend
    messages = [m for m in conv['messages'] if m['role'] != 'system']
    return jsonify({
        "id": conv_id,
        "title": conv.get("title", "Chat"),
        "messages": messages
    })

@app.route('/conversation/<conv_id>', methods=['DELETE'])
def delete_conversation(conv_id):
    """Delete a conversation"""
    global current_conversation_id
    if conv_id in conversations:
        del conversations[conv_id]
        if current_conversation_id == conv_id:
            current_conversation_id = None
        return jsonify({'success': True})
    return jsonify({'error': 'Conversation not found'}), 404

@app.route('/reset', methods=['POST'])
def reset_system():
    """Clear all conversations and reset system state"""
    global conversations, current_conversation_id
    conversations = {}
    current_conversation_id = None
    # Clear upload folder
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    return jsonify({'success': True})

@app.route('/health', methods=['GET'])
def health_check():
    """Check status of key models"""
    test_models = ["DeepSeek V4 Pro", "Gemini 1.5 Flash (Free)", "Dracarys Llama 3.1 70B"]
    status = {}
    
    for label in test_models:
        if label not in MODEL_CONFIGS:
            continue
        
        config = MODEL_CONFIGS[label]
        try:
            if config.get("provider") == "google":
                genai.configure(api_key=config["api_key"])
                model = genai.GenerativeModel(config["model"])
                # Just a very small request
                model.generate_content("ping", generation_config={"max_output_tokens": 1})
            elif config.get("provider") == "ollama":
                # Check if ollama is running and has the model
                models = ollama.list()
                if not any(m['name'] == config['model'] for m in models.get('models', [])):
                    status[label] = "offline (model missing)"
                    continue
            else:
                # NIM models
                client = get_client(label)
                client.chat.completions.create(
                    model=config["model"],
                    messages=[{"role": "user", "content": "p"}],
                    max_tokens=1
                )
            status[label] = "online"
        except Exception as e:
            status[label] = f"offline: {str(e)[:50]}"
            
    return jsonify(status)

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages with model selection and streaming response"""
    global current_conversation_id
    
    try:
        data = request.get_json()
        message_text = data.get('message', '').strip()
        files_data = data.get('files', [])
        selected_label = data.get('model', 'Auto')
        
        if not message_text and not files_data:
            return jsonify({'error': 'No message or files provided'}), 400
        
        if current_conversation_id is None:
            current_conversation_id = create_conversation()
        
        message_content = []
        if message_text:
            message_content.append({"type": "text", "text": message_text})
        
        for file_data in files_data:
            if file_data['type'] == 'image':
                message_content.append({"type": "image_url", "image_url": {"url": file_data['data']}})
            elif file_data['type'] == 'text':
                content = file_data.get('content') or file_data.get('data') or ""
                message_content.append({"type": "text", "text": f"File: {file_data['name']}\nContent:\n{content}"})
        
        conv = conversations[current_conversation_id]
        user_msg_id = str(uuid.uuid4())
        
        # Simplify content if only text
        if len(message_content) == 1 and message_content[0]['type'] == 'text':
            final_user_content = message_content[0]['text']
        else:
            final_user_content = message_content

        conv['messages'].append({
            "id": user_msg_id,
            "role": "user",
            "content": final_user_content,
            "timestamp": datetime.now().isoformat()
        })
        
        if selected_label == "Auto":
            actual_label, selection_reason = auto_select_model(message_text, files_data)
        else:
            actual_label = selected_label
            selection_reason = None

        config = MODEL_CONFIGS.get(actual_label, MODEL_CONFIGS["DeepSeek V4 Pro"])
        client = None
        if config.get("provider") not in ["google", "ollama"]:
            client = get_client(actual_label)

        def generate():
            try:
                start_time = datetime.now()
                api_messages = []
                for m in conv['messages']:
                    m_content = m["content"]
                    if not config.get("supports_vision", False) and isinstance(m_content, list):
                        text_parts = []
                        for part in m_content:
                            if part["type"] == "text":
                                text_parts.append(part["text"])
                        m_content = "\n".join(text_parts)
                    api_messages.append({"role": m["role"], "content": m_content})

                if not config.get('api_key'):
                    error_msg = f"API key for model '{actual_label}' is missing. Please set the environment variable."
                    print(f"DEBUG ERROR: {error_msg}")
                    yield json.dumps({"error": error_msg}) + "\n"
                    return

                params = {
                    "model": config["model"],
                    "messages": api_messages,
                    "temperature": config.get("temperature", 1),
                    "max_tokens": config.get("max_tokens", 4096),
                    "stream": True,
                    "timeout": 120
                }
                if "extra_body" in config:
                    params["extra_body"] = config["extra_body"]

                if config.get("provider") == "google":
                    try:
                        print(f"DEBUG: Starting Google Gemini request for {actual_label}")
                        genai.configure(api_key=config["api_key"])
                        model = genai.GenerativeModel(config["model"])
                        
                        # Format history for Gemini
                        gemini_history = []
                        for m in conv['messages'][:-1]:
                            role = "user" if m["role"] == "user" else "model"
                            content = m["content"]
                            if isinstance(content, list):
                                content = "\n".join([p["text"] for p in content if p["type"] == "text"])
                            gemini_history.append({"role": role, "parts": [content]})
                        
                        chat = model.start_chat(history=gemini_history)
                        # ...
                        response = chat.send_message(message_text, stream=True)
                        full_response = ""
                        for chunk in response:
                            if chunk.text:
                                full_response += chunk.text
                                yield json.dumps({"content": chunk.text}) + "\n"
                        
                        end_time = datetime.now()
                        elapsed_seconds = (end_time - start_time).total_seconds()
                        elapsed_time = f"{int(elapsed_seconds // 60):02d}:{int(elapsed_seconds % 60):02d}"
                        
                        assistant_msg_id = str(uuid.uuid4())
                        assistant_msg = {
                            "id": assistant_msg_id,
                            "role": "assistant",
                            "content": full_response,
                            "timestamp": datetime.now().isoformat(),
                            "status": "done",
                            "modelUsed": actual_label,
                            "actualModel": config["model"],
                            "elapsedTime": elapsed_time,
                            "selectionReason": selection_reason
                        }
                        conv['messages'].append(assistant_msg)
                        
                        yield json.dumps({
                            "status": "done",
                            "elapsedTime": elapsed_time,
                            "messageId": assistant_msg_id
                        }) + "\n"
                        return
                    except Exception as e:
                        print(f"DEBUG ERROR Gemini: {str(e)}")
                        yield json.dumps({"error": f"Gemini request failed: {str(e)}"}) + "\n"
                        return

                elif config.get("provider") == "ollama":
                    try:
                        print(f"DEBUG: Starting Ollama request for {actual_label}")
                        response = ollama.chat(
                            model=config["model"],
                            messages=api_messages,
                            stream=True
                        )
                        
                        full_response = ""
                        for chunk in response:
                            content = chunk.get('message', {}).get('content', '')
                            if content:
                                full_response += content
                                yield json.dumps({"content": content}) + "\n"
                        
                        end_time = datetime.now()
                        elapsed_seconds = (end_time - start_time).total_seconds()
                        elapsed_time = f"{int(elapsed_seconds // 60):02d}:{int(elapsed_seconds % 60):02d}"
                        
                        assistant_msg_id = str(uuid.uuid4())
                        assistant_msg = {
                            "id": assistant_msg_id,
                            "role": "assistant",
                            "content": full_response,
                            "timestamp": datetime.now().isoformat(),
                            "status": "done",
                            "modelUsed": actual_label,
                            "actualModel": config["model"],
                            "elapsedTime": elapsed_time,
                            "selectionReason": selection_reason
                        }
                        conv['messages'].append(assistant_msg)
                        
                        yield json.dumps({
                            "status": "done",
                            "elapsedTime": elapsed_time,
                            "messageId": assistant_msg_id
                        }) + "\n"
                        return
                    except Exception as e:
                        print(f"DEBUG ERROR Ollama: {str(e)}")
                        yield json.dumps({"error": f"Ollama request failed: {str(e)}"}) + "\n"
                        return

                # ── INITIAL METADATA ───────────────────────────────────
                yield ":" + " " * 4096 + "\n"
                yield json.dumps({
                    "status": "start",
                    "modelUsed": actual_label,
                    "selectionReason": selection_reason,
                    "conversation_id": current_conversation_id
                }) + "\n"

                if client is None and config.get("provider") not in ["google", "ollama"]:
                    yield json.dumps({"error": f"Model client failed to initialize. Please check if the API key for '{actual_label}' is configured correctly."}) + "\n"
                    return

                if config.get("category") == "image_generation":
                    try:
                        print(f"DEBUG: Starting Image Generation request for {actual_label}")
                        response = client.images.generate(
                            model=config["model"],
                            prompt=message_text,
                            response_format="b64_json"
                        )
                        
                        img_b64 = response.data[0].b64_json
                        full_response = f"![Generated Image](data:image/png;base64,{img_b64})"
                        
                        yield json.dumps({"content": full_response}) + "\n"
                        
                        end_time = datetime.now()
                        elapsed_seconds = (end_time - start_time).total_seconds()
                        elapsed_time = f"{int(elapsed_seconds // 60):02d}:{int(elapsed_seconds % 60):02d}"
                        
                        assistant_msg_id = str(uuid.uuid4())
                        assistant_msg = {
                            "id": assistant_msg_id,
                            "role": "assistant",
                            "content": full_response,
                            "timestamp": datetime.now().isoformat(),
                            "status": "done",
                            "modelUsed": actual_label,
                            "actualModel": config["model"],
                            "elapsedTime": elapsed_time,
                            "selectionReason": selection_reason
                        }
                        conv['messages'].append(assistant_msg)
                        
                        yield json.dumps({
                            "status": "done",
                            "elapsedTime": elapsed_time,
                            "messageId": assistant_msg_id
                        }) + "\n"
                        return
                    except Exception as e:
                        print(f"DEBUG ERROR Image Gen: {str(e)}")
                        yield json.dumps({"error": f"Image Generation failed: {str(e)}"}) + "\n"
                        return

                try:
                    print(f"DEBUG: Starting API request to {config['model']}")
                    completion = client.chat.completions.create(**params)
                    print(f"DEBUG: API request started, waiting for chunks...")
                    full_response = ""
                    reasoning_content = ""
                    for chunk in completion:
                        if not hasattr(chunk, "choices") or not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        chunk_data = {}
                        
                        if hasattr(delta, "content") and delta.content is not None:
                            full_response += delta.content
                            chunk_data["content"] = delta.content
                        
                        reasoning = (
                            getattr(delta, "reasoning", None) or
                            getattr(delta, "reasoning_content", None) or
                            getattr(delta, "thinking", None) or
                            getattr(delta, "thinking_content", None)
                        )
                        if reasoning:
                            reasoning_content += reasoning
                            chunk_data["reasoning"] = reasoning
                        
                        if chunk_data:
                            yield json.dumps(chunk_data) + "\n"
                        else:
                            yield ": heartbeat\n"
                            
                    if not full_response and not reasoning_content:
                        print(f"DEBUG: Model {actual_label} returned NO CONTENT and NO REASONING.")
                        yield json.dumps({"content": "The model returned an empty response. This can happen if the model is overloaded or the prompt was blocked by safety filters."}) + "\n"
                        
                    end_time = datetime.now()
                    elapsed_seconds = (end_time - start_time).total_seconds()
                    elapsed_time = f"{int(elapsed_seconds // 60):02d}:{int(elapsed_seconds % 60):02d}"
                    
                    # Save to history
                    assistant_msg_id = str(uuid.uuid4())
                    assistant_msg = {
                        "id": assistant_msg_id,
                        "role": "assistant",
                        "content": full_response,
                        "timestamp": datetime.now().isoformat(),
                        "status": "done",
                        "modelUsed": actual_label,
                        "actualModel": config["model"],
                        "elapsedTime": elapsed_time,
                        "selectionReason": selection_reason,
                        "reasoning": reasoning_content
                    }
                    conv['messages'].append(assistant_msg)
                    if len([m for m in conv['messages'] if m['role'] != 'system']) == 2:
                        conv['title'] = message_text[:50] + ("..." if len(message_text) > 50 else "")
                    
                    yield json.dumps({
                        "status": "done",
                        "elapsedTime": elapsed_time,
                        "messageId": assistant_msg_id
                    }) + "\n"
                    
                except Exception as e:
                    print(f"DEBUG ERROR: {str(e)}")
                    yield json.dumps({"error": str(e)}) + "\n"

            except Exception as e:
                print(f"OUTER GENERATE ERROR: {str(e)}")
                yield json.dumps({"error": str(e)}) + "\n"

        return Response(generate(), mimetype='text/event-stream', headers={
            'X-Accel-Buffering': 'no',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        })
    except Exception as e:
        print(f"CRITICAL CHAT ERROR: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Read file and prepare data
    file_ext = filename.rsplit('.', 1)[1].lower()
    
    if file_ext in {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}:
        # Convert image to base64
        with open(filepath, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode()
        mime_type = f"image/{file_ext}" if file_ext != 'jpg' else "image/jpeg"
        return jsonify({
            'type': 'image',
            'name': filename,
            'data': f"data:{mime_type};base64,{image_data}"
        })
    elif file_ext == 'txt':
        # Read text file
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({
            'type': 'text',
            'name': filename,
            'content': content
        })
    else:
        return jsonify({'error': 'File type not supported for reading'}), 400

if __name__ == '__main__':
    # Use the port assigned by Render, or default to 5000 for local testing
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get("DEBUG", "False").lower() == "true")
