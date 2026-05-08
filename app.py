import os
import sys

# Monkey patch for gevent if needed, but we'll try gthread first
if os.environ.get("USE_GEVENT", "False").lower() == "true":
    try:
        import gevent.monkey
        gevent.monkey.patch_all()
    except ImportError:
        pass

from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from openai import OpenAI
import uuid
from datetime import datetime
import json
from werkzeug.utils import secure_filename
import base64

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
# -------------------------------------------------------------------
MODEL_CONFIGS = {
    # ── REASONING ───────────────────────────────────────────────────
    "DeepSeek Pro": {
        "model": "deepseek-ai/deepseek-v4-pro",
        "api_key": os.getenv("DEEPSEEK_PRO_KEY"),
        "temperature": 1,
        "max_tokens": 8192,
        "supports_vision": True,
        "category": "reasoning",
        "badge": "🔬",
        "description": "Flagship reasoning model with vision",
        "extra_body": {"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}}
    },
    "DeepSeek Flash": {
        "model": "deepseek-ai/deepseek-v4-flash",
        "api_key": os.getenv("DEEPSEEK_FLASH_KEY"),
        "temperature": 1,
        "max_tokens": 16384,
        "supports_vision": False,
        "category": "reasoning",
        "badge": "⚡",
        "description": "Ultra-fast DeepSeek with thinking",
        "extra_body": {"chat_template_kwargs": {"thinking": True}}
    },
    "Kimi K2": {
        "model": "moonshotai/kimi-k2-thinking",
        "api_key": os.getenv("KIMI_K2_KEY"),
        "temperature": 1,
        "max_tokens": 16384,
        "supports_vision": False,
        "category": "reasoning",
        "badge": "🧠",
        "description": "Deep math & logical reasoning",
        "extra_body": {"chat_template_kwargs": {"thinking": True}}
    },
    "Kimi 2.6": {
        "model": "moonshotai/kimi-k2.6",
        "api_key": os.getenv("KIMI_26_KEY"),
        "temperature": 1,
        "max_tokens": 16384,
        "supports_vision": False,
        "category": "creative",
        "badge": "✍️",
        "description": "Creative writing & storytelling",
        "extra_body": {"chat_template_kwargs": {"thinking": True}}
    },

    # ── VISION / MULTIMODAL ─────────────────────────────────────────
    "Llama 4 Maverick": {
        "model": "meta/llama-4-maverick-17b-128e-instruct",
        "api_key": os.getenv("DEEPSEEK_PRO_KEY"),   # reuse any valid key
        "temperature": 0.7,
        "max_tokens": 8192,
        "supports_vision": True,
        "category": "vision",
        "badge": "👁️",
        "description": "Meta's latest multimodal powerhouse"
    },
    "Llama Vision 90B": {
        "model": "meta/llama-3.2-90b-vision-instruct",
        "api_key": os.getenv("DEEPSEEK_PRO_KEY"),
        "temperature": 0.7,
        "max_tokens": 8192,
        "supports_vision": True,
        "category": "vision",
        "badge": "🖼️",
        "description": "90B vision model for image analysis"
    },
    "Phi-4 Multimodal": {
        "model": "microsoft/phi-4-multimodal-instruct",
        "api_key": os.getenv("DEEPSEEK_PRO_KEY"),
        "temperature": 0.7,
        "max_tokens": 4096,
        "supports_vision": True,
        "category": "vision",
        "badge": "🔭",
        "description": "Microsoft multimodal — fast & efficient"
    },

    # ── CODING ──────────────────────────────────────────────────────
    "GLM 4.7": {
        "model": "z-ai/glm4.7",
        "api_key": os.getenv("GLM_47_KEY"),
        "temperature": 1,
        "max_tokens": 16384,
        "supports_vision": False,
        "category": "coding",
        "badge": "💻",
        "description": "Excellent for code generation & debugging",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True, "clear_thinking": False}}
    },
    "Qwen3 Coder": {
        "model": "qwen/qwen3-coder-480b-a35b-instruct",
        "api_key": os.getenv("GLM_47_KEY"),
        "temperature": 0.7,
        "max_tokens": 16384,
        "supports_vision": False,
        "category": "coding",
        "badge": "🚀",
        "description": "480B coder — best for complex code tasks"
    },
    "Devstral 2": {
        "model": "mistralai/devstral-2-123b-instruct-2512",
        "api_key": os.getenv("GLM_47_KEY"),
        "temperature": 0.7,
        "max_tokens": 16384,
        "supports_vision": False,
        "category": "coding",
        "badge": "🛠️",
        "description": "Mistral's dev-focused 123B coding model"
    },
    "GLM 5": {
        "model": "z-ai/glm5",
        "api_key": os.getenv("GLM_47_KEY"),
        "temperature": 0.8,
        "max_tokens": 16384,
        "supports_vision": False,
        "category": "coding",
        "badge": "🌟",
        "description": "Latest GLM with enhanced capabilities"
    },

    # ── GENERAL / FAST ───────────────────────────────────────────────
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
    "Nemotron Super 49B": {
        "model": "nvidia/llama-3.3-nemotron-super-49b-v1",
        "api_key": os.getenv("KIMI_26_KEY"),
        "temperature": 0.7,
        "max_tokens": 8192,
        "supports_vision": False,
        "category": "general",
        "badge": "⚙️",
        "description": "NVIDIA's fine-tuned high-performance model"
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
    "Minimax M2.7": {
        "model": "minimaxai/minimax-m2.7",
        "api_key": os.getenv("MINIMAX_M27_KEY"),
        "temperature": 1,
        "max_tokens": 8192,
        "supports_vision": False,
        "category": "general",
        "badge": "📋",
        "description": "Great for summarization & translation"
    },
}

# Store conversations in memory
conversations = {}
current_conversation_id = None

def get_client(actual_label):
    config = MODEL_CONFIGS.get(actual_label, MODEL_CONFIGS["DeepSeek Pro"])
    return OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=config['api_key'])

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

    # --- UI / frontend / design -> Qwen3 Coder ---
    ui_keywords = [
        "ui", "ux", "design", "html", "css", "tailwind", "frontend", "interface",
        "component", "layout", "responsive", "mobile", "web app", "website",
        "button", "form", "modal", "navbar", "sidebar", "dashboard", "landing page"
    ]
    if any(kw in t for kw in ui_keywords):
        return "Qwen3 Coder", "🚀 Qwen3 Coder selected for UI/frontend tasks"

    # --- Complex code -> Qwen3 Coder, else GLM 4.7 ---
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
        return "GLM 4.7", "💻 GLM 4.7 selected for coding tasks"

    # --- Math / logic / deep reasoning -> Kimi K2 ---
    math_keywords = [
        "prove", "proof", "theorem", "calculate", "integral", "derivative", "equation",
        "algebra", "geometry", "calculus", "statistics", "probability", "matrix",
        "algorithm", "complexity", "big o", "dynamic programming", "recursion",
        "logic", "reasoning", "deduce", "infer", "step by step", "step-by-step",
        "solve", "evaluate", "explain why"
    ]
    if any(kw in t for kw in math_keywords):
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

    # --- Summarization / translation -> Minimax M2.7 ---
    summary_keywords = [
        "summarize", "summary", "translate", "translation", "paraphrase",
        "tldr", "key points", "main points", "extract", "overview", "brief"
    ]
    if any(kw in t for kw in summary_keywords):
        return "Minimax M2.7", "📋 Minimax M2.7 selected for summarization"

    # --- Analysis / research -> DeepSeek Pro ---
    analysis_keywords = [
        "analyze", "analysis", "compare", "comparison", "research", "explain",
        "difference between", "pros and cons", "advantages", "disadvantages",
        "review", "assess", "what is", "who is", "how to",
        "best way", "recommend", "suggest", "plan", "strategy"
    ]
    if any(kw in t for kw in analysis_keywords):
        return "DeepSeek Pro", "🔬 DeepSeek Pro selected for analysis & research"

    # --- Short simple prompts -> Llama 3.3 70B (fast & reliable) ---
    if word_count <= 10:
        return "Llama 3.3 70B", "🦙 Llama 3.3 70B selected for quick responses"

    # --- Default: DeepSeek Pro ---
    return "DeepSeek Pro", "🤖 DeepSeek Pro selected for general intelligence"


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

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages with model selection and streaming response"""
    global current_conversation_id
    
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
            message_content.append({"type": "text", "text": f"File: {file_data['name']}\nContent:\n{file_data['content']}"})
    
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

    config = MODEL_CONFIGS.get(actual_label, MODEL_CONFIGS["DeepSeek Pro"])
    client = get_client(actual_label)

    def generate():
        start_time = datetime.now()
        api_messages = []
        for m in conv['messages']:
            m_content = m["content"]
            # If current model doesn't support vision, convert list content to string
            if not config.get("supports_vision", False) and isinstance(m_content, list):
                text_parts = []
                for part in m_content:
                    if part["type"] == "text":
                        text_parts.append(part["text"])
                m_content = "\n".join(text_parts)
            
            api_messages.append({"role": m["role"], "content": m_content})

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

        full_response = ""
        reasoning_content = ""
        
        # Initial metadata - send immediately to keep connection alive
        # Add padding to flush Render/Nginx buffers (4KB is usually enough)
        yield ":" + " " * 4096 + "\n" 
        
        yield json.dumps({
            "status": "start",
            "modelUsed": actual_label,
            "selectionReason": selection_reason,
            "conversation_id": current_conversation_id
        }) + "\n"

        try:
            print(f"DEBUG: Starting API request to {config['model']}")
            completion = client.chat.completions.create(**params)
            print(f"DEBUG: API request started, waiting for chunks...")
            for chunk in completion:
                if not hasattr(chunk, "choices") or not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                chunk_data = {}
                
                if hasattr(delta, "content") and delta.content is not None:
                    full_response += delta.content
                    chunk_data["content"] = delta.content
                
                reasoning = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None) or getattr(delta, "thinking", None)
                if reasoning:
                    reasoning_content += reasoning
                    chunk_data["reasoning"] = reasoning
                
                if chunk_data:
                    yield json.dumps(chunk_data) + "\n"
                else:
                    # Send a heartbeat to keep connection alive if chunk is empty
                    yield "\n"
            
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

    return Response(generate(), mimetype='text/event-stream', headers={
        'X-Accel-Buffering': 'no',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
    })

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
