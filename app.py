import os
import sys
import json
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, session, redirect, url_for
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv
import database
import time
import traceback

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default_secret_key_for_render_123')

# Initialize database tables if they don't exist
try:
    database.init_db()
except Exception as e:
    print("DB Init Error:", e)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Model Categories and Costs
# -------------------------------------------------------------------
MODEL_CONFIGS = {
    # Reasoning
    "Nemotron 3 Super 120B": {"model": "nvidia/nemotron-3-super-120b-a12b", "api_key": os.getenv("NVIDIA_API_KEY_NEMOTRON_3"), "category": "reasoning", "cost": 7.0, "badge": "🏛️", "provider": "nvidia", "extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget": 16384}},
    "Nemotron 3 Nano Omni": {"model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", "api_key": os.getenv("NVIDIA_API_KEY_NEMOTRON_3"), "category": "reasoning", "cost": 3.5, "badge": "🔭", "provider": "nvidia", "extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget": 16384}},
    "DeepSeek V4 Pro": {"model": "deepseek-ai/deepseek-v4-pro", "api_key": os.getenv("NVIDIA_API_KEY_PRO"), "category": "reasoning", "cost": 5.0, "badge": "🔬", "provider": "nvidia", "extra_body": {"chat_template_kwargs": {"thinking": True}}},
    "GLM 5.1": {"model": "z-ai/glm-5.1", "api_key": os.getenv("NVIDIA_API_KEY_GLM51"), "category": "reasoning", "cost": 6.0, "badge": "🧠", "provider": "nvidia", "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
    "DeepSeek R1": {"model": "deepseek-ai/deepseek-r1", "api_key": os.getenv("KIMI_26_KEY"), "category": "reasoning", "cost": 4.0, "badge": "🧠", "provider": "nvidia", "extra_body": {"chat_template_kwargs": {"thinking": True}}},
    "Stockmark 2 100B": {"model": "stockmark/stockmark-2-100b-instruct", "api_key": os.getenv("NVIDIA_API_KEY_STOCKMARK"), "category": "reasoning", "cost": 4.5, "badge": "📈", "provider": "nvidia", "extra_body": {"chat_template_kwargs": {"thinking": True}}},
    
    # Image Gen
    "Stable Diffusion 3.5 Large": {"model": "stabilityai/stable-diffusion-3.5-large", "api_key": os.getenv("NVIDIA_API_KEY_SD35"), "category": "image_generation", "cost": 10.0, "badge": "🎨", "provider": "nvidia"},
    "Stable Diffusion XL": {"model": "stabilityai/stable-diffusion-xl", "api_key": os.getenv("KIMI_26_KEY"), "category": "image_generation", "cost": 8.0, "badge": "🖼️", "provider": "nvidia"},
    
    # Vision
    "Llama 4 Maverick": {"model": "meta/llama-4-maverick-17b-128e-instruct", "api_key": os.getenv("KIMI_26_KEY"), "category": "vision", "cost": 4.0, "badge": "👁️", "provider": "nvidia", "supports_vision": True},
    "Llama Vision 90B": {"model": "meta/llama-3.2-90b-vision-instruct", "api_key": os.getenv("NVIDIA_API_KEY_VISION"), "category": "vision", "cost": 5.0, "badge": "🖼️", "provider": "nvidia", "supports_vision": True},
    
    # Coding
    "Qwen 2.5 Coder": {"model": "qwen/qwen2.5-coder-32b-instruct", "api_key": os.getenv("KIMI_26_KEY"), "category": "coding", "cost": 3.0, "badge": "🚀", "provider": "nvidia"},
    "GLM 4.7": {"model": "meta/llama-3.3-70b-instruct", "api_key": os.getenv("KIMI_26_KEY"), "category": "coding", "cost": 2.5, "badge": "💻", "provider": "nvidia"},
    
    # Creative
    "Kimi 2.6": {"model": "meta/llama-3.3-70b-instruct", "api_key": os.getenv("KIMI_26_KEY"), "category": "creative", "cost": 2.0, "badge": "✍️", "provider": "nvidia", "extra_body": {"chat_template_kwargs": {"thinking": True}}},
    "Gemma 2 27B": {"model": "google/gemma-2-27b-it", "api_key": os.getenv("KIMI_26_KEY"), "category": "creative", "cost": 1.5, "badge": "💎", "provider": "nvidia"},
    
    # General
    "Sarvam M": {"model": "sarvamai/sarvam-m", "api_key": os.getenv("NVIDIA_API_KEY_SARVAM"), "category": "general", "cost": 1.0, "badge": "🇮🇳", "provider": "nvidia"},
    "Llama 3.3 70B": {"model": "meta/llama-3.3-70b-instruct", "api_key": os.getenv("KIMI_26_KEY"), "category": "general", "cost": 1.0, "badge": "🦙", "provider": "nvidia"},
    "Gemini 1.5 Flash (Free)": {"model": "gemini-1.5-flash", "api_key": os.getenv("GEMINI_API_KEY"), "category": "general", "cost": 0.0, "badge": "♊", "provider": "google"}
}

@app.route('/models')
def get_models():
    models = []
    for label, config in MODEL_CONFIGS.items():
        models.append({
            'label': label,
            'category': config.get('category'),
            'badge': config.get('badge'),
            'cost': config.get('cost'),
            'description': config.get('description', '')
        })
    return jsonify(models)

DEFAULT_MODELS = {
    "reasoning": "Nemotron 3 Super 120B",
    "image_generation": "Stable Diffusion 3.5 Large",
    "vision": "Llama 4 Maverick",
    "coding": "Qwen 2.5 Coder",
    "creative": "Kimi 2.6",
    "general": "Llama 3.3 70B"
}

def auto_select_model(text):
    """
    Intelligently select the best model based on prompt content.
    Returns model_label.
    """
    t = text.lower().strip()
    word_count = len(t.split())

    # --- Simple greetings -> Llama 3.3 70B (Fast) ---
    greet_patterns = ["hi", "hello", "hey", "how are you", "what's up", "whats up"]
    if word_count <= 5 and any(t.startswith(g) or t == g for g in greet_patterns):
        return "Llama 3.3 70B"

    # --- UI / frontend / design -> Qwen 2.5 Coder ---
    ui_keywords = ["ui", "ux", "design", "html", "css", "tailwind", "frontend", "interface"]
    if any(kw in t for kw in ui_keywords):
        return "Qwen 2.5 Coder"

    # --- Coding tasks ---
    code_keywords = ["code", "program", "script", "function", "debug", "error", "bug", "python", "javascript", "react", "sql", "api"]
    if any(kw in t for kw in code_keywords):
        return "Qwen 2.5 Coder"

    # --- Math / logic / deep reasoning -> Nemotron 3 Super 120B ---
    math_keywords = ["prove", "calculate", "integral", "equation", "algebra", "logic", "reasoning", "solve", "step by step"]
    if any(kw in t for kw in math_keywords):
        return "Nemotron 3 Super 120B"

    # --- Creative / writing -> Kimi 2.6 ---
    creative_keywords = ["write a story", "poem", "creative", "essay", "draft", "letter", "blog post"]
    if any(kw in t for kw in creative_keywords):
        return "Kimi 2.6"

    # --- Image Generation -> Stable Diffusion 3.5 Large ---
    image_keywords = ["generate an image", "draw a", "create an image", "make a picture", "visualize"]
    if any(kw in t for kw in image_keywords):
        return "Stable Diffusion 3.5 Large"

    # --- Analysis / research -> DeepSeek V4 Pro ---
    analysis_keywords = ["analyze", "analysis", "compare", "research", "explain", "pros and cons"]
    if any(kw in t for kw in analysis_keywords):
        return "DeepSeek V4 Pro"

    # --- Default: Llama 3.3 70B for general tasks ---
    return "Llama 3.3 70B"

def get_current_user():
    # Return real user_id if logged in
    uid = session.get('user_id')
    if uid:
        return uid
    
    # Return guest_id if it exists
    if 'guest_id' not in session:
        session['guest_id'] = f"guest_{uuid.uuid4().hex[:8]}"
    
    return session['guest_id']

def is_logged_in():
    return 'user_id' in session

@app.route('/')
def index():
    return render_template('agent_platform.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    identifier = data.get('identifier')
    if not identifier: return jsonify({'error': 'Required'}), 400
    
    user = database.get_user_by_identifier(identifier)
    if not user:
        user_id = database.create_user(identifier)
    else:
        user_id = user['id']
    
    # Migrate guest conversations if they exist
    guest_id = session.get('guest_id')
    if guest_id:
        conn = database.get_db_connection()
        conn.execute('UPDATE conversations SET user_id = ? WHERE user_id = ?', (user_id, guest_id))
        conn.commit()
        conn.close()
    
    session['user_id'] = user_id
    session['identifier'] = identifier
    # Clear guest_id after migration
    session.pop('guest_id', None)
    
    return jsonify({'success': True, 'user_id': user_id})

@app.route('/user/me')
def get_me():
    if not is_logged_in():
        return jsonify({
            'logged_in': False, 
            'identifier': 'Guest User', 
            'credits_balance': 100.0, # Default for guest
            'id': get_current_user()
        })
    
    uid = session.get('user_id')
    conn = database.get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (uid,)).fetchone()
    conn.close()
    return jsonify({'logged_in': True, 'identifier': user['identifier'], 'credits_balance': user['credits_balance'], 'id': user['id']})

@app.route('/conversations')
def get_conversations():
    uid = get_current_user()
    if not uid: return jsonify([])
    convs = database.get_user_conversations(uid)
    return jsonify([dict(c) for c in convs])

@app.route('/logout')
def logout():
    session.clear()
    return jsonify({'success': True})

@app.route('/conversation/new', methods=['POST'])
def new_conversation():
    uid = get_current_user()
    data = request.get_json() or {}
    cat = data.get('category', 'general')
    title = 'New Chat'
    conv_id = database.create_conversation(uid, title, cat)
    return jsonify({'id': conv_id, 'title': title})

@app.route('/conversation/delete/<conv_id>', methods=['POST'])
def delete_conversation(conv_id):
    database.delete_conversation(conv_id)
    return jsonify({'success': True})

@app.route('/conversation/rename', methods=['POST'])
def rename_conversation():
    data = request.get_json()
    conv_id = data.get('id')
    new_title = data.get('title')
    database.rename_conversation(conv_id, new_title)
    return jsonify({'success': True})

@app.route('/conversation/<conv_id>')
def get_conversation(conv_id):
    conn = database.get_db_connection()
    messages = conn.execute('SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC', (conv_id,)).fetchall()
    conn.close()
    return jsonify({'messages': [dict(m) for m in messages]})

@app.route('/chat', methods=['POST'])
def chat():
    if not is_logged_in():
        return jsonify({'error': 'Login required to send messages'}), 401
    
    uid = session.get('user_id')
    
    data = request.get_json()
    text = data.get('message')
    conv_id = data.get('conversation_id')
    category = data.get('category', 'general')
    
    # Select model
    selected_model = data.get('model')
    if selected_model and selected_model in MODEL_CONFIGS:
        model_label = selected_model
    elif category == 'auto':
        model_label = auto_select_model(text)
    else:
        model_label = DEFAULT_MODELS.get(category, "Llama 3.3 70B")
    
    config = MODEL_CONFIGS[model_label]
    
    # Save user message
    database.save_message(conv_id, 'user', text, category=category)
    
    # Auto-rename if this is the first message
    conn = database.get_db_connection()
    msg_count = conn.execute('SELECT COUNT(*) FROM messages WHERE conversation_id = ?', (conv_id,)).fetchone()[0]
    if msg_count == 1:
        # Title from first 30 chars
        new_title = (text[:30] + '...') if len(text) > 30 else text
        database.rename_conversation(conv_id, new_title)
    conn.close()
    
    start_time = time.time()
    try:
        if config.get('provider') == 'google':
            genai.configure(api_key=config['api_key'])
            model = genai.GenerativeModel(config['model'])
            
            # Convert history for Gemini
            conn = database.get_db_connection()
            history = conn.execute('SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC', (conv_id,)).fetchall()
            conn.close()
            
            gemini_history = []
            for m in history[:-1]: # All except the last user message
                role = 'user' if m['role'] == 'user' else 'model'
                gemini_history.append({"role": role, "parts": [m['content']]})
            
            chat_session = model.start_chat(history=gemini_history)
            response = chat_session.send_message(text)
            ai_response = response.text
            reasoning = None
            
            latency = round(time.time() - start_time, 2)
            tokens_in = 0 
            tokens_out = 0
            credits_used = 0.0
        else:
            client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=config['api_key'])
            
            # Get history from DB
            conn = database.get_db_connection()
            history = conn.execute('SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC', (conv_id,)).fetchall()
            conn.close()
            
            messages = [{"role": m['role'], "content": m['content']} for m in history]
            
            params = {
                "model": config["model"],
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 4096
            }
            if "extra_body" in config: params["extra_body"] = config["extra_body"]
            
            completion = client.chat.completions.create(**params)
            ai_response = completion.choices[0].message.content
            
            # Extract reasoning
            reasoning = getattr(completion.choices[0].message, "reasoning_content", None)
            
            # Fallback: some models put reasoning in <thought> tags in content
            if not reasoning and ai_response and "<thought>" in ai_response:
                import re
                match = re.search(r'<thought>(.*?)</thought>', ai_response, re.DOTALL)
                if match:
                    reasoning = match.group(1).strip()
                    ai_response = ai_response.replace(match.group(0), "").strip()
            
            latency = round(time.time() - start_time, 2)
            tokens_in = completion.usage.prompt_tokens
            tokens_out = completion.usage.completion_tokens
            credits_used = config['cost']
        
        # Save assistant message
        database.save_message(conv_id, 'assistant', ai_response, 
                             model_name=model_label, category=category,
                             tokens_in=tokens_in, tokens_out=tokens_out,
                             latency=latency, credits_used=credits_used)
        
        routing_steps = [
            "Intent Classifier analyzes prompt...",
            f"Orchestrator selected {model_label}",
            "Connecting to API Provider...",
            "Context analysis completed",
            "Response generation finished"
        ]
        
        return jsonify({
            'content': ai_response,
            'reasoning': reasoning,
            'modelUsed': model_label,
            'creditsUsed': credits_used,
            'latency': latency,
            'tokens': f"{tokens_in}/{tokens_out}",
            'routing_steps': routing_steps
        })
        
    except Exception as e:
        print("CHAT ERROR:", str(e))
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
