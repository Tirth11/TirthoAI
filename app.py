import os
import sys
import json
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, session, redirect, url_for
from openai import OpenAI
from dotenv import load_dotenv
import database
import time
import traceback
import PyPDF2
from docx import Document
import pandas as pd

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.pdf':
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif ext == '.docx':
            doc = Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
            return df.to_string()
        elif ext == '.csv':
            df = pd.read_csv(file_path)
            return df.to_string()
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        return ""
    except Exception as e:
        print(f"Error parsing {ext}: {str(e)}")
        return ""

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
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Model Categories and Costs
# -------------------------------------------------------------------
MODEL_CONFIGS = {
    # Reasoning
    "Nemotron 3 Super 120B": {"model": "nvidia/nemotron-3-super-120b-a12b", "api_key": os.getenv("NVIDIA_API_KEY_NEMOTRON_3"), "category": "reasoning", "cost": 10.0, "badge": "🏛️", "provider": "nvidia", "extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget": 16384}},
    "DeepSeek V4 Pro": {"model": "deepseek-ai/deepseek-v4-pro", "api_key": os.getenv("NVIDIA_API_KEY_PRO"), "category": "reasoning", "cost": 10.0, "badge": "🔬", "provider": "nvidia", "extra_body": {"chat_template_kwargs": {"thinking": True}}},
    "GLM 5.1": {"model": "z-ai/glm-5.1", "api_key": os.getenv("NVIDIA_API_KEY_GLM51"), "category": "reasoning", "cost": 10.0, "badge": "🧠", "provider": "nvidia", "extra_body": {"chat_template_kwargs": {"enable_thinking": True, "clear_thinking": False}}},
    "DeepSeek R1": {"model": "deepseek-ai/deepseek-r1", "api_key": os.getenv("KIMI_26_KEY"), "category": "reasoning", "cost": 10.0, "badge": "🧠", "provider": "nvidia", "extra_body": {"chat_template_kwargs": {"thinking": True}}},
    "Stockmark 2 100B": {"model": "stockmark/stockmark-2-100b-instruct", "api_key": os.getenv("NVIDIA_API_KEY_STOCKMARK"), "category": "reasoning", "cost": 10.0, "badge": "📈", "provider": "nvidia", "extra_body": {"chat_template_kwargs": {"thinking": True}}},
    
    # Vision
    "Llama 3.2 90B Vision": {"model": "meta/llama-3.2-90b-vision-instruct", "api_key": os.getenv("NVIDIA_API_KEY_VISION"), "category": "vision", "cost": 5.0, "badge": "🖼️", "provider": "nvidia", "supports_vision": True},
    
    # Coding
    "Qwen 3 Coder 480B": {"model": "qwen/qwen3-coder-480b-a35b-instruct", "api_key": os.getenv("NVIDIA_API_KEY_STEP_GEMMA"), "category": "coding", "cost": 15.0, "badge": "💻", "provider": "nvidia"},
    "Llama 3.3 70B (Coding)": {"model": "meta/llama-3.3-70b-instruct", "api_key": os.getenv("KIMI_26_KEY"), "category": "coding", "cost": 15.0, "badge": "💻", "provider": "nvidia"},
    
    # Creative
    "Kimi 2.6": {"model": "moonshotai/kimi-k2.6", "api_key": os.getenv("NVIDIA_API_KEY_KIMI_K26"), "category": "creative", "cost": 2.0, "badge": "✍️", "provider": "nvidia", "extra_body": {"chat_template_kwargs": {"thinking": True}}},
    "Gemma 2 27B": {"model": "google/gemma-2-27b-it", "api_key": os.getenv("KIMI_26_KEY"), "category": "creative", "cost": 2.0, "badge": "💎", "provider": "nvidia"},
    
    # General
    "Sarvam M": {"model": "sarvamai/sarvam-m", "api_key": os.getenv("NVIDIA_API_KEY_SARVAM"), "category": "general", "cost": 1.0, "badge": "🇮🇳", "provider": "nvidia"},
    "Minimax m2.7": {"model": "minimaxai/minimax-m2.7", "api_key": os.getenv("NVIDIA_API_KEY_MINIMAX"), "category": "general", "cost": 1.0, "badge": "🌀", "provider": "nvidia"},
    "Step 3.5 Flash": {"model": "stepfun-ai/step-3.5-flash", "api_key": os.getenv("NVIDIA_API_KEY_STEP_GEMMA"), "category": "general", "cost": 1.0, "badge": "⚡", "provider": "nvidia"},
    "Llama 3.3 70B": {"model": "meta/llama-3.3-70b-instruct", "api_key": os.getenv("KIMI_26_KEY"), "category": "general", "cost": 1.0, "badge": "🦙", "provider": "nvidia"}
}

def get_available_models():
    """Return only models that have valid API keys configured."""
    available = {}
    for label, config in MODEL_CONFIGS.items():
        if config.get('api_key'):  # Only include if API key exists
            available[label] = config
    return available

@app.route('/models')
def get_models():
    models = []
    for label, config in get_available_models().items():
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
    "vision": "Llama 3.2 90B Vision",
    "coding": "Qwen 3 Coder 480B",
    "creative": "Kimi 2.6",
    "general": "Llama 3.3 70B"
}

def auto_select_model(text):
    """
    Intelligently select the best model based on prompt content.
    Only picks from models that have valid API keys.
    Returns model_label.
    """
    available = get_available_models()
    if not available:
        return None  # No models available
    
    t = (text or "").lower().strip()
    word_count = len(t.split()) if t else 0

    greet_patterns = ["hi", "hello", "hey", "how are you", "what's up", "whats up", "helo", "hy"]
    if word_count <= 5 and any(t.startswith(g) or t == g for g in greet_patterns):
        # Pick a general model if available
        for label, cfg in available.items():
            if cfg.get('category') == 'general':
                return label
        return next(iter(available))  # Fallback to first available

    # Keywords mapping
    mapping = {
        'coding': ['code', 'python', 'javascript', 'html', 'css', 'react', 'debug', 'function', 'class', 'develop', 'program'],
        'vision': ['analyze', 'look', 'see', 'describe', 'vision', 'image', 'photo', 'picture'],
        'reasoning': ['think', 'reason', 'solve', 'complex', 'math', 'logic', 'deep', 'explain', 'why'],
        'creative': ['story', 'poem', 'write', 'creative', 'blog', 'essay', 'draft']
    }

    scores = {cat: 0 for cat in mapping}
    for cat, kws in mapping.items():
        for kw in kws:
            if kw in t:
                scores[cat] += 1
    
    best_category = max(scores, key=scores.get)
    max_score = scores[best_category]

    # Find best available model for the category
    target_category = best_category if max_score >= 2 else 'general'
    
    for label, cfg in available.items():
        if cfg.get('category') == target_category:
            return label
    
    # Fallback: return first available model
    return next(iter(available))

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
            'credits_balance': 100.0,
            'id': get_current_user()
        })
    
    uid = session.get('user_id')
    
    # Check and Reset Credits if 24h passed
    conn = database.get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (uid,)).fetchone()
    
    if not user:
        conn.close()
        return jsonify({
            'logged_in': False, 
            'identifier': 'Guest User', 
            'credits_balance': 100.0,
            'id': get_current_user()
        })
    
    import datetime
    now = datetime.datetime.now()
    
    # Defensive check for last_credit_allocation
    last_reset_val = user['last_credit_allocation']
    if not last_reset_val:
        last_reset = now
        conn.execute('UPDATE users SET last_credit_allocation = ? WHERE id = ?', 
                     (now.strftime('%Y-%m-%d %H:%M:%S'), uid))
        conn.commit()
    else:
        try:
            last_reset = datetime.datetime.strptime(last_reset_val, '%Y-%m-%d %H:%M:%S')
        except:
            last_reset = now
    
    reset_time = last_reset + datetime.timedelta(hours=24)
    if now >= reset_time:
        conn.execute('UPDATE users SET credits_balance = 100.0, last_credit_allocation = ? WHERE id = ?', 
                     (now.strftime('%Y-%m-%d %H:%M:%S'), uid))
        conn.commit()
        user = conn.execute('SELECT * FROM users WHERE id = ?', (uid,)).fetchone()
        reset_time = now + datetime.timedelta(hours=24)

    conn.close()
    
    return jsonify({
        'logged_in': True, 
        'identifier': user['identifier'], 
        'credits_balance': user['credits_balance'], 
        'id': user['id'],
        'next_reset': reset_time.strftime('%Y-%m-%d %H:%M:%S')
    })

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'docx', 'xlsx', 'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        from werkzeug.utils import secure_filename
        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Determine if it's an image
        is_image = any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp'])
        
        return jsonify({
            'success': True, 
            'filename': filename, 
            'url': url_for('uploaded_file', filename=filename),
            'is_image': is_image
        })
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(UPLOAD_FOLDER, filename)

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
    
    formatted_messages = []
    import json
    for m in messages:
        msg = dict(m)
        if msg.get('files'):
            try:
                msg['files'] = json.loads(msg['files'])
            except:
                msg['files'] = []
        formatted_messages.append(msg)
        
    return jsonify({'messages': formatted_messages})

@app.route('/chat', methods=['POST'])
def chat():
    if not is_logged_in():
        return jsonify({'error': 'Login required to send messages'}), 401
    
    uid = session.get('user_id')
    
    try:
        data = request.get_json()
        text = data.get('message') or ""
        conv_id = data.get('conversation_id')
        # Select model and category safely
        selected_model = data.get('model')
        if selected_model and not isinstance(selected_model, str):
            selected_model = str(selected_model)
            
        category = data.get('category', 'general')
        if category and not isinstance(category, str):
            category = str(category)
        
        # --- Simple Spelling Correction / Cleanup ---
        def cleanup_text(t):
            corrections = {
                "moneky": "monkey",
                "helo": "hello",
                "hy": "hi",
                "waht": "what",
                "teh": "the",
                "generat": "generate",
                "imge": "image",
                "picutre": "picture"
            }
            if not t: return ""
            words = t.split()
            fixed_words = [corrections.get(w.lower(), w) for w in words]
            return " ".join(fixed_words)
            
        clean_text = cleanup_text(text)
        
        if selected_model and selected_model in MODEL_CONFIGS:
            # Verify the selected model has an API key
            if not MODEL_CONFIGS[selected_model].get('api_key'):
                return jsonify({'error': f'Model {selected_model} is not available (API key missing). Please select another model.'}), 400
            model_label = selected_model
        elif category == 'auto':
            model_label = auto_select_model(clean_text)
        else:
            # Find an available model in the requested category
            available = get_available_models()
            model_label = None
            for label, cfg in available.items():
                if cfg.get('category') == category:
                    model_label = label
                    break
            if not model_label:
                # Fallback to any available model
                model_label = auto_select_model(clean_text)
        
        if not model_label:
            return jsonify({'error': 'No AI models are currently available. Please check API key configuration.'}), 503
        
        if model_label not in MODEL_CONFIGS:
            return jsonify({'error': f'Model {model_label} not configured'}), 400
            
        config = MODEL_CONFIGS[model_label]
        
        # Get files from request early
        files = data.get('files', [])

        # Calculate dynamic cost
        actual_cost = config['cost']
        if len(clean_text) > 500: # Long answer / detailed answer
            actual_cost = max(actual_cost, 2.0)
        if files: # File upload analysis
            actual_cost = max(actual_cost, 5.0)
        
        # Check Credits (Logged in users only)
        if is_logged_in():
            uid = session.get('user_id')
            conn = database.get_db_connection()
            user = conn.execute('SELECT credits_balance, last_credit_allocation FROM users WHERE id = ?', (uid,)).fetchone()
            conn.close()
            
            if user['credits_balance'] < actual_cost:
                import datetime
                now = datetime.datetime.now()
                last_reset_val = user['last_credit_allocation']
                try:
                    last_reset = datetime.datetime.strptime(last_reset_val, '%Y-%m-%d %H:%M:%S') if last_reset_val else now
                except:
                    last_reset = now
                
                reset_time = last_reset + datetime.timedelta(hours=24)
                
                msg = f"Your credits have expired. Kindly wait until {reset_time.strftime('%I:%M %p %Y-%m-%d')} to receive your next 100 credits."
                return jsonify({'error': msg, 'expired': True, 'reset_at': reset_time.strftime('%Y-%m-%d %H:%M:%S')}), 402

        # Validate API Key
        if not config.get('api_key'):
            return jsonify({'error': f'API key for {model_label} is missing. Please check your .env file.'}), 500

        # Save user message
        database.save_message(conv_id, 'user', clean_text, category=category, files=files)
        
        # --- File Content Extraction ---
        context_from_files = ""
        for f_info in files:
            if not f_info.get('is_image'):
                file_path = os.path.join(UPLOAD_FOLDER, f_info['filename'])
                if os.path.exists(file_path):
                    content = extract_text(file_path)
                    if content:
                        context_from_files += f"\n--- File Content: {f_info['filename']} ---\n{content}\n"
        
        prompt_with_context = clean_text
        if context_from_files:
            prompt_with_context = f"User provided documents:\n{context_from_files}\n\nUser Message: {clean_text}"
        
        # Auto-rename if this is the first message
        conn = database.get_db_connection()
        msg_count = conn.execute('SELECT COUNT(*) FROM messages WHERE conversation_id = ?', (conv_id,)).fetchone()[0]
        if msg_count == 1:
            # Title from first 30 chars
            new_title = (clean_text[:30] + '...') if len(clean_text) > 30 else clean_text
            database.rename_conversation(conv_id, new_title)
        conn.close()
        
        # --- Prompt Refinement (Internal) ---
        # If the prompt is messy or short, we use a fast model to 'clean' it for the target model.
        # This helps with the user's "not correcting spelling mistakes" issue.
        refined_text = prompt_with_context
        if category != 'image_generation' and len(text) < 100:
             # Skip for image gen to keep prompt original, but for others we can clean up
             pass # In a production app, we might call a fast model here to fix "moneky" to "monkey"
             # For now, let's just make the keyword matching more robust (already done).

        start_time = time.time()
        try:
            client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=config['api_key'])
            
            # Get history from DB
            conn = database.get_db_connection()
            history = conn.execute('SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC', (conv_id,)).fetchall()
            conn.close()
            
            messages = [{"role": m['role'], "content": m['content']} for m in history]
            # Replace the last message (the current user input) with the one containing context
            if messages and messages[-1]['role'] == 'user':
                messages[-1]['content'] = refined_text
            
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
            credits_used = actual_cost
        except Exception as api_error:
            print(f"API ERROR ({model_label}):", str(api_error))
            return jsonify({'error': f'Agent {model_label} failed: {str(api_error)}'}), 500
        
        # Save assistant message
        database.save_message(conv_id, 'assistant', ai_response, 
                             model_name=model_label, category=category,
                             tokens_in=tokens_in, tokens_out=tokens_out,
                             latency=latency, credits_used=credits_used)
        
        # Get updated balance
        new_balance = 0.0
        if is_logged_in():
            conn = database.get_db_connection()
            new_user_state = conn.execute('SELECT credits_balance FROM users WHERE id = ?', (session.get('user_id'),)).fetchone()
            new_balance = new_user_state['credits_balance']
            conn.close()
        else:
            new_balance = 100.0

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
            'routing_steps': routing_steps,
            'new_balance': new_balance
        })
        
    except Exception as e:
        print("CHAT ERROR:", str(e))
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    database.init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
