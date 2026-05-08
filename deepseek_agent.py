from flask import Flask, render_template, request, jsonify, send_from_directory
from openai import OpenAI
import os
import uuid
from datetime import datetime
import json
from werkzeug.utils import secure_filename
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'docx', 'csv', 'xlsx', 'json'}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
MAX_TEXT_CHARS = 12000
MAX_RESPONSE_TOKENS = 2048
UI_VERSION = "2026-05-07"
MODEL_MAP = {
    "auto": "auto",
    "general": "deepseek-ai/deepseek-v4-pro",
    "fast": "deepseek-ai/deepseek-v4-flash",
    "doc_qa": "moonshotai/kimi-k2.6",
    "legal": "deepseek-ai/deepseek-v4-pro",
    "compare": "moonshotai/kimi-k2.6",
    "summary": "moonshotai/kimi-k2.6",
    "report": "z-ai/glm4.7",
    "email": "deepseek-ai/deepseek-v4-flash",
    "code": "deepseek-ai/deepseek-v4-pro",
    "deepseek_pro": "deepseek-ai/deepseek-v4-pro",
    "deepseek_flash": "deepseek-ai/deepseek-v4-flash",
    "glm": "z-ai/glm4.7",
    "kimi": "moonshotai/kimi-k2-thinking",
    "kimi_26": "moonshotai/kimi-k2.6",
    "minimax": "minimaxai/minimax-m2.7",
    "llama": "meta/llama-3.3-70b-instruct",
}
MODEL_LABELS = {
    "deepseek-ai/deepseek-v4-pro": "DeepSeek Pro",
    "deepseek-ai/deepseek-v4-flash": "DeepSeek Flash",
    "z-ai/glm4.7": "GLM 4.7",
    "moonshotai/kimi-k2-thinking": "Kimi K2",
    "moonshotai/kimi-k2.6": "Kimi 2.6",
    "minimaxai/minimax-m2.7": "Minimax M2.7",
    "meta/llama-3.3-70b-instruct": "Llama 3.3 70B",
}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Initialize OpenAI client with DeepSeek v4-pro
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("DEEPSEEK_PRO_KEY")
)

# Store conversations in memory
conversations = {}
current_conversation_id = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_conversation():
    conv_id = str(uuid.uuid4())[:8]
    conversations[conv_id] = {
        "id": conv_id,
        "title": "New Chat",
        "created_at": datetime.now().isoformat(),
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful coding assistant with vision capabilities to analyze images and files."
            }
        ]
    }
    return conv_id

@app.route('/')
def index():
    response = render_template('index.html')
    return response

@app.route('/version', methods=['GET'])
def version():
    return jsonify({
        "ui_version": UI_VERSION,
        "models": MODEL_MAP,
        "model_labels": MODEL_LABELS
    })

def auto_route(message_text, files_meta):
    lowered = (message_text or "").lower()
    has_files = len(files_meta) > 0
    file_exts = {f.get("ext") for f in files_meta if f.get("ext")}

    if has_files and (file_exts & {"pdf", "docx", "xlsx", "csv", "json"}):
        return "moonshotai/kimi-k2.6", "Long document detected"
    if any(k in lowered for k in ["code", "bug", "stack", "trace", "compile", "error", "refactor"]):
        return "deepseek-ai/deepseek-v4-pro", "Coding or debugging task"
    if any(k in lowered for k in ["reason", "prove", "logic", "theorem", "analysis"]):
        return "z-ai/glm4.7", "Deep reasoning request"
    if len(lowered) < 200 and not has_files:
        return "deepseek-ai/deepseek-v4-flash", "Short, fast query"
    return "meta/llama-3.3-70b-instruct", "General task (Stable)"

@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

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

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages with file uploads"""
    global current_conversation_id
    
    # Handle both JSON and multipart/form-data
    if request.is_json:
        data = request.get_json()
        message_text = data.get('message', '').strip()
        files_data = data.get('files', [])
        mode = (data.get('mode') or 'auto').strip()
        model_override = (data.get('model') or '').strip()
    else:
        # Standard multipart/form-data (used by current deepseek_enhanced.html)
        message_text = request.form.get('message', '').strip()
        files_data = [] # Multipart files would be in request.files
        mode = (request.form.get('mode') or 'auto').strip()
        model_override = (request.form.get('model') or '').strip()
    
    # Handling file uploads if they are in request.files (as done by FormData)
    uploaded_files = request.files.getlist('files')
    processed_files = []
    files_meta = []
    
    for file in uploaded_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            file_ext = filename.rsplit('.', 1)[1].lower()
            files_meta.append({"name": filename, "ext": file_ext})
            if file_ext in {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}:
                with open(filepath, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode()
                processed_files.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{file_ext};base64,{image_data}"}
                })
            else:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                if len(content) > MAX_TEXT_CHARS:
                    content = content[:MAX_TEXT_CHARS] + "\n\n[Truncated for speed]"
                processed_files.append({
                    "type": "text",
                    "text": f"File: {filename}\nContent:\n{content}"
                })

    if not message_text and not processed_files and not files_data:
        return jsonify({'error': 'No message or files provided'}), 400
    
    # Create new conversation if none exists
    if current_conversation_id is None:
        current_conversation_id = create_conversation()
    
    # Build message content
    message_content = []
    
    # Add text
    if message_text:
        message_content.append({
            "type": "text",
            "text": message_text
        })
    
    # Add processed files
    message_content.extend(processed_files)
    
    # Add files from JSON data (legacy/compatibility)
    for file_data in files_data:
        if file_data['type'] == 'image':
            message_content.append({
                "type": "image_url",
                "image_url": {"url": file_data['data']}
            })
        elif file_data['type'] == 'text':
            message_content.append({
                "type": "text",
                "text": f"File: {file_data['name']}\nContent:\n{file_data['content']}"
            })
    
    # Add to conversation
    conv = conversations[current_conversation_id]
    conv['messages'].append({
        "role": "user",
        "content": message_content
    })
    
    try:
        auto_reason = None
        if model_override:
            model_id = MODEL_MAP.get(model_override, model_override)
        elif mode == "auto":
            model_id, auto_reason = auto_route(message_text, files_meta)
        else:
            model_id = MODEL_MAP.get(mode, "deepseek-ai/deepseek-v4-pro")

        completion = client.chat.completions.create(
            model=model_id,
            messages=conv['messages'],
            temperature=0.7,
            top_p=0.9,
            max_tokens=MAX_RESPONSE_TOKENS,
            extra_body={"chat_template_kwargs": {"thinking": False}},
            stream=False
        )
        
        # Collect response
        full_response = ""
        if getattr(completion, "choices", None):
            full_response = completion.choices[0].message.content or ""
        
        # Add response to conversation
        conv['messages'].append({
            "role": "assistant",
            "content": full_response
        })
        
        # Update conversation title if first message
        if len([m for m in conv['messages'] if m['role'] != 'system']) == 2:  # user + assistant
            title_text = message_text if message_text else "New Chat"
            conv['title'] = title_text[:50] + ("..." if len(title_text) > 50 else "")
        
        return jsonify({
            'response': full_response,
            'conversation_id': current_conversation_id,
            'conv_id': current_conversation_id, # For JS compatibility
            'model_label': MODEL_LABELS.get(model_id, model_id),
            'model_id': model_id,
            'auto_reason': auto_reason
        })
    
    except Exception as e:
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
    app.run(host='0.0.0.0', port=5005, debug=False)
