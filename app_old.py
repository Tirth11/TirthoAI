from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import os
import uuid
from datetime import datetime
import json
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Initialize OpenAI client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
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
    return render_template('deepseek_enhanced.html')

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
        'id': conv['id'],
        'title': conv['title'],
        'messages': messages
    })

@app.route('/chat', methods=['POST'])
def chat():
    try:
        global current_conversation_id
        
        data = request.form
        user_message = data.get('message', '').strip()
        
        if not current_conversation_id:
            current_conversation_id = create_conversation()
        
        # Handle file uploads
        files_data = []
        if 'files' in request.files:
            files = request.files.getlist('files')
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    
                    # Read file and encode for API
                    with open(filepath, 'rb') as f:
                        file_content = f.read()
                    
                    # Check if it's an image
                    if filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}:
                        # Encode image to base64
                        encoded = base64.b64encode(file_content).decode()
                        files_data.append({
                            "type": "image",
                            "name": filename,
                            "data": encoded,
                            "media_type": f"image/{filename.rsplit('.', 1)[1].lower()}"
                        })
                    else:
                        # For text files
                        files_data.append({
                            "type": "file",
                            "name": filename,
                            "content": file_content.decode('utf-8', errors='ignore')
                        })
        
        # Build message content
        content = []
        
        # Add images if any
        for file_data in files_data:
            if file_data['type'] == 'image':
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{file_data['media_type']};base64,{file_data['data']}"
                    }
                })
        
        # Add text message
        message_text = f"{user_message}\n\n"
        for file_data in files_data:
            if file_data['type'] == 'file':
                message_text += f"[Attached file: {file_data['name']}]\n{file_data['content']}\n\n"
        
        content.append({
            "type": "text",
            "text": message_text if message_text.strip() else user_message
        })
        
        if not user_message and not files_data:
            return jsonify({'error': 'Empty message and no files'}), 400
        
        # Add user message to conversation
        conversations[current_conversation_id]['messages'].append({
            "role": "user",
            "content": content if len(content) > 1 else (content[0]['text'] if content else user_message)
        })
        
        # Update conversation title from first message
        if len(conversations[current_conversation_id]['messages']) == 2:
            title = user_message[:50] if user_message else "File Analysis"
            conversations[current_conversation_id]['title'] = title
        
        # Get response from DeepSeek
        completion = client.chat.completions.create(
            model="deepseek-ai/deepseek-v4-pro",
            messages=conversations[current_conversation_id]['messages'],
            temperature=1,
            top_p=0.95,
            max_tokens=16384,
            extra_body={"chat_template_kwargs": {"thinking": False}},
            stream=True
        )
        
        response_content = ""
        for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            if chunk.choices and chunk.choices[0].delta.content is not None:
                response_content += chunk.choices[0].delta.content
        
        # Add assistant response to conversation
        conversations[current_conversation_id]['messages'].append({
            "role": "assistant",
            "content": response_content
        })
        
        return jsonify({
            'success': True,
            'response': response_content,
            'conv_id': current_conversation_id
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

if __name__ == '__main__':
    app.run(debug=False, port=5000, host='0.0.0.0')
