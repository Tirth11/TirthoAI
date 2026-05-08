from flask import Flask, render_template, request, jsonify
import requests
import json

app = Flask(__name__)

# Kimi K2.6 configuration
KIMI_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
KIMI_API_KEY = "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"

# Store conversation history
conversation_history = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant with advanced thinking capabilities."
    }
]

@app.route('/')
def index():
    return render_template('kimi26.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Add user message to history
        conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        headers = {
            "Authorization": f"Bearer {KIMI_API_KEY}",
            "Accept": "text/event-stream"
        }
        
        payload = {
            "model": "meta/llama-3.3-70b-instruct",
            "messages": conversation_history,
            "max_tokens": 16384,
            "temperature": 1.00,
            "top_p": 1.00,
            "stream": True,
            "chat_template_kwargs": {"thinking": True},
        }
        
        response = requests.post(KIMI_API_URL, headers=headers, json=payload, stream=True)
        
        thinking_content = ""
        response_content = ""
        
        # Process streaming response
        for line in response.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    try:
                        json_str = line_str[6:]
                        if json_str.strip():
                            data_obj = json.loads(json_str)
                            
                            # Extract thinking and content from choices
                            if 'choices' in data_obj and len(data_obj['choices']) > 0:
                                choice = data_obj['choices'][0]
                                if 'delta' in choice:
                                    delta = choice['delta']
                                    
                                    # Get thinking content
                                    if 'thinking' in delta:
                                        thinking_content += delta['thinking']
                                    
                                    # Get text content
                                    if 'content' in delta:
                                        response_content += delta['content']
                    except json.JSONDecodeError:
                        pass
        
        # Add assistant response to history
        conversation_history.append({
            "role": "assistant",
            "content": response_content
        })
        
        return jsonify({
            'success': True,
            'response': response_content,
            'thinking': thinking_content
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_history():
    global conversation_history
    conversation_history = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant with advanced thinking capabilities."
        }
    ]
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=False, port=5003, host='0.0.0.0')
