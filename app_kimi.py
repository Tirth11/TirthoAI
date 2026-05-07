from flask import Flask, render_template, request, jsonify
from openai import OpenAI

app = Flask(__name__)

# Initialize OpenAI client with Nvidia API for Kimi K2
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

# Store conversation history
conversation_history = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant with deep thinking and reasoning capabilities."
    }
]

@app.route('/')
def index():
    return render_template('kimi.html')

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
        
        # Get response from Kimi K2 with streaming
        completion = client.chat.completions.create(
            model="moonshotai/kimi-k2-thinking",
            messages=conversation_history,
            temperature=1,
            top_p=0.9,
            max_tokens=16384,
            stream=True
        )
        
        reasoning_content = ""
        response_content = ""
        
        # Process streaming chunks
        for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            
            delta = getattr(chunk.choices[0], "delta", None)
            if not delta:
                continue
            
            # Collect reasoning
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_content += reasoning
            
            # Collect response content
            content = getattr(delta, "content", None)
            if content:
                response_content += content
        
        # Add assistant response to history
        conversation_history.append({
            "role": "assistant",
            "content": response_content
        })
        
        return jsonify({
            'success': True,
            'response': response_content,
            'reasoning': reasoning_content
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_history():
    global conversation_history
    conversation_history = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant with deep thinking and reasoning capabilities."
        }
    ]
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=False, port=5002, host='0.0.0.0')
