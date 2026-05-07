from flask import Flask, render_template, request, jsonify
from openai import OpenAI

app = Flask(__name__)

# Initialize OpenAI client with Nvidia API for DeepSeek v4-flash
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

# Store conversation history
conversation_history = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant with fast reasoning capabilities."
    }
]

@app.route('/')
def index():
    return render_template('deepseek_flash.html')

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
        
        # Get response from DeepSeek v4-flash
        completion = client.chat.completions.create(
            model="deepseek-ai/deepseek-v4-flash",
            messages=conversation_history,
            temperature=1,
            top_p=0.95,
            max_tokens=16384,
            extra_body={"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}},
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
            
            # Collect reasoning (try both naming conventions)
            reasoning = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
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
            "content": "You are a helpful AI assistant with fast reasoning capabilities."
        }
    ]
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=False, port=5004, host='0.0.0.0')
