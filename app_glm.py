from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import os

app = Flask(__name__)

# Initialize OpenAI client with Nvidia API for GLM4.7
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

# Store conversation history
conversation_history = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant with advanced reasoning capabilities."
    }
]

@app.route('/')
def index():
    return render_template('glm.html')

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
        
        # Get response from GLM4.7 with streaming
        completion = client.chat.completions.create(
            model="z-ai/glm4.7",
            messages=conversation_history,
            temperature=1,
            top_p=1,
            max_tokens=16384,
            extra_body={"chat_template_kwargs": {"enable_thinking": True, "clear_thinking": False}},
            stream=True
        )
        
        reasoning_content = ""
        response_content = ""
        
        # Process streaming chunks
        for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            if len(chunk.choices) == 0 or getattr(chunk.choices[0], "delta", None) is None:
                continue
            
            delta = chunk.choices[0].delta
            
            # Collect reasoning
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_content += reasoning
            
            # Collect response content
            if getattr(delta, "content", None) is not None:
                response_content += delta.content
        
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
            "content": "You are a helpful AI assistant with advanced reasoning capabilities."
        }
    ]
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=False, port=5001, host='0.0.0.0')
