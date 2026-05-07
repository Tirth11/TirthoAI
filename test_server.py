from flask import Flask
app = Flask(__name__)
@app.route('/')
def hello():
    return "Hello from TirthoAI! If you see this, the server is reachable."
if __name__ == '__main__':
    app.run(port=5001, host='0.0.0.0')
