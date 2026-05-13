import sqlite3
import os
import uuid
from datetime import datetime

DATABASE_NAME = 'agent_platform.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        identifier TEXT UNIQUE, -- Email or Phone
        credits_balance REAL DEFAULT 100.0,
        last_credit_allocation TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Conversations table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        user_id TEXT,
        title TEXT DEFAULT 'New Chat',
        category TEXT,
        last_model TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')

    # Messages table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT,
        role TEXT, -- user, assistant
        content TEXT,
        model_name TEXT,
        category TEXT,
        tokens_in INTEGER DEFAULT 0,
        tokens_out INTEGER DEFAULT 0,
        latency REAL DEFAULT 0.0,
        credits_used REAL DEFAULT 0.0,
        files TEXT, -- JSON array of file objects
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (conversation_id) REFERENCES conversations (id)
    )
    ''')

    conn.commit()
    
    # Migration: Add columns if they don't exist
    try:
        cursor.execute('ALTER TABLE users ADD COLUMN last_credit_allocation TIMESTAMP')
        cursor.execute('UPDATE users SET last_credit_allocation = CURRENT_TIMESTAMP WHERE last_credit_allocation IS NULL')
    except sqlite3.OperationalError:
        pass
        
    try:
        cursor.execute('ALTER TABLE messages ADD COLUMN files TEXT')
    except sqlite3.OperationalError:
        pass
        
    conn.commit()
    conn.close()
    print("Database initialized successfully.")

# Helper functions
def get_user_by_identifier(identifier):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE identifier = ?', (identifier,)).fetchone()
    conn.close()
    return user

def create_user(identifier):
    conn = get_db_connection()
    user_id = str(uuid.uuid4())
    conn.execute('INSERT INTO users (id, identifier, last_login) VALUES (?, ?, ?)',
                 (user_id, identifier, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    return user_id

def get_user_conversations(user_id):
    conn = get_db_connection()
    convs = conn.execute('''
        SELECT *, 
        CASE 
            WHEN date(created_at) = date('now') THEN 'Today'
            WHEN date(created_at) = date('now', '-1 day') THEN 'Yesterday'
            ELSE 'Previous'
        END as time_period
        FROM conversations 
        WHERE user_id = ? 
        AND id IN (SELECT conversation_id FROM messages)
        ORDER BY created_at DESC
    ''', (user_id,)).fetchall()
    conn.close()
    return convs

def delete_conversation(conv_id):
    conn = get_db_connection()
    conn.execute('DELETE FROM messages WHERE conversation_id = ?', (conv_id,))
    conn.execute('DELETE FROM conversations WHERE id = ?', (conv_id,))
    conn.commit()
    conn.close()

def rename_conversation(conv_id, new_title):
    conn = get_db_connection()
    conn.execute('UPDATE conversations SET title = ? WHERE id = ?', (new_title, conv_id))
    conn.commit()
    conn.close()

def create_conversation(user_id, title='New Chat', category=None):
    conn = get_db_connection()
    conv_id = str(uuid.uuid4())[:8]
    conn.execute('INSERT INTO conversations (id, user_id, title, category) VALUES (?, ?, ?, ?)',
                 (conv_id, user_id, title, category))
    conn.commit()
    conn.close()
    return conv_id

def save_message(conv_id, role, content, model_name=None, category=None, tokens_in=0, tokens_out=0, latency=0.0, credits_used=0.0, files=None):
    conn = get_db_connection()
    msg_id = str(uuid.uuid4())
    import json
    files_json = json.dumps(files) if files else None
    
    conn.execute('''
        INSERT INTO messages (id, conversation_id, role, content, model_name, category, tokens_in, tokens_out, latency, credits_used, files)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (msg_id, conv_id, role, content, model_name, category, tokens_in, tokens_out, latency, credits_used, files_json))
    
    # Deduct credits if assistant
    if role == 'assistant' and credits_used > 0:
        cursor = conn.cursor()
        user_id = cursor.execute('SELECT user_id FROM conversations WHERE id = ?', (conv_id,)).fetchone()['user_id']
        cursor.execute('UPDATE users SET credits_balance = credits_balance - ? WHERE id = ?', (credits_used, user_id))
    
    conn.commit()
    conn.close()
    return msg_id

if __name__ == '__main__':
    init_db()
