import sqlite3
from datetime import datetime
import uuid


class ChatDatabase:
    def __init__(self, db_path="chat_history.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TEXT
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            role TEXT,
            content TEXT,
            created_at TEXT
        )
        """)

        self.conn.commit()

    # -------------------------
    # Conversation Methods
    # -------------------------

    def create_conversation(self, title="New Chat"):
        conv_id = str(uuid.uuid4())
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO conversations VALUES (?, ?, ?)",
            (conv_id, title, datetime.utcnow().isoformat())
        )
        self.conn.commit()
        return conv_id

    def get_conversations(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, title FROM conversations ORDER BY created_at DESC")
        return cursor.fetchall()

    def delete_conversation(self, conv_id):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM messages WHERE conversation_id=?", (conv_id,))
        cursor.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
        self.conn.commit()

    # -------------------------
    # Message Methods
    # -------------------------

    def add_message(self, conv_id, role, content):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conv_id, role, content, datetime.utcnow().isoformat())
        )
        self.conn.commit()

    def get_messages(self, conv_id):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id",
            (conv_id,)
        )
        return cursor.fetchall()
