import sqlite3

def setup_full_db():
    conn = sqlite3.connect('smart_ledger.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            description TEXT,
            amount INTEGER,
            category TEXT,
            source_type TEXT,
            group_id INTEGER
        )
    ''')
    
    conn.commit()
    conn.close()

setup_full_db()