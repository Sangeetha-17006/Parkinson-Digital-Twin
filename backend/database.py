
from pathlib import Path
import sqlite3


# ==========================================
# 1. PROJECT AND DATABASE PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_PATH = BASE_DIR / "digital_twin.db"


# ==========================================
# 2. DATABASE CONNECTION
# ==========================================

def get_connection():
    """Create and return a database connection."""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


# ==========================================
# 3. INITIALIZE DATABASE
# ==========================================

def initialize_database():
    """Create the tables required by the Digital Twin."""

    connection = get_connection()

    try:
        cursor = connection.cursor()

        # Preserve the existing prediction history table.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_data TEXT,
                prediction TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Store predictions evaluated against known actual labels.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER,
                actual_label INTEGER NOT NULL
                    CHECK (actual_label IN (0, 1)),
                predicted_label INTEGER NOT NULL
                    CHECK (predicted_label IN (0, 1)),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id)
                    REFERENCES predictions(id)
            )
        """)

        connection.commit()

        print("Database tables initialized successfully.")

    finally:
        connection.close()
