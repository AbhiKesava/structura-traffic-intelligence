"""
Database module for Structura Traffic Intelligence Platform.
SQLite-backed persistent storage for:
- User accounts (with PBKDF2-HMAC-SHA256 password hashing) & active sessions
- Per-user saved application state (inputs, preferences, saved corridors, datasets)
- Predictions history (scoped per user, with search, filter, export)
- Uploaded datasets metadata (scoped per user)
- Model training & evaluation metrics
"""

import sqlite3
import json
import hashlib
import secrets
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "structura.db"
DEFAULT_DEMO_EMAIL = "alex.taylor@structura.io"


def get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000)
    return f"{salt}${dk.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, expected_hex = stored_hash.split("$", 1)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000)
        return secrets.compare_digest(dk.hex(), expected_hex)
    except Exception:
        return False


def _ensure_column(cursor, table: str, column: str, definition: str):
    cursor.execute(f"PRAGMA table_info({table})")
    existing = {row["name"] for row in cursor.fetchall()}
    if column not in existing:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def init_db():
    conn = get_db()
    cursor = conn.cursor()

    # 1. Users Table (persistent accounts)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT DEFAULT 'Traffic analyst',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        last_login_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 2. User Sessions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_sessions (
        token TEXT PRIMARY KEY,
        user_email TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 3. Per-User Saved Workspace State (inputs, corridors, preferences, etc.)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_data (
        user_email TEXT PRIMARY KEY,
        state_json TEXT NOT NULL DEFAULT '{}',
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 4. Predictions History Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prediction_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        user_email TEXT DEFAULT 'alex.taylor@structura.io',
        module_type TEXT NOT NULL,
        road_segment TEXT NOT NULL,
        traffic_volume REAL,
        avg_speed REAL,
        occupancy REAL,
        time_of_day TEXT,
        day_of_week TEXT,
        temperature REAL,
        rainfall REAL,
        predicted_congestion TEXT,
        predicted_speed REAL,
        confidence REAL,
        probabilities_json TEXT,
        model_name TEXT,
        notes TEXT
    )
    """)

    # 5. Uploaded Datasets Metadata
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS uploaded_datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT DEFAULT 'alex.taylor@structura.io',
        filename TEXT NOT NULL,
        stored_path TEXT,
        upload_time DATETIME DEFAULT CURRENT_TIMESTAMP,
        row_count INTEGER,
        col_count INTEGER,
        features_json TEXT,
        summary_stats_json TEXT,
        status TEXT
    )
    """)

    # 6. Model Training Records & Benchmark Comparisons
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_benchmarks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        task_type TEXT NOT NULL,
        trained_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        train_accuracy REAL,
        val_accuracy REAL,
        test_accuracy REAL,
        precision REAL,
        recall REAL,
        f1_score REAL,
        mae REAL,
        mse REAL,
        rmse REAL,
        r2_score REAL,
        training_time_sec REAL,
        prediction_time_ms REAL,
        metrics_json TEXT
    )
    """)

    # Safe migrations for pre-existing databases
    _ensure_column(cursor, "prediction_history", "user_email", "TEXT DEFAULT 'alex.taylor@structura.io'")
    _ensure_column(cursor, "uploaded_datasets", "user_email", "TEXT DEFAULT 'alex.taylor@structura.io'")
    _ensure_column(cursor, "uploaded_datasets", "stored_path", "TEXT")

    # Seed default demo account if missing
    cursor.execute("SELECT id FROM users WHERE email = ?", (DEFAULT_DEMO_EMAIL,))
    if cursor.fetchone() is None:
        cursor.execute(
            "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)",
            ("Alex Taylor", DEFAULT_DEMO_EMAIL, hash_password("structura2026"), "Traffic analyst")
        )
        cursor.execute(
            "INSERT OR IGNORE INTO user_data (user_email, state_json) VALUES (?, ?)",
            (DEFAULT_DEMO_EMAIL, json.dumps({}))
        )

    conn.commit()
    conn.close()


# =============================================================================
# USER AUTHENTICATION & SESSION HELPERS
# =============================================================================

def _format_user(row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "email": row["email"],
        "role": row["role"] or "Traffic analyst",
        "created_at": row["created_at"],
        "last_login_at": row["last_login_at"],
    }


def create_user(name: str, email: str, password: str, role: str = "Traffic analyst") -> Dict[str, Any]:
    clean_email = email.strip().lower()
    clean_name = name.strip() or clean_email.split("@")[0]

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE email = ?", (clean_email,))
    if cursor.fetchone() is not None:
        conn.close()
        raise ValueError("An account with this email already exists. Please sign in instead.")

    pw_hash = hash_password(password)
    cursor.execute(
        "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)",
        (clean_name, clean_email, pw_hash, role)
    )
    cursor.execute(
        "INSERT OR IGNORE INTO user_data (user_email, state_json) VALUES (?, ?)",
        (clean_email, json.dumps({}))
    )
    conn.commit()

    cursor.execute("SELECT * FROM users WHERE email = ?", (clean_email,))
    user = _format_user(cursor.fetchone())
    conn.close()
    return user


def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    clean_email = email.strip().lower()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (clean_email,))
    row = cursor.fetchone()
    if row is None or not verify_password(password, row["password_hash"]):
        conn.close()
        return None

    cursor.execute(
        "UPDATE users SET last_login_at = CURRENT_TIMESTAMP WHERE email = ?",
        (clean_email,)
    )
    conn.commit()
    cursor.execute("SELECT * FROM users WHERE email = ?", (clean_email,))
    user = _format_user(cursor.fetchone())
    conn.close()
    return user


def ensure_quick_user(email: str, name: str, role: str = "Traffic analyst") -> Dict[str, Any]:
    clean_email = email.strip().lower()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (clean_email,))
    row = cursor.fetchone()
    if row is None:
        cursor.execute(
            "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)",
            (name.strip() or clean_email.split("@")[0], clean_email, hash_password(secrets.token_hex(16)), role)
        )
        cursor.execute(
            "INSERT OR IGNORE INTO user_data (user_email, state_json) VALUES (?, ?)",
            (clean_email, json.dumps({}))
        )
    else:
        cursor.execute(
            "UPDATE users SET last_login_at = CURRENT_TIMESTAMP WHERE email = ?",
            (clean_email,)
        )
    conn.commit()
    cursor.execute("SELECT * FROM users WHERE email = ?", (clean_email,))
    user = _format_user(cursor.fetchone())
    conn.close()
    return user


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    clean_email = email.strip().lower()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (clean_email,))
    row = cursor.fetchone()
    conn.close()
    return _format_user(row) if row else None


def create_session(user_email: str) -> str:
    clean_email = user_email.strip().lower()
    token = secrets.token_urlsafe(32)
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO user_sessions (token, user_email) VALUES (?, ?)",
        (token, clean_email)
    )
    conn.commit()
    conn.close()
    return token


def get_user_by_token(token: str) -> Optional[Dict[str, Any]]:
    if not token:
        return None
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.* FROM user_sessions s
        JOIN users u ON u.email = s.user_email
        WHERE s.token = ?
    """, (token.strip(),))
    row = cursor.fetchone()
    conn.close()
    return _format_user(row) if row else None


def delete_session(token: str) -> None:
    if not token:
        return
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM user_sessions WHERE token = ?", (token.strip(),))
    conn.commit()
    conn.close()


# =============================================================================
# PER-USER SAVED STATE (SURVIVES LOGOUT & RE-LOGIN)
# =============================================================================

def get_user_state(user_email: str) -> Dict[str, Any]:
    clean_email = (user_email or DEFAULT_DEMO_EMAIL).strip().lower()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT state_json FROM user_data WHERE user_email = ?", (clean_email,))
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return {}
    try:
        return json.loads(row["state_json"]) or {}
    except Exception:
        return {}


def save_user_state(user_email: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    clean_email = (user_email or DEFAULT_DEMO_EMAIL).strip().lower()
    current = get_user_state(clean_email)
    if isinstance(patch, dict):
        for k, v in patch.items():
            if isinstance(v, dict) and isinstance(current.get(k), dict):
                current[k].update(v)
            else:
                current[k] = v

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO user_data (user_email, state_json, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(user_email) DO UPDATE SET
            state_json = excluded.state_json,
            updated_at = CURRENT_TIMESTAMP
    """, (clean_email, json.dumps(current)))
    conn.commit()
    conn.close()
    return current


def get_user_datasets(user_email: str) -> List[Dict[str, Any]]:
    clean_email = (user_email or DEFAULT_DEMO_EMAIL).strip().lower()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, filename, stored_path, upload_time, row_count, col_count, status FROM uploaded_datasets WHERE user_email = ? ORDER BY id DESC LIMIT 10",
        (clean_email,)
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def clear_user_datasets(user_email: str) -> None:
    clean_email = (user_email or DEFAULT_DEMO_EMAIL).strip().lower()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM uploaded_datasets WHERE user_email = ?", (clean_email,))
    conn.commit()
    conn.close()


# =============================================================================
# PREDICTION HISTORY (SCOPED PER USER)
# =============================================================================

def save_prediction(
    module_type: str,
    road_segment: str,
    traffic_volume: float,
    avg_speed: float,
    occupancy: float,
    time_of_day: str,
    day_of_week: str,
    temperature: float,
    rainfall: float,
    predicted_congestion: Optional[str] = None,
    predicted_speed: Optional[float] = None,
    confidence: Optional[float] = None,
    probabilities: Optional[Dict[str, float]] = None,
    model_name: str = "MLP",
    notes: Optional[str] = None,
    user_email: Optional[str] = None
) -> int:
    clean_email = (user_email or DEFAULT_DEMO_EMAIL).strip().lower()
    conn = get_db()
    cursor = conn.cursor()
    prob_str = json.dumps(probabilities) if probabilities else None

    cursor.execute("""
    INSERT INTO prediction_history (
        user_email, module_type, road_segment, traffic_volume, avg_speed, occupancy,
        time_of_day, day_of_week, temperature, rainfall,
        predicted_congestion, predicted_speed, confidence, probabilities_json,
        model_name, notes
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        clean_email, module_type, road_segment, traffic_volume, avg_speed, occupancy,
        time_of_day, day_of_week, temperature, rainfall,
        predicted_congestion, predicted_speed, confidence, prob_str,
        model_name, notes
    ))
    conn.commit()
    pred_id = cursor.lastrowid
    conn.close()
    return pred_id


def get_prediction_history(
    limit: int = 50,
    search: Optional[str] = None,
    congestion_filter: Optional[str] = None,
    module_filter: Optional[str] = None,
    user_email: Optional[str] = None
) -> List[Dict[str, Any]]:
    conn = get_db()
    cursor = conn.cursor()
    query = "SELECT * FROM prediction_history WHERE 1=1"
    params: List[Any] = []

    if user_email:
        query += " AND LOWER(COALESCE(user_email, 'alex.taylor@structura.io')) = ?"
        params.append(user_email.strip().lower())
    if search:
        query += " AND (road_segment LIKE ? OR model_name LIKE ?)"
        params.extend([f"%{search}%", f"%{search}%"])
    if congestion_filter and congestion_filter != "ALL":
        query += " AND predicted_congestion = ?"
        params.append(congestion_filter)
    if module_filter and module_filter != "ALL":
        query += " AND module_type = ?"
        params.append(module_filter)

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    results = [dict(r) for r in rows]
    conn.close()
    return results


def get_history_stats(user_email: Optional[str] = None) -> Dict[str, Any]:
    conn = get_db()
    cursor = conn.cursor()
    where = ""
    params: List[Any] = []
    if user_email:
        where = " WHERE LOWER(COALESCE(user_email, 'alex.taylor@structura.io')) = ?"
        params = [user_email.strip().lower()]

    cursor.execute(f"SELECT COUNT(*) as total FROM prediction_history{where}", params)
    total = cursor.fetchone()["total"]

    and_clause = " AND predicted_congestion IS NOT NULL" if where else " WHERE predicted_congestion IS NOT NULL"
    cursor.execute(f"""
        SELECT predicted_congestion, COUNT(*) as cnt
        FROM prediction_history{where}{and_clause}
        GROUP BY predicted_congestion
    """, params)
    breakdown = {r["predicted_congestion"]: r["cnt"] for r in cursor.fetchall()}

    cursor.execute(f"""
        SELECT AVG(predicted_speed) as avg_pred_spd, AVG(confidence) as avg_conf
        FROM prediction_history{where}
    """, params)
    row = cursor.fetchone()
    avg_pred_spd = row["avg_pred_spd"] or 0.0
    avg_conf = row["avg_conf"] or 0.0

    conn.close()
    return {
        "total_predictions": total,
        "congestion_breakdown": breakdown,
        "average_predicted_speed": round(avg_pred_spd, 1),
        "average_confidence": round(avg_conf, 2)
    }


init_db()
