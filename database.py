"""SQLite layer: students + attendance tables."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from config import DB_PATH


@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with db() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS students (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                roll_no    TEXT UNIQUE NOT NULL,
                name       TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attendance (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
                date       TEXT NOT NULL,
                time       TEXT NOT NULL,
                UNIQUE (student_id, date)
            );
            """
        )


# ---------------- students ----------------
def add_student(roll_no, name):
    with db() as c:
        try:
            cur = c.execute(
                "INSERT INTO students (roll_no, name, created_at) VALUES (?, ?, ?)",
                (roll_no.strip(), name.strip(), datetime.now().isoformat(timespec="seconds")),
            )
        except sqlite3.IntegrityError:
            raise ValueError(f"Roll number '{roll_no}' already exists")
        return cur.lastrowid


def get_student_by_roll(roll_no):
    with db() as c:
        r = c.execute("SELECT * FROM students WHERE roll_no = ?", (roll_no.strip(),)).fetchone()
        return dict(r) if r else None


def get_student(student_id):
    with db() as c:
        r = c.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
        return dict(r) if r else None


def list_students():
    with db() as c:
        return [dict(r) for r in c.execute("SELECT * FROM students ORDER BY roll_no")]


def delete_student(student_id):
    with db() as c:
        c.execute("DELETE FROM students WHERE id = ?", (student_id,))


# ---------------- attendance ----------------
def mark_attendance(student_id, when=None):
    """Mark present once per day. Returns True if newly marked, False if already marked."""
    when = when or datetime.now()
    with db() as c:
        try:
            c.execute(
                "INSERT INTO attendance (student_id, date, time) VALUES (?, ?, ?)",
                (student_id, when.strftime("%Y-%m-%d"), when.strftime("%H:%M:%S")),
            )
            return True
        except sqlite3.IntegrityError:
            return False


def attendance_between(start, end):
    """Rows (dicts) for start <= date <= end (ISO date strings)."""
    with db() as c:
        rows = c.execute(
            """SELECT a.student_id, s.roll_no, s.name, a.date, a.time
               FROM attendance a JOIN students s ON s.id = a.student_id
               WHERE a.date BETWEEN ? AND ?
               ORDER BY a.date DESC, a.time DESC""",
            (start, end),
        ).fetchall()
        return [dict(r) for r in rows]


def present_on(date_str):
    return attendance_between(date_str, date_str)
