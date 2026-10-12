"""SQLite layer: students + attendance tables, auto roll numbers, daily CSV export."""
import csv
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import config


@contextmanager
def db():
    conn = sqlite3.connect(config.DB_PATH)
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
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                roll_no     TEXT UNIQUE NOT NULL,
                name        TEXT NOT NULL,
                father_name TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL
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
        # older databases have no father_name column
        cols = {r["name"] for r in c.execute("PRAGMA table_info(students)")}
        if "father_name" not in cols:
            c.execute("ALTER TABLE students ADD COLUMN father_name TEXT NOT NULL DEFAULT ''")


# ---------------- roll numbers ----------------
def _roll_prefix(now=None):
    now = now or datetime.now()
    return f"{config.ROLL_PREFIX}{now:%y}-"


def next_roll():
    """Next free roll number in the series, e.g. FA26-0001 -> FA26-0002."""
    prefix = _roll_prefix()
    with db() as c:
        rows = c.execute("SELECT roll_no FROM students WHERE roll_no LIKE ?", (prefix + "%",)).fetchall()
    highest = 0
    for r in rows:
        tail = r["roll_no"][len(prefix):]
        if tail.isdigit():
            highest = max(highest, int(tail))
    return f"{prefix}{highest + 1:04d}"


# ---------------- students ----------------
def create_student(name, father_name):
    """Create a student with the next roll number. Returns (student_id, roll_no)."""
    for _ in range(5):                                   # retry if two people register at once
        roll = next_roll()
        with db() as c:
            try:
                cur = c.execute(
                    "INSERT INTO students (roll_no, name, father_name, created_at) VALUES (?, ?, ?, ?)",
                    (roll, name.strip(), father_name.strip(), datetime.now().isoformat(timespec="seconds")),
                )
                return cur.lastrowid, roll
            except sqlite3.IntegrityError:
                continue
    raise RuntimeError("Could not create a unique roll number. Please try again.")


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
    """Mark present once per day. Returns (newly_marked, time_string).

    The day's CSV file in exports/ is refreshed automatically after every new mark.
    """
    when = when or datetime.now()
    day = when.strftime("%Y-%m-%d")
    with db() as c:
        try:
            c.execute(
                "INSERT INTO attendance (student_id, date, time) VALUES (?, ?, ?)",
                (student_id, day, when.strftime("%H:%M:%S")),
            )
            fresh = True
        except sqlite3.IntegrityError:
            fresh = False
        row = c.execute(
            "SELECT time FROM attendance WHERE student_id = ? AND date = ?", (student_id, day)
        ).fetchone()
    if fresh:
        save_daily_csv(day)
    return fresh, row["time"]


def attendance_between(start, end):
    """Rows (dicts) for start <= date <= end (ISO date strings)."""
    with db() as c:
        rows = c.execute(
            """SELECT a.student_id, s.roll_no, s.name, s.father_name, a.date, a.time
               FROM attendance a JOIN students s ON s.id = a.student_id
               WHERE a.date BETWEEN ? AND ?
               ORDER BY a.date DESC, a.time DESC""",
            (start, end),
        ).fetchall()
        return [dict(r) for r in rows]


def present_on(date_str):
    return attendance_between(date_str, date_str)


def day_sheet(date_str):
    """Every registered student with Present / Absent for one day (used for the CSV)."""
    present = {r["student_id"]: r["time"] for r in present_on(date_str)}
    sheet = []
    for s in list_students():
        t = present.get(s["id"])
        sheet.append({
            "roll_no": s["roll_no"],
            "name": s["name"],
            "father_name": s["father_name"],
            "status": "Present" if t else "Absent",
            "time": t or "",
        })
    return sheet


def day_csv_text(date_str):
    sheet = day_sheet(date_str)
    from io import StringIO
    buf = StringIO()
    w = csv.DictWriter(buf, fieldnames=["roll_no", "name", "father_name", "status", "time"])
    w.writeheader()
    w.writerows(sheet)
    return buf.getvalue()


def save_daily_csv(date_str):
    """Write exports/attendance_<date>.csv (called automatically on every new mark)."""
    config.EXPORT_DIR.mkdir(exist_ok=True)
    path = config.EXPORT_DIR / f"attendance_{date_str}.csv"
    path.write_text(day_csv_text(date_str), encoding="utf-8")
    return path
