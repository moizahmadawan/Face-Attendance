"""Business logic behind the live camera: register a student, take attendance."""
from datetime import datetime

import config
import database as db
import face_utils as fu

_model_cache = {"key": None, "rec": None}


def get_recognizer():
    """Load the LBPH model once and reload it only when the file changes."""
    path = config.MODEL_PATH
    if not path.exists():
        _model_cache.update(key=None, rec=None)
        return None
    st = path.stat()
    key = (st.st_mtime_ns, st.st_size)
    if _model_cache["key"] != key:
        _model_cache.update(key=key, rec=fu.load_recognizer())
    return _model_cache["rec"]


def _fail(nonce, title, message):
    return {"nonce": nonce, "ok": False, "title": title, "message": message}


def register_live(nonce, name, father_name, frames):
    """Register a student from the frames captured by the live circle scan."""
    name, father_name = (name or "").strip(), (father_name or "").strip()
    if len(name) < 2 or len(father_name) < 2:
        return _fail(nonce, "Details missing", "Enter the student name and father's name, then try again.")

    faces = fu.faces_from_frames(frames)
    if len(faces) < config.MIN_LIVE_SAMPLES:
        return _fail(nonce, "Face not clear",
                     "The scan was not clear enough. Move to a brighter place and try again.")

    try:
        dup_id = fu.find_duplicate(get_recognizer(), faces)
    except Exception:
        dup_id = None
    if dup_id:
        other = db.get_student(dup_id)
        if other:
            return _fail(nonce, "Already registered",
                         f"This face is already registered as {other['name']} ({other['roll_no']}).")

    student_id, roll = db.create_student(name, father_name)
    try:
        fu.save_samples(student_id, faces)
        fu.add_to_model(student_id, faces)
        fu.save_avatar(student_id, frames[min(3, len(frames) - 1)])
    except Exception as exc:                      # undo a half-finished registration
        db.delete_student(student_id)
        fu.remove_student_data(student_id)
        return _fail(nonce, "Registration failed", f"Something went wrong: {exc}")

    return {
        "nonce": nonce, "ok": True, "name": name, "father": father_name, "roll": roll,
        "joined": datetime.now().strftime("%d %b %Y"), "samples": len(faces),
    }


def scan_attendance(nonce, frames):
    """Recognise the person in front of the camera and mark them present."""
    try:
        rec = get_recognizer()
    except Exception as exc:
        return {"nonce": nonce, "status": "error", "message": str(exc)}
    if rec is None:
        return {"nonce": nonce, "status": "no_model"}

    faces = fu.faces_from_frames(frames)
    status, student_id, conf = fu.identify(rec, faces)
    if status != "match":
        return {"nonce": nonce, "status": status}

    student = db.get_student(student_id)
    if not student:                               # model still knows a deleted student
        return {"nonce": nonce, "status": "unknown"}

    fresh, when = db.mark_attendance(student_id)
    return {
        "nonce": nonce, "status": "marked" if fresh else "already",
        "name": student["name"], "roll": student["roll_no"], "time": when,
    }
