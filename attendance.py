"""
Live attendance using the webcam.
Usage:  python attendance.py        (press 'q' to quit)
A student is marked present after being recognised for several consecutive frames,
and only once per day.
"""
from collections import defaultdict

import cv2

import config
import database as db
import face_utils as fu


def main():
    db.init_db()
    rec = fu.load_recognizer()
    if rec is None:
        print("No trained model. Run register.py first.")
        return
    names = {s["id"]: s["name"] for s in db.list_students()}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the webcam.")
        return

    streak = defaultdict(int)       # student_id -> consecutive recognised frames
    marked = set()                  # already handled in this session
    banner, banner_frames = "", 0

    print("Attendance running. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = fu.recognize(frame, rec)
        seen = set()
        for r in results:
            sid = r["student_id"]
            if sid is None or sid not in names:
                continue
            seen.add(sid)
            streak[sid] += 1
            if streak[sid] >= config.CONSECUTIVE_FRAMES and sid not in marked:
                marked.add(sid)
                if db.mark_attendance(sid):
                    banner, banner_frames = f"{names[sid]} marked PRESENT", 60
                else:
                    banner, banner_frames = f"{names[sid]} already marked today", 60
        for sid in list(streak):                      # reset streak if face disappeared
            if sid not in seen:
                streak[sid] = 0

        fu.annotate(frame, results, names)
        if banner_frames > 0:
            cv2.putText(frame, banner, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            banner_frames -= 1
        cv2.putText(frame, f"Present this session: {len(marked)}", (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Face Attendance - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
