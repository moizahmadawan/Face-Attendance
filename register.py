"""
Register a student using the webcam.
Usage:  python register.py                       (asks for roll no + name)
        python register.py --roll 22-185 --name "Moiz"
Look at the camera; slightly turn your head / change expression while it captures.
Press 'q' to stop early.
"""
import argparse

import cv2

import config
import database as db
import face_utils as fu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll")
    ap.add_argument("--name")
    ap.add_argument("--samples", type=int, default=config.SAMPLES_PER_STUDENT)
    args = ap.parse_args()

    roll = args.roll or input("Roll number: ").strip()
    name = args.name or input("Student name: ").strip()
    if not roll or not name:
        print("Roll number and name are required.")
        return

    db.init_db()
    existing = db.get_student_by_roll(roll)
    sid = existing["id"] if existing else db.add_student(roll, name)
    if existing:
        print(f"Roll {roll} already registered ({existing['name']}). Adding more samples.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the webcam.")
        return

    saved, frame_no = 0, 0
    print("Capturing... look at the camera. Press 'q' to stop.")
    while saved < args.samples:
        ok, frame = cap.read()
        if not ok:
            break
        gray = fu.to_gray(frame)
        box = fu.largest_face(fu.detect_faces(gray))
        frame_no += 1
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            if frame_no % 4 == 0:                       # don't save near-identical frames
                saved += fu.save_samples(sid, [fu.preprocess(gray, box)], flip=False)
        cv2.putText(frame, f"Captured: {saved}/{args.samples}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Register face - press q to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

    if saved == 0:
        print("No face captured. Make sure your face is well lit and visible.")
        return
    total = fu.train()
    print(f"Done! {saved} samples saved for {name}. Model trained on {total} images.")


if __name__ == "__main__":
    main()
