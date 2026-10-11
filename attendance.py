
"""
Live face attendance using the webcam.

Usage:
    python attendance.py

Press 'q' to quit.
A student is marked present after being recognised for several
consecutive frames, and only once per day.
"""

from collections import defaultdict

import cv2
import numpy as np

import config
import database as db
import face_utils as fu


# ===================== UI COLORS (BGR) =====================

BG_COLOR = (24, 17, 11)
PANEL_COLOR = (43, 34, 26)
WHITE = (245, 245, 245)
MUTED = (175, 185, 200)
BLUE = (245, 145, 75)
PURPLE = (205, 95, 125)
GREEN = (90, 205, 125)
RED = (85, 85, 235)
YELLOW = (70, 220, 245)


# ===================== DRAWING HELPERS =====================

def draw_panel(frame, x1, y1, x2, y2, color, alpha=0.88):
    """Draw a translucent rectangular panel."""
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        color,
        thickness=-1,
    )

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0,
        frame,
    )


def draw_text(
    frame,
    text,
    position,
    scale=0.6,
    color=WHITE,
    thickness=1,
):
    """Draw clean anti-aliased text."""
    cv2.putText(
        frame,
        str(text),
        position,
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_interface(frame, present_count, total_students, banner, banner_frames):
    """Draw the attendance dashboard over the webcam frame."""
    height, width = frame.shape[:2]

    # Header
    draw_panel(
        frame,
        0,
        0,
        width,
        76,
        BG_COLOR,
        0.91,
    )

    # Accent strip
    cv2.rectangle(
        frame,
        (0, 0),
        (width, 4),
        BLUE,
        -1,
    )

    draw_text(
        frame,
        "FACE ATTENDANCE",
        (22, 32),
        0.85,
        WHITE,
        2,
    )

    draw_text(
        frame,
        "LIVE RECOGNITION SYSTEM",
        (23, 55),
        0.38,
        MUTED,
        1,
    )

    # Live indicator
    cv2.circle(
        frame,
        (width - 160, 32),
        5,
        GREEN,
        -1,
        cv2.LINE_AA,
    )

    draw_text(
        frame,
        "LIVE",
        (width - 145, 37),
        0.5,
        GREEN,
        1,
    )

    # Statistics panel
    card_y1 = 94
    card_y2 = 158
    card_width = min(220, max((width - 60) // 3, 120))

    # Present card
    draw_panel(
        frame,
        20,
        card_y1,
        20 + card_width,
        card_y2,
        PANEL_COLOR,
        0.90,
    )

    draw_text(
        frame,
        "SESSION PRESENT",
        (32, 118),
        0.38,
        MUTED,
    )

    draw_text(
        frame,
        present_count,
        (32, 146),
        0.9,
        GREEN,
        2,
    )

    # Registered students card
    card2_x1 = 32 + card_width
    card2_x2 = card2_x1 + card_width

    if card2_x2 < width - 15:
        draw_panel(
            frame,
            card2_x1,
            card_y1,
            card2_x2,
            card_y2,
            PANEL_COLOR,
            0.90,
        )

        draw_text(
            frame,
            "REGISTERED",
            (card2_x1 + 12, 118),
            0.38,
            MUTED,
        )

        draw_text(
            frame,
            total_students,
            (card2_x1 + 12, 146),
            0.9,
            BLUE,
            2,
        )

    # Status notification
    if banner_frames > 0 and banner:
        banner_color = (
            GREEN
            if "PRESENT" in banner
            else YELLOW
        )

        panel_top = height - 100

        draw_panel(
            frame,
            20,
            panel_top,
            width - 20,
            height - 55,
            BG_COLOR,
            0.92,
        )

        cv2.rectangle(
            frame,
            (20, panel_top),
            (25, height - 55),
            banner_color,
            -1,
        )

        draw_text(
            frame,
            banner,
            (38, panel_top + 29),
            0.56,
            banner_color,
            1,
        )

    # Footer
    draw_panel(
        frame,
        0,
        height - 35,
        width,
        height,
        BG_COLOR,
        0.88,
    )

    draw_text(
        frame,
        "Press Q to quit",
        (18, height - 12),
        0.42,
        MUTED,
    )

    draw_text(
        frame,
        "FACE RECOGNITION",
        (width - 190, height - 12),
        0.38,
        MUTED,
    )


# ===================== MAIN APPLICATION =====================

def main():
    db.init_db()

    rec = fu.load_recognizer()

    if rec is None:
        print("No trained model. Run register.py first.")
        return

    names = {
        student["id"]: student["name"]
        for student in db.list_students()
    }

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open the webcam.")
        return

    # Request a suitable camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    streak = defaultdict(int)
    marked = set()

    banner = ""
    banner_frames = 0

    window_name = "Face Attendance"
    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    print("Live attendance started. Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                print("Could not read a frame from the webcam.")
                break

            # Mirror the camera for a natural preview
            frame = cv2.flip(frame, 1)

            results = fu.recognize(
                frame,
                rec,
            )

            seen = set()

            for result in results:
                sid = result["student_id"]

                if sid is None or sid not in names:
                    continue

                seen.add(sid)
                streak[sid] += 1

                required_frames = getattr(
                    config,
                    "CONSECUTIVE_FRAMES",
                    5,
                )

                if (
                    streak[sid] >= required_frames
                    and sid not in marked
                ):
                    marked.add(sid)

                    if db.mark_attendance(sid):
                        banner = (
                            f"{names[sid]} - MARKED PRESENT"
                        )
                    else:
                        banner = (
                            f"{names[sid]} - ALREADY MARKED TODAY"
                        )

                    banner_frames = 60

            # Reset streaks when faces disappear
            for sid in list(streak):
                if sid not in seen:
                    streak[sid] = 0

            # Draw recognition boxes and labels
            annotated = fu.annotate(
                frame.copy(),
                results,
                names,
            )

            # Draw the updated interface
            draw_interface(
                annotated,
                len(marked),
                len(names),
                banner,
                banner_frames,
            )

            if banner_frames > 0:
                banner_frames -= 1

            cv2.imshow(
                window_name,
                annotated,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
