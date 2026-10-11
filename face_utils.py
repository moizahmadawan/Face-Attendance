
import cv2
from pathlib import Path

_CASCADE_PATH = (
    Path(cv2.data.haarcascades)
    / "haarcascade_frontalface_default.xml"
)

_CASCADE = cv2.CascadeClassifier(str(_CASCADE_PATH))


def _check_cascade():
    global _CASCADE

    # Cascade dobara load karo agar pehle load nahi hua
    if _CASCADE.empty():
        if not _CASCADE_PATH.exists():
            raise RuntimeError(
                f"Haar Cascade XML file not found: {_CASCADE_PATH}"
            )

        _CASCADE = cv2.CascadeClassifier(str(_CASCADE_PATH))

    if _CASCADE.empty():
        raise RuntimeError(
            "OpenCV Haar Cascade XML load nahi ho saka. "
            "requirements.txt mein opencv-contrib-python-headless use karo."
        )


def detect_faces(gray):
    """Return detected faces as (x, y, w, h)."""
    _check_cascade()

    if gray is None or gray.size == 0:
        return []

    if len(gray.shape) != 2:
        raise ValueError("detect_faces() ko grayscale image chahiye.")

    gray = cv2.equalizeHist(gray)

    boxes = _CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
    )

    return [tuple(int(v) for v in box) for box in boxes]
