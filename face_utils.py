
"""Face detection, image processing and LBPH face recognition."""

import shutil
from pathlib import Path

import cv2
import numpy as np
import config


# ===================== HAAR CASCADE =====================

_CASCADE_PATH = (
    Path(__file__).resolve().parent
    / "cascade"
    / "haarcascade_frontalface_default.xml"
)

_CASCADE = cv2.CascadeClassifier(str(_CASCADE_PATH))


def _check_cascade():
    """Ensure the Haar Cascade XML file is loaded."""
    global _CASCADE

    if _CASCADE.empty():
        if not _CASCADE_PATH.is_file():
            raise RuntimeError(
                f"Haar Cascade XML file not found: {_CASCADE_PATH}"
            )

        _CASCADE = cv2.CascadeClassifier(str(_CASCADE_PATH))

    if _CASCADE.empty():
        raise RuntimeError(
            "OpenCV could not load the Haar Cascade XML file. "
            "Check your OpenCV installation and requirements.txt."
        )


# ===================== IMAGE PROCESSING =====================

def to_gray(frame_bgr):
    """Convert a valid BGR image to grayscale."""
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Input image is empty.")

    if len(frame_bgr.shape) != 3:
        raise ValueError("Expected a color BGR image.")

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def detect_faces(gray):
    """Return face bounding boxes as (x, y, w, h)."""
    _check_cascade()

    if gray is None or gray.size == 0:
        return []

    if len(gray.shape) != 2:
        raise ValueError("detect_faces() requires a grayscale image.")

    if gray.dtype != np.uint8:
        gray = cv2.convertScaleAbs(gray)

    gray = np.ascontiguousarray(gray)
    equalized = cv2.equalizeHist(gray)

    min_size = getattr(config, "MIN_FACE_SIZE", (30, 30))

    boxes = _CASCADE.detectMultiScale(
        equalized,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=tuple(min_size),
    )

    return [
        tuple(int(value) for value in box)
        for box in boxes
    ]


def largest_face(boxes):
    """Return the largest detected face, or None."""
    if not boxes:
        return None

    return max(
        boxes,
        key=lambda box: box[2] * box[3],
    )


def preprocess(gray, box):
    """Crop, resize and equalize a detected face."""
    if gray is None or gray.size == 0:
        raise ValueError("Cannot preprocess an empty image.")

    x, y, w, h = (int(value) for value in box)

    if w <= 0 or h <= 0:
        raise ValueError("Invalid face bounding box.")

    face = gray[y:y + h, x:x + w]

    if face.size == 0:
        raise ValueError("The detected face crop is empty.")

    face_size = tuple(config.FACE_SIZE)

    resized = cv2.resize(face, face_size)
    return cv2.equalizeHist(resized)


def decode_image(data: bytes):
    """Decode uploaded image bytes into a BGR image."""
    if not data:
        return None

    array = np.frombuffer(data, dtype=np.uint8)

    if array.size == 0:
        return None

    return cv2.imdecode(array, cv2.IMREAD_COLOR)


# ===================== DATASET MANAGEMENT =====================

def student_dir(student_id):
    """Create and return a student's face dataset directory."""
    config.DATASET_DIR.mkdir(parents=True, exist_ok=True)

    directory = config.DATASET_DIR / str(student_id)
    directory.mkdir(parents=True, exist_ok=True)

    return directory


def sample_count(student_id):
    """Count saved JPG face samples for a student."""
    directory = config.DATASET_DIR / str(student_id)

    if not directory.exists():
        return 0

    return len(list(directory.glob("*.jpg")))


def save_samples(student_id, faces, flip=True):
    """
    Save preprocessed grayscale face images.
    Optionally save mirrored copies as data augmentation.
    """
    directory = student_dir(student_id)
    index = sample_count(student_id)
    saved = 0

    for face in faces:
        if face is None or face.size == 0:
            continue

        if len(face.shape) != 2:
            raise ValueError(
                "save_samples() expects preprocessed grayscale faces."
            )

        variants = [face]

        if flip:
            variants.append(cv2.flip(face, 1))

        for variant in variants:
            file_path = directory / f"img_{index}.jpg"

            success = cv2.imwrite(str(file_path), variant)

            if not success:
                raise RuntimeError(
                    f"Could not save face sample: {file_path}"
                )

            index += 1
            saved += 1

    return saved


def remove_student_data(student_id):
    """Remove a student's stored face images."""
    directory = config.DATASET_DIR / str(student_id)

    shutil.rmtree(directory, ignore_errors=True)


# ===================== LBPH RECOGNIZER =====================

def _new_recognizer():
    """Create an OpenCV LBPH face recognizer."""
    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "OpenCV face module is unavailable. "
            "Install opencv-contrib-python-headless and remove "
            "other conflicting OpenCV packages."
        )

    if not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        raise RuntimeError(
            "LBPHFaceRecognizer_create is unavailable in this "
            "OpenCV installation."
        )

    return cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8,
    )


def train():
    """
    Train LBPH using dataset/<student_id>/*.jpg.
    The folder name is the student's numeric ID.
    Returns the number of images used.
    """
    config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    faces = []
    labels = []

    for folder in sorted(config.DATASET_DIR.iterdir()):
        if not folder.is_dir() or not folder.name.isdigit():
            continue

        student_id = int(folder.name)

        for image_path in sorted(folder.glob("*.jpg")):
            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_GRAYSCALE,
            )

            if image is None or image.size == 0:
                continue

            image = cv2.resize(
                image,
                tuple(config.FACE_SIZE),
            )

            faces.append(image)
            labels.append(student_id)

    if not faces:
        if config.MODEL_PATH.exists():
            config.MODEL_PATH.unlink()

        raise ValueError(
            "No valid face images found. Register a student first."
        )

    recognizer = _new_recognizer()
    recognizer.train(
        faces,
        np.asarray(labels, dtype=np.int32),
    )

    recognizer.write(str(config.MODEL_PATH))

    return len(faces)


def load_recognizer():
    """Load the saved LBPH model, or return None if absent."""
    if not config.MODEL_PATH.exists():
        return None

    if config.MODEL_PATH.stat().st_size == 0:
        raise RuntimeError(
            "The saved model file is empty. Register students and retrain."
        )

    recognizer = _new_recognizer()

    try:
        recognizer.read(str(config.MODEL_PATH))
    except cv2.error as exc:
        raise RuntimeError(
            "The saved recognition model could not be loaded. "
            "Register students and train the model again."
        ) from exc

    return recognizer


# ===================== FACE RECOGNITION =====================

def recognize(frame_bgr, recognizer):
    """
    Detect faces and identify them with LBPH.

    Returns a list of dictionaries containing:
    box, student_id (None for unknown), confidence.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return []

    if recognizer is None:
        raise ValueError("No trained recognizer was provided.")

    gray = to_gray(frame_bgr)
    results = []

    threshold = float(config.CONFIDENCE_THRESHOLD)

    for box in detect_faces(gray):
        face = preprocess(gray, box)
        label, confidence = recognizer.predict(face)

        is_known = float(confidence) <= threshold

        results.append({
            "box": box,
            "student_id": int(label) if is_known else None,
            "confidence": float(confidence),
        })

    return results


def annotate(frame, results, names):
    """Draw recognition boxes and labels on a BGR image."""
    if frame is None or frame.size == 0:
        return frame

    for result in results:
        x, y, w, h = result["box"]

        student_id = result["student_id"]
        known = student_id is not None and student_id in names

        color = (0, 180, 0) if known else (0, 0, 255)

        if known:
            label = str(names[student_id])
        else:
            label = "Unknown"

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            2,
        )

        text_y = max(y - 10, 20)

        cv2.putText(
            frame,
            f"{label} ({result['confidence']:.0f})",
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return frame
