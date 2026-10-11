"""Face detection (Haar Cascade) + recognition (LBPH) helpers."""

import shutil

import cv2
import numpy as np

import config


# ---------------- face detector ----------------

_CASCADE_PATH = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

_CASCADE = cv2.CascadeClassifier(_CASCADE_PATH)


def _check_cascade():
    """Raise an error if the Haar Cascade classifier failed to load."""
    if _CASCADE.empty():
        raise RuntimeError(
            "Haar Cascade classifier could not be loaded. "
            "Check your OpenCV installation."
        )


# ---------------- detection / preprocessing ----------------

def to_gray(frame_bgr):
    """Convert a valid BGR image to grayscale."""
    if frame_bgr is None:
        raise ValueError("No image received.")

    if not isinstance(frame_bgr, np.ndarray):
        raise ValueError("Image must be a NumPy array.")

    if frame_bgr.size == 0:
        raise ValueError("Image is empty.")

    if frame_bgr.ndim == 2:
        gray = frame_bgr

    elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2GRAY)

    else:
        raise ValueError("Unsupported image format.")

    if gray.dtype != np.uint8:
        gray = cv2.convertScaleAbs(gray)

    return np.ascontiguousarray(gray)


def detect_faces(gray):
    """Detect faces and return (x, y, w, h) bounding boxes."""
    if gray is None or not isinstance(gray, np.ndarray):
        return []

    if gray.size == 0:
        return []

    # Accept grayscale images only.
    if gray.ndim != 2:
        gray = to_gray(gray)

    if gray.dtype != np.uint8:
        gray = cv2.convertScaleAbs(gray)

    # Very small images cannot be reliably processed.
    if gray.shape[0] < 20 or gray.shape[1] < 20:
        return []

    _check_cascade()

    gray = np.ascontiguousarray(gray)
    gray = cv2.equalizeHist(gray)

    min_size = tuple(int(v) for v in config.MIN_FACE_SIZE)

    boxes = _CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=min_size,
    )

    return [
        tuple(int(v) for v in box)
        for box in boxes
    ]


def largest_face(boxes):
    """Return the largest face box, or None."""
    if not boxes:
        return None

    return max(
        boxes,
        key=lambda box: box[2] * box[3],
    )


def preprocess(gray, box):
    """Crop, resize and equalise a detected face."""
    if gray is None or not isinstance(gray, np.ndarray):
        raise ValueError("Invalid grayscale image.")

    if gray.size == 0:
        raise ValueError("Grayscale image is empty.")

    if gray.ndim != 2:
        gray = to_gray(gray)

    if gray.dtype != np.uint8:
        gray = cv2.convertScaleAbs(gray)

    x, y, w, h = (int(v) for v in box)

    if w <= 0 or h <= 0:
        raise ValueError("Invalid face bounding box.")

    height, width = gray.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Face bounding box is outside the image.")

    face = gray[y1:y2, x1:x2]

    if face.size == 0:
        raise ValueError("Could not crop the detected face.")

    face = cv2.resize(
        face,
        tuple(config.FACE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    return cv2.equalizeHist(face)


def decode_image(data: bytes):
    """Decode uploaded image bytes into a BGR image."""
    if not data:
        return None

    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        if arr.size == 0:
            return None

        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if image is None or image.size == 0:
            return None

        return image

    except (cv2.error, ValueError, TypeError):
        return None


# ---------------- dataset management ----------------

def student_dir(student_id):
    """Create and return a student's dataset directory."""
    directory = config.DATASET_DIR / str(student_id)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def sample_count(student_id):
    """Count saved JPG samples for a student."""
    directory = config.DATASET_DIR / str(student_id)

    if not directory.exists():
        return 0

    return sum(
        1 for path in directory.glob("*.jpg")
        if path.is_file()
    )


def save_samples(student_id, faces, flip=True):
    """Save preprocessed face images and optional mirrored copies."""
    directory = student_dir(student_id)
    index = sample_count(student_id)
    saved = 0

    for face in faces:
        if face is None or not isinstance(face, np.ndarray):
            continue

        if face.size == 0:
            continue

        if face.ndim != 2:
            face = to_gray(face)

        if face.dtype != np.uint8:
            face = cv2.convertScaleAbs(face)

        face = cv2.resize(
            face,
            tuple(config.FACE_SIZE),
        )

        variants = [face]

        if flip:
            variants.append(cv2.flip(face, 1))

        for variant in variants:
            path = directory / f"img_{index}.jpg"

            success = cv2.imwrite(str(path), variant)

            if not success:
                raise IOError(f"Failed to save face image: {path}")

            index += 1
            saved += 1

    return saved


def remove_student_data(student_id):
    """Remove a student's saved face images."""
    directory = config.DATASET_DIR / str(student_id)

    shutil.rmtree(directory, ignore_errors=True)


# ---------------- training / recognition ----------------

def _new_recognizer():
    """Create an OpenCV LBPH face recognizer."""
    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "OpenCV face module is unavailable. "
            "Install opencv-contrib-python-headless."
        )

    if not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        raise RuntimeError(
            "LBPH recognizer is unavailable in this OpenCV installation."
        )

    return cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8,
    )


def train():
    """Train LBPH on dataset images and save the trained model."""
    config.DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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

            image = np.ascontiguousarray(image)

            faces.append(image)
            labels.append(student_id)

    if not faces:
        if config.MODEL_PATH.exists():
            config.MODEL_PATH.unlink()

        raise ValueError(
            "No valid face images found. "
            "Register at least one student first."
        )

    recognizer = _new_recognizer()

    recognizer.train(
        faces,
        np.asarray(labels, dtype=np.int32),
    )

    config.MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    recognizer.write(str(config.MODEL_PATH))

    return len(faces)


def load_recognizer():
    """Load the trained LBPH model, or return None if absent."""
    if not config.MODEL_PATH.exists():
        return None

    recognizer = _new_recognizer()
    recognizer.read(str(config.MODEL_PATH))

    return recognizer


def recognize(frame_bgr, recognizer):
    """Detect faces and identify them using the trained LBPH model."""
    if recognizer is None:
        raise ValueError(
            "No trained face model is available. Train the model first."
        )

    gray = to_gray(frame_bgr)
    results = []

    for box in detect_faces(gray):
        face = preprocess(gray, box)

        label, confidence = recognizer.predict(face)

        known = confidence <= config.CONFIDENCE_THRESHOLD

        results.append({
            "box": box,
            "student_id": int(label) if known else None,
            "confidence": float(confidence),
        })

    return results


# ---------------- drawing / annotation ----------------

def annotate(frame, results, names):
    """Draw face boxes and student labels on an image."""
    if frame is None or not isinstance(frame, np.ndarray):
        raise ValueError("Invalid frame for annotation.")

    output = frame.copy()

    for result in results:
        x, y, w, h = result["box"]

        known = result["student_id"] is not None

        color = (0, 200, 0) if known else (0, 0, 255)

        if known:
            student_name = names.get(
                result["student_id"],
                "Unknown",
            )
        else:
            student_name = "Unknown"

        label = (
            f"{student_name} "
            f"({result['confidence']:.0f})"
        )

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            color,
            2,
        )

        # Keep the label background inside the image.
        label_y = max(y - 28, 0)

        cv2.rectangle(
            output,
            (x, label_y),
            (x + w, min(y, output.shape[0] - 1)),
            color,
            -1,
        )

        cv2.putText(
            output,
            label,
            (x + 4, max(y - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output
