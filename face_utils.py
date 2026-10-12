"""Face samples, LBPH training and recognition.

Faces are detected live in the browser (MediaPipe Face Mesh). The browser sends
aligned 224x224 face crops; this module only turns them into training samples
and recognises them with OpenCV's LBPH recognizer.
"""
import base64
import shutil
from collections import Counter

import cv2
import numpy as np

import config

# elliptical mask: keeps the face, drops background and hair corners
_MASK = np.zeros(config.FACE_SIZE[::-1], dtype=np.uint8)
cv2.ellipse(
    _MASK,
    (config.FACE_SIZE[0] // 2, config.FACE_SIZE[1] // 2),
    (int(config.FACE_SIZE[0] * 0.43), int(config.FACE_SIZE[1] * 0.49)),
    0, 0, 360, 255, -1,
)


# ===================== IMAGE HELPERS =====================

def decode_data_url(data_url):
    """'data:image/jpeg;base64,....' -> BGR image (or None)."""
    if not data_url or not isinstance(data_url, str):
        return None
    payload = data_url.split(",", 1)[-1]
    try:
        raw = base64.b64decode(payload)
    except Exception:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def preprocess(image):
    """Aligned face crop (BGR or gray) -> normalised grayscale face for LBPH.

    Returns None when the crop is unusable (empty / flat).
    """
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.resize(gray, config.FACE_SIZE, interpolation=cv2.INTER_AREA)
    if float(gray.std()) < 12:           # blank or covered camera
        return None
    gray = cv2.equalizeHist(gray)
    return cv2.bitwise_and(gray, gray, mask=_MASK)


def faces_from_frames(frames):
    """List of data-URLs -> list of preprocessed faces (bad frames skipped)."""
    faces = []
    for data_url in frames or []:
        face = preprocess(decode_data_url(data_url))
        if face is not None:
            faces.append(face)
    return faces


# ===================== DATASET + AVATARS =====================

def student_dir(student_id):
    d = config.DATASET_DIR / str(student_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def sample_count(student_id):
    d = config.DATASET_DIR / str(student_id)
    return len(list(d.glob("*.jpg"))) if d.exists() else 0


def save_samples(student_id, faces):
    directory = student_dir(student_id)
    index = sample_count(student_id)
    saved = 0
    for face in faces:
        if not cv2.imwrite(str(directory / f"img_{index}.jpg"), face):
            raise RuntimeError("Could not save a face sample.")
        index += 1
        saved += 1
    return saved


def save_avatar(student_id, frame_data_url):
    """Store one clean profile picture (shown on the student card)."""
    image = decode_data_url(frame_data_url)
    if image is None:
        return
    image = cv2.resize(image, (160, 160), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(config.AVATAR_DIR / f"{student_id}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 85])


def avatar_data_uri(student_id):
    path = config.AVATAR_DIR / f"{student_id}.jpg"
    if not path.exists():
        return None
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def remove_student_data(student_id):
    shutil.rmtree(config.DATASET_DIR / str(student_id), ignore_errors=True)
    (config.AVATAR_DIR / f"{student_id}.jpg").unlink(missing_ok=True)


# ===================== LBPH MODEL =====================

def _new_recognizer():
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        raise RuntimeError(
            "OpenCV face module is unavailable. Install opencv-contrib-python-headless "
            "and uninstall other OpenCV packages."
        )
    return cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)


def train():
    """Train from dataset/<student_id>/*.jpg. Returns the number of images used."""
    faces, labels = [], []
    for folder in sorted(config.DATASET_DIR.iterdir()):
        if not folder.is_dir() or not folder.name.isdigit():
            continue
        for path in sorted(folder.glob("*.jpg")):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None or img.size == 0:
                continue
            faces.append(cv2.resize(img, config.FACE_SIZE))
            labels.append(int(folder.name))
    if not faces:
        config.MODEL_PATH.unlink(missing_ok=True)
        raise ValueError("No face images found. Register a student first.")
    rec = _new_recognizer()
    rec.train(faces, np.asarray(labels, dtype=np.int32))
    rec.write(str(config.MODEL_PATH))
    return len(faces)


def add_to_model(student_id, faces):
    """Add one student's faces to the saved model without retraining everyone."""
    labels = np.full(len(faces), int(student_id), dtype=np.int32)
    if config.MODEL_PATH.exists() and config.MODEL_PATH.stat().st_size > 0:
        rec = load_recognizer()
        rec.update(faces, labels)
    else:
        rec = _new_recognizer()
        rec.train(faces, labels)
    rec.write(str(config.MODEL_PATH))


def load_recognizer():
    if not config.MODEL_PATH.exists():
        return None
    if config.MODEL_PATH.stat().st_size == 0:
        raise RuntimeError("The saved model file is empty. Register students again.")
    rec = _new_recognizer()
    try:
        rec.read(str(config.MODEL_PATH))
    except cv2.error as exc:
        raise RuntimeError("The saved model could not be loaded. Register students again.") from exc
    return rec


# ===================== RECOGNITION =====================

def _predict_all(recognizer, faces):
    out = []
    for f in faces:
        label, conf = recognizer.predict(f)
        out.append((int(label), float(conf)))
    return out


def identify(recognizer, faces):
    """Vote over several frames of one person.

    Returns (status, student_id, confidence):
      'match'    - confident match
      'unclear'  - looks like someone registered but the picture is poor
      'unknown'  - no registered face matches
    """
    if recognizer is None:
        return "no_model", None, None
    preds = _predict_all(recognizer, faces)
    if len(preds) < config.MIN_VOTES:
        return "unclear", None, None

    soft = config.CONFIDENCE_THRESHOLD + config.UNCLEAR_MARGIN
    close = [p for p in preds if p[1] <= soft]
    if len(close) < config.MIN_VOTES:
        return "unknown", None, float(np.median([p[1] for p in preds]))

    label, votes = Counter(p[0] for p in close).most_common(1)[0]
    if votes < config.MIN_VOTES:
        return "unclear", None, None
    median_conf = float(np.median([p[1] for p in close if p[0] == label]))
    if median_conf <= config.CONFIDENCE_THRESHOLD:
        return "match", label, median_conf
    return "unclear", None, median_conf


def find_duplicate(recognizer, faces):
    """Return the student_id whose saved face already matches these faces, else None."""
    if recognizer is None or not faces:
        return None
    preds = _predict_all(recognizer, faces)
    strong = [p for p in preds if p[1] <= config.DUPLICATE_THRESHOLD]
    if len(strong) < max(4, int(len(preds) * 0.4)):
        return None
    label, votes = Counter(p[0] for p in strong).most_common(1)[0]
    return label if votes >= max(4, int(len(preds) * 0.35)) else None
