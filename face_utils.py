"""Face detection (Haar Cascade) + recognition (LBPH) helpers."""
import shutil

import cv2
import numpy as np

import config

_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# ---------------- detection / preprocessing ----------------
def to_gray(frame_bgr):
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def detect_faces(gray):
    """Return list of (x, y, w, h) boxes."""
    boxes = _CASCADE.detectMultiScale(
        cv2.equalizeHist(gray), scaleFactor=1.2, minNeighbors=5, minSize=config.MIN_FACE_SIZE
    )
    return [tuple(int(v) for v in b) for b in boxes]


def largest_face(boxes):
    return max(boxes, key=lambda b: b[2] * b[3]) if boxes else None


def preprocess(gray, box):
    """Crop face, resize and equalise so lighting differences matter less."""
    x, y, w, h = box
    face = cv2.resize(gray[y:y + h, x:x + w], config.FACE_SIZE)
    return cv2.equalizeHist(face)


def decode_image(data: bytes):
    """Bytes (upload / camera snapshot) -> BGR image, or None."""
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ---------------- dataset ----------------
def student_dir(student_id):
    d = config.DATASET_DIR / str(student_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def sample_count(student_id):
    d = config.DATASET_DIR / str(student_id)
    return len(list(d.glob("*.jpg"))) if d.exists() else 0


def save_samples(student_id, faces, flip=True):
    """Save preprocessed faces (+ mirrored copies as cheap augmentation). Returns count saved."""
    d = student_dir(student_id)
    idx = sample_count(student_id)
    saved = 0
    for f in faces:
        variants = [f, cv2.flip(f, 1)] if flip else [f]
        for v in variants:
            cv2.imwrite(str(d / f"img_{idx}.jpg"), v)
            idx += 1
            saved += 1
    return saved


def remove_student_data(student_id):
    shutil.rmtree(config.DATASET_DIR / str(student_id), ignore_errors=True)


# ---------------- training / recognition ----------------
def _new_recognizer():
    return cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)


def train():
    """Train LBPH on everything in dataset/. Label = student id. Returns number of images."""
    faces, labels = [], []
    for folder in sorted(config.DATASET_DIR.iterdir()):
        if not (folder.is_dir() and folder.name.isdigit()):
            continue
        for img_path in folder.glob("*.jpg"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(cv2.resize(img, config.FACE_SIZE))
                labels.append(int(folder.name))
    if not faces:
        if config.MODEL_PATH.exists():
            config.MODEL_PATH.unlink()
        raise ValueError("No face images found. Register at least one student first.")
    rec = _new_recognizer()
    rec.train(faces, np.array(labels))
    rec.write(str(config.MODEL_PATH))
    return len(faces)


def load_recognizer():
    if not config.MODEL_PATH.exists():
        return None
    rec = _new_recognizer()
    rec.read(str(config.MODEL_PATH))
    return rec


def recognize(frame_bgr, recognizer):
    """Detect + identify every face. Returns [{'box', 'student_id' (None=unknown), 'confidence'}]."""
    gray = to_gray(frame_bgr)
    results = []
    for box in detect_faces(gray):
        label, conf = recognizer.predict(preprocess(gray, box))
        known = conf <= config.CONFIDENCE_THRESHOLD
        results.append({"box": box, "student_id": label if known else None, "confidence": float(conf)})
    return results


def annotate(frame, results, names):
    """Draw boxes + labels. names: {student_id: 'Name'}."""
    for r in results:
        x, y, w, h = r["box"]
        known = r["student_id"] is not None
        color = (0, 200, 0) if known else (0, 0, 255)
        label = names.get(r["student_id"], "Unknown") if known else "Unknown"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(frame, (x, y - 28), (x + w, y), color, -1)
        cv2.putText(frame, f"{label} ({r['confidence']:.0f})", (x + 4, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame
