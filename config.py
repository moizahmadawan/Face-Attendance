"""Central settings for the Face Recognition Attendance System."""
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"          # dataset/<student_id>/img_0.jpg ...
MODEL_DIR = BASE_DIR / "trainer"
MODEL_PATH = MODEL_DIR / "model.yml"
DB_PATH = BASE_DIR / "attendance.db"

FACE_SIZE = (200, 200)          # every face is resized to this before training
SAMPLES_PER_STUDENT = 40        # photos captured during webcam registration
CONFIDENCE_THRESHOLD = 70       # LBPH: LOWER = better match. Above this = "Unknown"
CONSECUTIVE_FRAMES = 5          # frames a face must be recognised before marking
MIN_FACE_SIZE = (80, 80)        # ignore tiny / far-away faces

DATASET_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
