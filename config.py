"""Central settings for the Face Recognition Attendance System."""
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"          # dataset/<student_id>/img_0.jpg ...
AVATAR_DIR = BASE_DIR / "avatars"           # avatars/<student_id>.jpg (profile picture)
EXPORT_DIR = BASE_DIR / "exports"           # exports/attendance_YYYY-MM-DD.csv (auto-saved daily)
MODEL_DIR = BASE_DIR / "trainer"
MODEL_PATH = MODEL_DIR / "model.yml.gz"     # gzip keeps the model small
DB_PATH = BASE_DIR / "attendance.db"

ROLL_PREFIX = "FA"              # roll numbers look like FA26-0001, FA26-0002, ...
FACE_SIZE = (200, 200)          # every face is resized to this before training
MIN_LIVE_SAMPLES = 24           # usable face frames needed to finish a live registration
CONFIDENCE_THRESHOLD = 70       # LBPH: LOWER = better match. Above this = "Unknown"
UNCLEAR_MARGIN = 10             # between THRESHOLD and THRESHOLD+MARGIN = "face not clear"
DUPLICATE_THRESHOLD = 50        # a new registration this close to a saved face is blocked
MIN_VOTES = 3                   # of the frames sent per scan, this many must agree

for _d in (DATASET_DIR, AVATAR_DIR, EXPORT_DIR, MODEL_DIR):
    _d.mkdir(exist_ok=True)
