# Face Attendance System

Live face registration and attendance. No photo uploads: everything is verified by a live camera
scan inside a full-screen circle, on desktop and on mobile.

## How it works
- **Register**: type the student name and father's name. The roll number is created automatically
  (`FA26-0001`, `FA26-0002`, ...). Tap *Start face verification*, then look straight, turn left,
  right, up and down until the circle completes. The app says "Registration successful".
- **Attendance**: tap *Start attendance camera* and stand in front of it. The face is detected inside the circle,
  attendance is marked and the app says so. If the picture is not clear it tells the person to move to the light.
- **Daily save**: every mark is stored in SQLite and written to `exports/attendance_YYYY-MM-DD.csv`
  (Present / Absent for every student). CSV downloads are on the Overview and Reports tabs.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501. The face engine (MediaPipe Face Mesh) is bundled in
`components/face_cam/mp`, so no internet is needed after install.

## Use it on a phone
Browsers only allow the camera on **https** (or localhost). Pick one:
1. Deploy to Streamlit Community Cloud (https automatically).
2. Tunnel: `ngrok http 8501` or `cloudflared tunnel --url http://localhost:8501`.
3. Same Wi-Fi with a self-signed certificate:
   ```bash
   openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -days 365 -subj "/CN=localhost"
   streamlit run app.py --server.sslCertFile cert.pem --server.sslKeyFile key.pem
   ```
   Then open `https://<your-pc-ip>:8501` on the phone and accept the certificate warning.

## Files
| File | Purpose |
|---|---|
| `app.py` | Dashboard (Overview, Attendance, Register, Students, Reports) |
| `components/face_cam/` | Live circular camera (full screen, voice, pose steps) |
| `services.py` | Register a student / mark attendance from camera frames |
| `face_utils.py` | LBPH training and recognition |
| `database.py` | SQLite, auto roll numbers, daily CSV |
| `config.py` | Roll prefix, thresholds |

## Settings (`config.py`)
- `ROLL_PREFIX` changes the roll series (`FA` -> e.g. `UOG`).
- `CONFIDENCE_THRESHOLD`: lower = stricter matching.
- `DUPLICATE_THRESHOLD`: blocks registering the same face twice.

## Tips
Good front light, face inside the circle, one person at a time. Do not sit with a window behind you.
Face photos are stored locally in `dataset/` and `avatars/` (both ignored by git).
