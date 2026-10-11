"""Streamlit dashboard for the Face Recognition Attendance System.  Run: streamlit run app.py"""
from datetime import date, timedelta

import cv2
import pandas as pd
import streamlit as st

import database as db
import face_utils as fu

st.set_page_config(page_title="Face Attendance", page_icon="🧑‍💻", layout="wide")
db.init_db()

st.title("🧑‍💻 Face Recognition Attendance System")
tab_today, tab_reports, tab_students, tab_scan = st.tabs(
    ["📅 Today", "📊 Reports", "👥 Students", "📸 Take Attendance"]
)


def students_df():
    s = db.list_students()
    df = pd.DataFrame(s, columns=["id", "roll_no", "name", "created_at"])
    df["samples"] = [fu.sample_count(i) for i in df["id"]]
    return df


# ======================= TODAY =======================
with tab_today:
    today = date.today().isoformat()
    students = db.list_students()
    present = db.present_on(today)
    present_ids = {r["student_id"] for r in present}
    c1, c2, c3 = st.columns(3)
    c1.metric("Total students", len(students))
    c2.metric("Present today", len(present_ids))
    c3.metric("Absent today", max(len(students) - len(present_ids), 0))

    left, right = st.columns(2)
    with left:
        st.subheader("✅ Present")
        if present:
            df = pd.DataFrame(present)[["roll_no", "name", "time"]]
            st.dataframe(df, width="stretch", hide_index=True)
            st.download_button("Download CSV", df.to_csv(index=False), f"attendance_{today}.csv", "text/csv")
        else:
            st.info("Nobody marked present yet.")
    with right:
        st.subheader("❌ Absent")
        absent = [s for s in students if s["id"] not in present_ids]
        if absent:
            st.dataframe(pd.DataFrame(absent)[["roll_no", "name"]], width="stretch", hide_index=True)
        else:
            st.success("Everyone is present!" if students else "No students registered yet.")

# ======================= REPORTS =======================
with tab_reports:
    st.subheader("Attendance report")
    rng = st.date_input("Date range", value=(date.today() - timedelta(days=30), date.today()))
    if isinstance(rng, tuple) and len(rng) == 2:
        start, end = rng[0].isoformat(), rng[1].isoformat()
        rows = db.attendance_between(start, end)
        if rows:
            df = pd.DataFrame(rows)
            days = df["date"].nunique()
            counts = df.groupby(["roll_no", "name"]).size().reset_index(name="days_present")
            summary = students_df()[["roll_no", "name"]].merge(counts, on=["roll_no", "name"], how="left")
            summary["days_present"] = summary["days_present"].fillna(0).astype(int)
            summary["attendance_%"] = (summary["days_present"] / days * 100).round(1)
            st.caption(f"{days} class day(s) with recorded attendance in this range")
            st.bar_chart(summary.set_index("name")["attendance_%"])
            st.dataframe(summary, width="stretch", hide_index=True)
            st.markdown("**Detailed log**")
            st.dataframe(df[["date", "time", "roll_no", "name"]], width="stretch", hide_index=True)
            st.download_button("Download full log (CSV)", df.to_csv(index=False), f"attendance_{start}_{end}.csv", "text/csv")
        else:
            st.info("No attendance records in this range.")
    else:
        st.info("Select both a start and an end date.")

# ======================= STUDENTS =======================
with tab_students:
    st.subheader("Registered students")
    sdf = students_df()
    if sdf.empty:
        st.info("No students yet. Add one below.")
    else:
        st.dataframe(sdf[["roll_no", "name", "samples", "created_at"]], width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("➕ Register a student")
    st.caption("Upload 5+ clear photos (front-facing, good light) or take snapshots. "
               "For the best results use `register.py` with the webcam (40 samples).")
    with st.form("register_form", clear_on_submit=True):
        roll = st.text_input("Roll number")
        name = st.text_input("Full name")
        uploads = st.file_uploader("Face photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        snap = st.camera_input("Or take a photo (optional)")
        submitted = st.form_submit_button("Register & train")
    if submitted:
        if not roll.strip() or not name.strip():
            st.error("Roll number and name are required.")
        else:
            images = [u.getvalue() for u in uploads] + ([snap.getvalue()] if snap else [])
            faces = []
            for data in images:
                img = fu.decode_image(data)
                if img is None:
                    continue
                gray = fu.to_gray(img)
                box = fu.largest_face(fu.detect_faces(gray))
                if box is not None:
                    faces.append(fu.preprocess(gray, box))
            if not faces:
                st.error("No face detected in the provided photos. Use clear, front-facing photos.")
            else:
                existing = db.get_student_by_roll(roll)
                sid = existing["id"] if existing else db.add_student(roll, name)
                fu.save_samples(sid, faces)
                total = fu.train()
                st.success(f"Registered {name} with {len(faces)} face(s). Model trained on {total} images.")
                if len(faces) < 5:
                    st.warning("Only a few faces were found. Add more photos for better accuracy.")
                st.rerun()

    st.markdown("---")
    st.subheader("🗑 Delete a student")
    if not sdf.empty:
        options = {f"{r.roll_no} - {r.name}": int(r.id) for r in sdf.itertuples()}
        choice = st.selectbox("Student", list(options))
        sure = st.checkbox("I understand this deletes the student, their photos and attendance records")
        if st.button("Delete", disabled=not sure):
            sid = options[choice]
            db.delete_student(sid)
            fu.remove_student_data(sid)
            try:
                fu.train()
            except ValueError:
                pass            # no students left
            st.success("Deleted.")
            st.rerun()

# ======================= TAKE ATTENDANCE =======================
with tab_scan:
    st.subheader("Mark attendance from a photo")
    st.caption("Take a snapshot (or upload a class photo). Every recognised student is marked present for today.")
    rec = fu.load_recognizer()
    if rec is None:
        st.warning("No trained model yet. Register at least one student first.")
    else:
        shot = st.camera_input("Camera")
        upload = st.file_uploader("...or upload a photo", type=["jpg", "jpeg", "png"], key="scan_upload")
        data = shot.getvalue() if shot else (upload.getvalue() if upload else None)
        if data:
            frame = fu.decode_image(data)
            names = {s["id"]: s["name"] for s in db.list_students()}
            results = fu.recognize(frame, rec)
            if not results:
                st.warning("No face detected. Try again with better lighting.")
            else:
                for r in results:
                    sid = r["student_id"]
                    if sid in names:
                        if db.mark_attendance(sid):
                            st.success(f"✅ {names[sid]} marked present")
                        else:
                            st.info(f"ℹ️ {names[sid]} was already marked today")
                    else:
                        st.error("❓ Unknown face (not registered)")
                st.image(cv2.cvtColor(fu.annotate(frame, results, names), cv2.COLOR_BGR2RGB), width="stretch")
