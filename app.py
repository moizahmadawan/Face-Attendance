
"""Streamlit dashboard for the Face Recognition Attendance System."""

from datetime import date, timedelta

import cv2
import pandas as pd
import streamlit as st

import database as db
import face_utils as fu

st.set_page_config(
    page_title="Face Attendance",
    page_icon="🧑‍💻",
    layout="wide",
)

db.init_db()

st.title("🧑‍💻 Face Recognition Attendance System")

tab_today, tab_reports, tab_students, tab_scan = st.tabs(
    ["📅 Today", "📊 Reports", "👥 Students", "📸 Take Attendance"]
)


def students_df():
    students = db.list_students()

    columns = ["id", "roll_no", "name", "created_at"]
    df = pd.DataFrame(students, columns=columns)

    if not df.empty:
        df["samples"] = [
            fu.sample_count(student_id)
            for student_id in df["id"]
        ]
    else:
        df["samples"] = pd.Series(dtype="int64")

    return df


# ===================== TODAY =====================
with tab_today:
    today = date.today().isoformat()

    students = db.list_students()
    present = db.present_on(today)

    present_ids = {row["student_id"] for row in present}

    col1, col2, col3 = st.columns(3)

    col1.metric("Total students", len(students))
    col2.metric("Present today", len(present_ids))
    col3.metric(
        "Absent today",
        max(len(students) - len(present_ids), 0),
    )

    left, right = st.columns(2)

    with left:
        st.subheader("✅ Present")

        if present:
            df = pd.DataFrame(present)[
                ["roll_no", "name", "time"]
            ]

            st.dataframe(
                df,
                width="stretch",
                hide_index=True,
            )

            st.download_button(
                "Download CSV",
                data=df.to_csv(index=False),
                file_name=f"attendance_{today}.csv",
                mime="text/csv",
            )
        else:
            st.info("Nobody marked present yet.")

    with right:
        st.subheader("❌ Absent")

        absent = [
            student
            for student in students
            if student["id"] not in present_ids
        ]

        if absent:
            st.dataframe(
                pd.DataFrame(absent)[["roll_no", "name"]],
                width="stretch",
                hide_index=True,
            )
        elif students:
            st.success("Everyone is present!")
        else:
            st.info("No students registered yet.")


# ===================== REPORTS =====================
with tab_reports:
    st.subheader("Attendance report")

    date_range = st.date_input(
        "Date range",
        value=(
            date.today() - timedelta(days=30),
            date.today(),
        ),
    )

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date = date_range[0].isoformat()
        end_date = date_range[1].isoformat()

        if start_date > end_date:
            st.error("Start date must be before end date.")
        else:
            rows = db.attendance_between(start_date, end_date)

            if rows:
                report_df = pd.DataFrame(rows)
                days = max(report_df["date"].nunique(), 1)

                counts = (
                    report_df.groupby(["roll_no", "name"])
                    .size()
                    .reset_index(name="days_present")
                )

                summary = students_df()[["roll_no", "name"]].merge(
                    counts,
                    on=["roll_no", "name"],
                    how="left",
                )

                summary["days_present"] = (
                    summary["days_present"]
                    .fillna(0)
                    .astype(int)
                )

                summary["attendance_%"] = (
                    summary["days_present"] / days * 100
                ).round(1)

                st.caption(
                    f"{days} date(s) with recorded attendance "
                    "in this range"
                )

                st.bar_chart(
                    summary.set_index("name")["attendance_%"]
                )

                st.dataframe(
                    summary,
                    width="stretch",
                    hide_index=True,
                )

                st.markdown("**Detailed log**")

                details = report_df[
                    ["date", "time", "roll_no", "name"]
                ]

                st.dataframe(
                    details,
                    width="stretch",
                    hide_index=True,
                )

                st.download_button(
                    "Download full log (CSV)",
                    data=report_df.to_csv(index=False),
                    file_name=f"attendance_{start_date}_{end_date}.csv",
                    mime="text/csv",
                )
            else:
                st.info("No attendance records in this range.")
    else:
        st.info("Select both a start date and an end date.")


# ===================== STUDENTS =====================
with tab_students:
    st.subheader("Registered students")

    sdf = students_df()

    if sdf.empty:
        st.info("No students yet. Add one below.")
    else:
        st.dataframe(
            sdf[["roll_no", "name", "samples", "created_at"]],
            width="stretch",
            hide_index=True,
        )

    st.divider()
    st.subheader("➕ Register a student")

    st.caption(
        "Upload clear, front-facing photos in good lighting. "
        "Use multiple photos for better recognition."
    )

    with st.form("register_form", clear_on_submit=True):
        roll = st.text_input("Roll number")
        name = st.text_input("Full name")

        uploads = st.file_uploader(
            "Upload face photos",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

        snapshot = st.camera_input("Or take a photo")

        submitted = st.form_submit_button("Register & train")

    if submitted:
        if not roll.strip() or not name.strip():
            st.error("Roll number and name are required.")
        else:
            uploaded_data = [file.getvalue() for file in (uploads or [])]

            if snapshot is not None:
                uploaded_data.append(snapshot.getvalue())

            faces = []

            try:
                for image_data in uploaded_data:
                    image = fu.decode_image(image_data)

                    if image is None:
                        continue

                    gray = fu.to_gray(image)
                    boxes = fu.detect_faces(gray)
                    box = fu.largest_face(boxes)

                    if box is not None:
                        faces.append(fu.preprocess(gray, box))

                if not faces:
                    st.error(
                        "No face detected. Upload clear, front-facing "
                        "photos with good lighting."
                    )
                else:
                    existing = db.get_student_by_roll(roll.strip())

                    if existing:
                        student_id = existing["id"]
                        st.info(
                            "This roll number already exists. "
                            "New face samples will be added."
                        )
                    else:
                        student_id = db.add_student(
                            roll.strip(),
                            name.strip(),
                        )

                    fu.save_samples(student_id, faces)
                    total = fu.train()

                    st.success(
                        f"Registered {name.strip()}. "
                        f"Found {len(faces)} usable face photo(s). "
                        f"Model trained on {total} image(s)."
                    )

                    if len(faces) < 5:
                        st.warning(
                            "Fewer than 5 usable photos were found. "
                            "Add more varied photos for better results."
                        )

                    st.rerun()

            except Exception as exc:
                st.error(f"Registration failed: {exc}")

    st.divider()
    st.subheader("🗑 Delete a student")

    if not sdf.empty:
        options = {
            f"{row.roll_no} - {row.name}": int(row.id)
            for row in sdf.itertuples()
        }

        choice = st.selectbox("Select student", list(options.keys()))

        confirmed = st.checkbox(
            "I understand this deletes the student, their photos "
            "and attendance records."
        )

        if st.button("Delete student", disabled=not confirmed):
            student_id = options[choice]

            try:
                db.delete_student(student_id)
                fu.remove_student_data(student_id)

                try:
                    fu.train()
                except ValueError:
                    # No training samples remain.
                    pass

                st.success("Student deleted.")
                st.rerun()

            except Exception as exc:
                st.error(f"Could not delete student: {exc}")


# ===================== TAKE ATTENDANCE =====================
with tab_scan:
    st.subheader("📸 Mark attendance from a photo")

    st.caption(
        "Capture a photo or upload a class photo. "
        "Check recognition results before relying on attendance records."
    )

    try:
        recognizer = fu.load_recognizer()
    except Exception as exc:
        recognizer = None
        st.error(f"Could not load the recognition model: {exc}")

    if recognizer is None:
        st.warning(
            "No trained model found. Register students with face photos first."
        )
    else:
        snapshot = st.camera_input("Take attendance photo")

        upload = st.file_uploader(
            "Or upload a photo",
            type=["jpg", "jpeg", "png"],
            key="scan_upload",
        )

        image_data = (
            snapshot.getvalue()
            if snapshot is not None
            else upload.getvalue() if upload is not None else None
        )

        if image_data:
            frame = fu.decode_image(image_data)

            if frame is None:
                st.error("The uploaded image could not be read.")
            else:
                students = db.list_students()
                names = {
                    student["id"]: student["name"]
                    for student in students
                }

                try:
                    results = fu.recognize(frame, recognizer)

                    if not results:
                        st.warning(
                            "No faces detected. Try a clearer photo "
                            "with better lighting."
                        )
                    else:
                        for result in results:
                            student_id = result["student_id"]

                            if (
                                student_id is not None
                                and student_id in names
                            ):
                                marked = db.mark_attendance(student_id)

                                if marked:
                                    st.success(
                                        f"✅ {names[student_id]} marked present."
                                    )
                                else:
                                    st.info(
                                        f"ℹ️ {names[student_id]} "
                                        "was already marked today."
                                    )
                            else:
                                st.warning(
                                    "❓ Unknown face. Verify identity manually."
                                )

                        annotated = fu.annotate(
                            frame.copy(),
                            results,
                            names,
                        )

                        st.image(
                            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                            width="stretch",
                        )

                except Exception as exc:
                    st.error(f"Attendance scanning failed: {exc}")
