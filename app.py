
"""Streamlit dashboard for the Face Recognition Attendance System."""

from datetime import date, timedelta
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

import database as db
import face_utils as fu


# ===================== PAGE CONFIG =====================

st.set_page_config(
    page_title="Face Attendance",
    page_icon="FA",
    layout="wide",
)


# ===================== LOAD EXTERNAL CSS =====================

def load_css():
    """Load dashboard styles from style.css."""
    css_path = Path(__file__).resolve().parent / "style.css"

    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        st.markdown(
            f"<style>{css}</style>",
            unsafe_allow_html=True,
        )


load_css()


# ===================== DATABASE INITIALIZATION =====================

db.init_db()


# ===================== DASHBOARD HEADER =====================

st.markdown(
    """
    <div class="hero-box">
        <div class="hero-label">ATTENDANCE MANAGEMENT</div>
        <h1>Face Attendance System</h1>
        <p>
            Intelligent attendance tracking, student management,
            and attendance reporting in one place.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ===================== NAVIGATION TABS =====================

tab_today, tab_reports, tab_students, tab_scan = st.tabs(
    [
        "Overview",
        "Reports",
        "Students",
        "Take Attendance",
    ]
)


# ===================== HELPER FUNCTIONS =====================

def students_df():
    """Return registered students and their face sample counts."""
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


# ===================== OVERVIEW =====================

with tab_today:
    today = date.today().isoformat()

    students = db.list_students()
    present = db.present_on(today)

    present_ids = {
        row["student_id"]
        for row in present
    }

    total_students = len(students)
    total_present = len(present_ids)
    total_absent = max(total_students - total_present, 0)

    st.subheader("Today's Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Students",
        total_students,
    )

    col2.metric(
        "Present Today",
        total_present,
    )

    col3.metric(
        "Absent Today",
        total_absent,
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Present Students")

        if present:
            present_df = pd.DataFrame(present)[
                ["roll_no", "name", "time"]
            ]

            st.dataframe(
                present_df,
                width="stretch",
                hide_index=True,
            )

            st.download_button(
                "Download Attendance CSV",
                data=present_df.to_csv(index=False),
                file_name=f"attendance_{today}.csv",
                mime="text/csv",
                key="download_present",
            )

        else:
            st.info("No students have been marked present today.")

    with right:
        st.subheader("Absent Students")

        absent = [
            student
            for student in students
            if student["id"] not in present_ids
        ]

        if absent:
            absent_df = pd.DataFrame(absent)[
                ["roll_no", "name"]
            ]

            st.dataframe(
                absent_df,
                width="stretch",
                hide_index=True,
            )

        elif students:
            st.success("All registered students are present.")

        else:
            st.info("No students have been registered yet.")


# ===================== ATTENDANCE REPORTS =====================

with tab_reports:
    st.subheader("Attendance Reports")

    st.caption(
        "Review attendance records and student attendance percentages."
    )

    date_range = st.date_input(
        "Select Date Range",
        value=(
            date.today() - timedelta(days=30),
            date.today(),
        ),
        key="report_date_range",
    )

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date = date_range[0].isoformat()
        end_date = date_range[1].isoformat()

        if start_date > end_date:
            st.error("The start date must be before the end date.")

        else:
            rows = db.attendance_between(
                start_date,
                end_date,
            )

            if rows:
                report_df = pd.DataFrame(rows)

                days = max(
                    report_df["date"].nunique(),
                    1,
                )

                counts = (
                    report_df.groupby(["roll_no", "name"])
                    .size()
                    .reset_index(name="days_present")
                )

                summary = students_df()[
                    ["roll_no", "name"]
                ].merge(
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
                    "in the selected range."
                )

                st.subheader("Attendance Percentage")

                chart_df = summary.set_index("name")[
                    ["attendance_%"]
                ]

                st.bar_chart(chart_df)

                st.subheader("Student Summary")

                st.dataframe(
                    summary,
                    width="stretch",
                    hide_index=True,
                )

                st.subheader("Detailed Attendance Log")

                details = report_df[
                    ["date", "time", "roll_no", "name"]
                ]

                st.dataframe(
                    details,
                    width="stretch",
                    hide_index=True,
                )

                st.download_button(
                    "Download Full Report CSV",
                    data=report_df.to_csv(index=False),
                    file_name=(
                        f"attendance_{start_date}_{end_date}.csv"
                    ),
                    mime="text/csv",
                    key="download_full_report",
                )

            else:
                st.info(
                    "No attendance records were found "
                    "for the selected date range."
                )

    else:
        st.info("Select both a start date and an end date.")


# ===================== STUDENT MANAGEMENT =====================

with tab_students:
    st.subheader("Registered Students")

    sdf = students_df()

    if sdf.empty:
        st.info(
            "No students have been registered yet. "
            "Use the form below to add a student."
        )

    else:
        st.dataframe(
            sdf[
                ["roll_no", "name", "samples", "created_at"]
            ],
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "Download Student List CSV",
            data=sdf[
                ["roll_no", "name", "samples", "created_at"]
            ].to_csv(index=False),
            file_name="registered_students.csv",
            mime="text/csv",
            key="download_students",
        )

    st.divider()

    st.subheader("Register a Student")

    st.caption(
        "Upload clear, front-facing photos with good lighting. "
        "Multiple photos can improve recognition performance."
    )

    with st.form(
        "register_form",
        clear_on_submit=True,
    ):
        roll = st.text_input(
            "Roll Number",
            placeholder="Enter student roll number",
        )

        name = st.text_input(
            "Full Name",
            placeholder="Enter student's full name",
        )

        uploads = st.file_uploader(
            "Upload Face Photos",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

        snapshot = st.camera_input(
            "Or Take a Student Photo",
        )

        submitted = st.form_submit_button(
            "Register Student and Train Model",
        )

    if submitted:
        if not roll.strip() or not name.strip():
            st.error(
                "Both roll number and full name are required."
            )

        else:
            uploaded_data = [
                file.getvalue()
                for file in (uploads or [])
            ]

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
                        faces.append(
                            fu.preprocess(gray, box)
                        )

                if not faces:
                    st.error(
                        "No face was detected. Upload clear, "
                        "front-facing photos with good lighting."
                    )

                else:
                    existing = db.get_student_by_roll(
                        roll.strip()
                    )

                    if existing:
                        student_id = existing["id"]

                        st.info(
                            "This roll number already exists. "
                            "The new face samples will be added "
                            "to the existing student."
                        )

                    else:
                        student_id = db.add_student(
                            roll.strip(),
                            name.strip(),
                        )

                    fu.save_samples(
                        student_id,
                        faces,
                    )

                    total = fu.train()

                    st.success(
                        f"Student {name.strip()} registered. "
                        f"Usable photos: {len(faces)}. "
                        f"Total training images: {total}."
                    )

                    if len(faces) < 5:
                        st.warning(
                            "Fewer than 5 usable photos were found. "
                            "Consider adding more photos from "
                            "different angles and lighting conditions."
                        )

                    st.rerun()

            except Exception as exc:
                st.error(
                    f"Student registration failed: {exc}"
                )

    st.divider()

    st.subheader("Delete a Student")

    if not sdf.empty:
        options = {
            f"{row.roll_no} - {row.name}": int(row.id)
            for row in sdf.itertuples()
        }

        choice = st.selectbox(
            "Select Student",
            list(options.keys()),
            key="delete_student_choice",
        )

        confirmed = st.checkbox(
            "I understand that this deletes the student, "
            "their photos, and attendance records.",
            key="confirm_delete_student",
        )

        if st.button(
            "Delete Student",
            disabled=not confirmed,
            key="delete_student_button",
        ):
            student_id = options[choice]

            try:
                db.delete_student(student_id)
                fu.remove_student_data(student_id)

                try:
                    fu.train()
                except ValueError:
                    # No training samples remain.
                    pass

                st.success("Student deleted successfully.")
                st.rerun()

            except Exception as exc:
                st.error(
                    f"Could not delete the student: {exc}"
                )


# ===================== TAKE ATTENDANCE =====================

with tab_scan:
    st.subheader("Take Attendance")

    st.caption(
        "Capture a photo or upload an image to identify registered "
        "students. Review the results before relying on attendance records."
    )

    try:
        recognizer = fu.load_recognizer()

    except Exception as exc:
        recognizer = None

        st.error(
            f"Could not load the recognition model: {exc}"
        )

    if recognizer is None:
        st.warning(
            "No trained model was found. Register students with "
            "face photos before taking attendance."
        )

    else:
        snapshot = st.camera_input(
            "Capture Attendance Photo",
            key="attendance_camera",
        )

        upload = st.file_uploader(
            "Or Upload an Attendance Photo",
            type=["jpg", "jpeg", "png"],
            key="scan_upload",
        )

        image_data = (
            snapshot.getvalue()
            if snapshot is not None
            else (
                upload.getvalue()
                if upload is not None
                else None
            )
        )

        if image_data:
            frame = fu.decode_image(image_data)

            if frame is None:
                st.error(
                    "The uploaded image could not be read."
                )

            else:
                students = db.list_students()

                names = {
                    student["id"]: student["name"]
                    for student in students
                }

                try:
                    results = fu.recognize(
                        frame,
                        recognizer,
                    )

                    if not results:
                        st.warning(
                            "No faces were detected. Try a clearer "
                            "photo with better lighting."
                        )

                    else:
                        st.subheader("Recognition Results")

                        for result in results:
                            student_id = result["student_id"]

                            if (
                                student_id is not None
                                and student_id in names
                            ):
                                marked = db.mark_attendance(
                                    student_id
                                )

                                if marked:
                                    st.success(
                                        f"{names[student_id]} "
                                        "marked present."
                                    )

                                else:
                                    st.info(
                                        f"{names[student_id]} "
                                        "was already marked present today."
                                    )

                            else:
                                st.warning(
                                    "Unknown face detected. "
                                    "Verify the person's identity manually."
                                )

                        annotated = fu.annotate(
                            frame.copy(),
                            results,
                            names,
                        )

                        st.subheader("Processed Image")

                        st.image(
                            cv2.cvtColor(
                                annotated,
                                cv2.COLOR_BGR2RGB,
                            ),
                            width="stretch",
                        )

                except Exception as exc:
                    st.error(
                        f"Attendance scanning failed: {exc}"
                    )
