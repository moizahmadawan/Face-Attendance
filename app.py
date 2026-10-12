"""Streamlit dashboard for the Face Recognition Attendance System."""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

import database as db
import face_utils as fu
import services
import ui
from face_cam import face_cam

# ===================== PAGE CONFIG =====================

st.set_page_config(page_title="Face Attendance", page_icon="📸", layout="wide")


def load_css():
    css_path = Path(__file__).resolve().parent / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


load_css()
db.init_db()

# ===================== HEADER =====================

st.markdown(
    """
    <div class="hero-box">
        <svg class="hero-icon" viewBox="0 0 120 120" fill="none"
             xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
            <path d="M8 36V16a8 8 0 0 1 8-8h20M84 8h20a8 8 0 0 1 8 8v20M112 84v20a8 8 0 0 1-8 8H84M36 112H16a8 8 0 0 1-8-8V84"
                  stroke="#3ee0b0" stroke-width="5" stroke-linecap="round"/>
            <circle cx="60" cy="48" r="16" stroke="#9fd8ff" stroke-width="4"/>
            <path d="M30 94c4-17 15-26 30-26s26 9 30 26" stroke="#9fd8ff"
                  stroke-width="4" stroke-linecap="round"/>
            <rect class="scan" x="20" y="22" width="80" height="3" rx="1.5" fill="#3ee0b0"/>
        </svg>
        <div class="hero-label">Live face recognition</div>
        <h1>Face Attendance System</h1>
        <p>Register once with a live face scan. Attendance is marked the moment a face appears
           and saved automatically every day.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_today, tab_scan, tab_register, tab_students, tab_reports = st.tabs(
    ["Overview", "Attendance", "Register", "Students", "Reports"]
)


# ===================== HELPERS =====================

def avatar_map(students):
    return {s["id"]: fu.avatar_data_uri(s["id"]) for s in students}


def students_df():
    students = db.list_students()
    df = pd.DataFrame(students, columns=["id", "roll_no", "name", "father_name", "created_at"])
    df["samples"] = [fu.sample_count(i) for i in df["id"]] if not df.empty else pd.Series(dtype="int64")
    return df


today = date.today().isoformat()

# ===================== OVERVIEW =====================

with tab_today:
    students = db.list_students()
    present = db.present_on(today)
    present_ids = {r["student_id"] for r in present}
    absent = [s for s in students if s["id"] not in present_ids]
    avatars = avatar_map(students)

    st.subheader("Today's overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total students", len(students))
    c2.metric("Present today", len(present_ids))
    c3.metric("Absent today", len(absent))

    st.write("")
    left, right = st.columns(2)
    with left:
        st.subheader("Present")
        if present:
            st.markdown(ui.person_rows(present, avatars, "time", "present"), unsafe_allow_html=True)
        else:
            st.info("Nobody has been marked present yet today.")
    with right:
        st.subheader("Absent")
        if absent:
            st.markdown(ui.person_rows(absent, avatars, None, "absent"), unsafe_allow_html=True)
        elif students:
            st.success("Everyone is present today.")
        else:
            st.info("No students registered yet. Open the Register tab to add one.")

    st.write("")
    if students:
        st.download_button(
            "Download today's attendance (CSV)",
            data=db.day_csv_text(today),
            file_name=f"attendance_{today}.csv",
            mime="text/csv",
            key="dl_today",
        )
        st.caption(f"Saved automatically to exports/attendance_{today}.csv every time someone is marked.")

# ===================== TAKE ATTENDANCE =====================

with tab_scan:
    st.subheader("Take attendance")
    st.caption("Tap start and stand in front of the camera. Your face is detected inside the circle and "
               "attendance is marked automatically.")

    payload = face_cam("attend", result=st.session_state.get("scan_result"), key="scan_cam")
    if payload and payload.get("type") == "attend" and payload.get("nonce") != st.session_state.get("scan_last"):
        st.session_state["scan_last"] = payload["nonce"]
        st.session_state["scan_result"] = services.scan_attendance(payload["nonce"], payload.get("frames"))
        st.rerun()

    st.write("")
    st.subheader("Checked in today")
    now_present = db.present_on(today)
    if now_present:
        st.markdown(ui.person_rows(now_present[:8], avatar_map(db.list_students()), "time", "present"),
                    unsafe_allow_html=True)
    else:
        st.info("No attendance marked yet today.")

# ===================== REGISTER =====================

with tab_register:
    if st.session_state.pop("_clear_register", False):
        st.session_state["reg_name"] = ""
        st.session_state["reg_father"] = ""

    st.subheader("Register a student")
    st.caption("Enter the details, then verify with a live face scan. The roll number is created automatically.")

    form_col, cam_col = st.columns([1, 1], gap="large")
    with form_col:
        name = st.text_input("Student name", key="reg_name", placeholder="e.g. Ali Ahmad")
        father = st.text_input("Father's name", key="reg_father", placeholder="e.g. Ahmad Khan")
        roll_preview = db.next_roll()
        st.markdown(ui.id_card(roll_preview, name, father), unsafe_allow_html=True)
        st.caption("Verification takes about 15 seconds: look straight, then turn left, right, up and down.")

    with cam_col:
        ready = len(name.strip()) >= 2 and len(father.strip()) >= 2
        payload = face_cam("register", ready=ready, result=st.session_state.get("reg_result"), key="reg_cam")
        if payload and payload.get("type") == "register" and payload.get("nonce") != st.session_state.get("reg_last"):
            st.session_state["reg_last"] = payload["nonce"]
            result = services.register_live(payload["nonce"], name, father, payload.get("frames"))
            st.session_state["reg_result"] = result
            if result.get("ok"):
                st.session_state["_clear_register"] = True
            st.rerun()

# ===================== STUDENTS =====================

with tab_students:
    st.subheader("Registered students")
    sdf = students_df()
    all_students = db.list_students()

    if sdf.empty:
        st.info("No students registered yet. Use the Register tab to add the first one.")
    else:
        st.markdown(
            ui.student_grid(all_students, avatar_map(all_students),
                            dict(zip(sdf["id"], sdf["samples"]))),
            unsafe_allow_html=True,
        )
        st.write("")
       st.download_button(
            "Download student list (CSV)",
            data=sdf[["roll_no", "name", "father_name", "samples", "created_at"]].to_csv(index=False),
            file_name="registered_students.csv",
            mime="text/csv",
            key="dl_students",
        )

       st.divider()
        st.subheader("Remove a student")
        options = {f"{r.roll_no} - {r.name}": int(r.id) for r in sdf.itertuples()}
        choice = st.selectbox("Select student", list(options.keys()), key="delete_student_choice")
        confirmed = st.checkbox(
            "I understand this deletes the student, their face scans and attendance records.",
            key="confirm_delete_student",
        )
        if st.button("Delete student", disabled=not confirmed, key="delete_student_button"):
            student_id = options[choice]
            try:
                db.delete_student(student_id)
                fu.remove_student_data(student_id)
                try:
                    fu.train()
                except ValueError:
                    pass                      # no students left
                st.session_state["flash"] = "Student deleted."
                st.rerun()
            except Exception as exc:
                st.error(f"Could not delete the student: {exc}")

        if st.session_state.get("flash"):
            st.success(st.session_state.pop("flash"))

# ===================== REPORTS =====================

with tab_reports:
    st.subheader("Daily sheet")
    sheet_day = st.date_input("Day", value=date.today(), key="sheet_day")
    sheet_iso = sheet_day.isoformat()
    sheet = pd.DataFrame(db.day_sheet(sheet_iso))
    if sheet.empty:
        st.info("No students registered yet.")
    else:
        st.dataframe(sheet, width="stretch", hide_index=True)
        st.download_button(
            "Download this day (CSV)",
            data=db.day_csv_text(sheet_iso),
            file_name=f"attendance_{sheet_iso}.csv",
            mime="text/csv",
            key="dl_sheet",
        )

    st.divider()
    st.subheader("Date range report")
    date_range = st.date_input(
        "Select date range",
        value=(date.today() - timedelta(days=30), date.today()),
        key="report_date_range",
    )

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range[0].isoformat(), date_range[1].isoformat()
        if start_date > end_date:
            st.error("The start date must be before the end date.")
        else:
            rows = db.attendance_between(start_date, end_date)
            if rows:
                report_df = pd.DataFrame(rows)
                days = max(report_df["date"].nunique(), 1)
                counts = (
                    report_df.groupby(["roll_no", "name"]).size().reset_index(name="days_present")
                )
                summary = students_df()[["roll_no", "name", "father_name"]].merge(
                    counts, on=["roll_no", "name"], how="left"
                )
                summary["days_present"] = summary["days_present"].fillna(0).astype(int)
                summary["attendance_%"] = (summary["days_present"] / days * 100).round(1)

                st.caption(f"{days} day(s) with recorded attendance in this range.")
                st.bar_chart(
                    summary.assign(student=summary["roll_no"] + " - " + summary["name"])
                    .set_index("student")[["attendance_%"]]
                )
                st.dataframe(summary, width="stretch", hide_index=True)
                st.download_button(
                    "Download full report (CSV)",
                    data=report_df.drop(columns=["student_id"]).to_csv(index=False),
                    file_name=f"attendance_{start_date}_{end_date}.csv",
                    mime="text/csv",
                    key="dl_range",
                )
            else:
                st.info("No attendance records were found for the selected dates.")
    else:
        st.info("Select both a start date and an end date.")
