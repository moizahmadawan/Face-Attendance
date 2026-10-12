"""Small HTML builders for the dashboard (cards, rows, ID card)."""
from html import escape

PALETTE = ["#3ee0b0", "#5cc8ff", "#f5c25c", "#ff8fa3", "#b69cff", "#7be08a"]


def initials(name):
    parts = [p for p in (name or "").split() if p]
    return (parts[0][0] + (parts[1][0] if len(parts) > 1 else "")).upper() if parts else "?"


def avatar(name, uri=None, size=46):
    if uri:
        return f'<img class="av" src="{uri}" alt="" style="width:{size}px;height:{size}px">'
    color = PALETTE[sum(map(ord, name or "?")) % len(PALETTE)]
    return (f'<div class="av av-text" style="width:{size}px;height:{size}px;'
            f'background:{color}22;color:{color};border-color:{color}55">{escape(initials(name))}</div>')


def student_grid(students, avatars, samples):
    if not students:
        return ""
    cards = []
    for s in students:
        father = escape(s.get("father_name") or "-")
        cards.append(
            f'<div class="scard">{avatar(s["name"], avatars.get(s["id"]), 58)}'
            f'<div class="scard-body"><div class="scard-name">{escape(s["name"])}</div>'
            f'<div class="scard-sub">S/O {father}</div>'
            f'<div class="scard-meta"><span class="tag tag-roll">{escape(s["roll_no"])}</span>'
            f'<span class="tag">{samples.get(s["id"], 0)} scans</span>'
            f'<span class="tag">{escape(s["created_at"][:10])}</span></div></div></div>'
        )
    return f'<div class="sgrid">{"".join(cards)}</div>'


def person_rows(people, avatars, right_key=None, tone="present"):
    """people: dicts with name, roll_no, id/student_id and optional time."""
    rows = []
    for p in people:
        sid = p.get("student_id", p.get("id"))
        right = p.get(right_key, "") if right_key else ""
        chip = f'<span class="tag tag-{tone}">{escape(str(right)[:5])}</span>' if right else \
               f'<span class="tag tag-{tone}">{"Present" if tone == "present" else "Absent"}</span>'
        rows.append(
            f'<div class="prow prow-{tone}">{avatar(p["name"], avatars.get(sid), 40)}'
            f'<div class="prow-body"><div class="prow-name">{escape(p["name"])}</div>'
            f'<div class="prow-sub">{escape(p["roll_no"])}</div></div>{chip}</div>'
        )
    return f'<div class="plist">{"".join(rows)}</div>'


def id_card(roll, name="", father=""):
    n = escape(name.strip()) if name.strip() else "Student name"
    f = escape(father.strip()) if father.strip() else "Father's name"
    dim_n = "" if name.strip() else " dim"
    dim_f = "" if father.strip() else " dim"
    return (
        '<div class="idcard"><div class="idcard-top"><span class="idcard-label">Roll number (auto)</span>'
        '<span class="idcard-chip"></span></div>'
        f'<div class="idcard-roll">{escape(roll)}</div>'
        f'<div class="idcard-name{dim_n}">{n}</div>'
        f'<div class="idcard-sub{dim_f}">S/O {f}</div></div>'
    )
