"""Streamlit wrapper for the live circular face camera (components/face_cam)."""
from pathlib import Path

import streamlit.components.v1 as components

_component = components.declare_component(
    "face_cam",
    path=str(Path(__file__).resolve().parent / "components" / "face_cam"),
)


def face_cam(mode, ready=True, result=None, key=None):
    """Show the live camera.

    mode:   "register" or "attend"
    ready:  register mode only - enables the Start button once name + father name are filled
    result: answer from Python for the last payload (must carry the payload's `nonce`)

    Returns the newest payload sent by the browser, or None:
        {"type": "register"|"attend", "nonce": str, "frames": [dataURL, ...]}
    """
    return _component(mode=mode, ready=bool(ready), result=result, key=key, default=None)
