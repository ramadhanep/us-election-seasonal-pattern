import streamlit as st
from typing import Optional

from core.auth import verify_pin


def _mask_pin(pin: str) -> str:
    return "•" * len(pin)


def _init_state():
    if "pin_input" not in st.session_state:
        st.session_state.pin_input = ""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "auth_name" not in st.session_state:
        st.session_state.auth_name = None


def _append_digit(d: str):
    if len(st.session_state.pin_input) < 6 and d.isdigit():
        st.session_state.pin_input += d


def _backspace():
    st.session_state.pin_input = st.session_state.pin_input[:-1]


def _clear():
    st.session_state.pin_input = ""


def _try_submit() -> Optional[str]:
    pin = st.session_state.pin_input
    if len(pin) != 6:
        return None
    name = verify_pin(pin)
    if name:
        st.session_state.authenticated = True
        st.session_state.auth_name = name
        try:
            st.toast(f"Welcome, {name}.")
        except Exception:
            st.success(f"Welcome, {name}.")
        return name
    else:
        st.error("Invalid PIN. Please try again.")
        return None


def render_lock():
    _init_state()

    st.markdown("<h1 style='margin-bottom:0'>Access Required</h1>", unsafe_allow_html=True)
    st.caption("This is a personal tool. Contact the owner to request access.")

    # PIN display
    pin_box = _mask_pin(st.session_state.pin_input).ljust(6, "◦")
    st.markdown(f"<div style='font-size:2rem;letter-spacing:0.5rem;'>{pin_box}</div>", unsafe_allow_html=True)

    # Numpad layout
    rows = [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"], ["Clear", "0", "Back"]]
    for row in rows:
        cols = st.columns(3)
        for i, key in enumerate(row):
            if cols[i].button(key, use_container_width=True):
                if key.isdigit():
                    _append_digit(key)
                elif key == "Back":
                    _backspace()
                elif key == "Clear":
                    _clear()

    # Submit row
    submit_col = st.columns([1, 1, 1])[1]
    if submit_col.button("Enter", use_container_width=True):
        name = _try_submit()
        if name:
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

    # Auto-submit when 6 digits reached
    if len(st.session_state.pin_input) == 6 and not st.session_state.authenticated:
        name = _try_submit()
        if name:
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

    # Contact/owner notice
    st.markdown("---")
    st.info(
        "This software is private and for personal use only. "
        "If you need access, please contact the owner.")
