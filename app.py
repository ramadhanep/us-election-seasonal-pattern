import streamlit as st
import pandas as pd

from ui.theme import CUSTOM_CSS, SIDEBAR_CSS
from views.home import render_home
from views.docs import render_docs
from views.lock import render_lock


# Set Streamlit page config for wide layout
st.set_page_config(layout="wide", page_title="Election Cycle Seasonal Chart")

# Hide Streamlit Cloud toolbar and viewer/profile badges; slim sidebar
st.markdown(CUSTOM_CSS + SIDEBAR_CSS, unsafe_allow_html=True)


def main():
    # Load symbol list once
    stock_df = pd.read_csv("stocks.csv")

    # Auth gate: render lock screen until authenticated
    if not st.session_state.get("authenticated", False):
        with st.sidebar:
            st.markdown("<h2 style='margin-bottom:0'>Locked</h2>", unsafe_allow_html=True)
            st.caption("This software is private. Contact the owner to request access.")
        render_lock()
        return

    # Sidebar navigation (radio for better usability)
    st.sidebar.markdown("<h2 style='margin-bottom:0'>Menu</h2>", unsafe_allow_html=True)
    page = st.sidebar.radio("", ["Home", "Docs"], index=0)

    st.sidebar.markdown("---")
    st.sidebar.caption("Created by [Dy.](https://ramadhanep.com)")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.auth_name = None
        st.session_state.pin_input = ""
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

    if page == "Home":
        render_home(stock_df)
    else:
        render_docs(stock_df)


if __name__ == "__main__":
    main()
