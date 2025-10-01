import streamlit as st
import pandas as pd

from ui.theme import HIDE_UI_CSS, SIDEBAR_CSS
from views.home import render_home
from views.docs import render_docs


# Set Streamlit page config for wide layout
st.set_page_config(layout="wide", page_title="Election Cycle Seasonal Chart")

# Hide Streamlit Cloud toolbar and viewer/profile badges; also slim sidebar
st.markdown(HIDE_UI_CSS + SIDEBAR_CSS, unsafe_allow_html=True)


def main():
    # Load symbol list once
    stock_df = pd.read_csv("stocks.csv")

    # Sidebar navigation
    st.sidebar.markdown("<h2 style='margin-bottom:0'>Menu</h2>", unsafe_allow_html=True)
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    label = st.sidebar.selectbox(
        "",
        ["🏠 Home", "📄 Docs"],
        index=0 if st.session_state.page == "Home" else 1,
    )
    page = "Home" if label.startswith("🏠") else "Docs"
    st.session_state.page = page

    st.sidebar.markdown("---")
    st.sidebar.caption("Created by [Dy.](https://ramadhanep.com)")

    if page == "Home":
        render_home(stock_df)
    else:
        render_docs(stock_df)


if __name__ == "__main__":
    main()
