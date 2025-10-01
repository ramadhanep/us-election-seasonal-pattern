import streamlit as st
import pandas as pd


def render_docs(stock_df: pd.DataFrame):
    st.markdown("<h1>Docs</h1>", unsafe_allow_html=True)
    st.write("This app displays seasonal patterns by election cycle for your selected stock.")
    st.write("Symbols loaded from stocks.csv:")
    st.dataframe(stock_df, use_container_width=True)

    st.markdown("---")
    st.markdown("**Acknowledgment**")
    st.info(
        "Special thanks to Jeffrey A. Hirsch for the concept and style behind the 'Hirsch-style seasonal pattern.' "
        "This app follows that approach; I implemented the code for personal, non-commercial use only.")
