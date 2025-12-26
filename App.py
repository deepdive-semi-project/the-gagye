import streamlit as st
import pandas as pd

st.set_page_config(page_title="The 가계", page_icon=":memo:")

with st.container(horizontal_alignment="center"):
    st.title(
        ":orange[:material/checklist:] The 가계",
        width="content",
    )

df = pd.read_csv("data/expenses_20251224.csv")

st.dataframe(df)
