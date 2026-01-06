from datetime import datetime
import pandas as pd
import streamlit as st

def init_state():
    if "budgets" not in st.session_state:
        st.session_state.budgets = {}

    if "spend_df" not in st.session_state:
        st.session_state.spend_df = pd.DataFrame(
            columns=["row_id", "date_time", "merchant", "item", "category", "amount"]
        )

    if "row_id_seq" not in st.session_state:
        st.session_state["row_id_seq"] = 0

    if "month_input" not in st.session_state:
        st.session_state["month_input"] = datetime.now().strftime("%Y-%m")

    if "pending_month" not in st.session_state:
        st.session_state["pending_month"] = None

    if "last_json_upload_key" not in st.session_state:
        st.session_state["last_json_upload_key"] = None

    if "total_budgets" not in st.session_state:
        st.session_state.total_budgets = {} 