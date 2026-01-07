# utils_state.py
from datetime import datetime
import pandas as pd
import streamlit as st
#======= 추가설치
from sqlalchemy import create_engine

# AWS RDS 연결 정보 변수에 저장
DB_USER = "admin"
DB_PASS = "todtjdai6"
DB_HOST = "deepdive-thegagye.cvsoa0ysideo.ap-northeast-2.rds.amazonaws.com"
DB_PORT = "3306"
DB_NAME = "THE-GAGYE"  

# DB 연결 엔진
def get_db_engine():
    db_url = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(db_url)

def init_state():
    engine = get_db_engine()
    if "spend_df" not in st.session_state:
        try:
            st.session_state.spend_df = pd.read_sql("SELECT * FROM Transactions", engine)
        except:
            st.session_state.spend_df = pd.DataFrame(
                columns=["id", "transaction_date", "merchant_name", "description", "category_name", "amount"]
            )
    
    if "budgets" not in st.session_state:
        st.session_state.budgets = {}

    #  앱 시작 시 AWS RDS에서 기존 데이터 불러오기 
    if "spend_df" not in st.session_state:
        try:
            st.session_state.spend_df = pd.read_sql("SELECT * FROM spend_log", engine)
        except Exception as e:
            st.session_state.spend_df = pd.DataFrame(
                columns=["row_id", "date_time", "merchant", "item", "category", "amount"]
            )

    st.session_state["row_id_seq"] = len(st.session_state.spend_df)

    if "month_input" not in st.session_state:
        st.session_state["month_input"] = datetime.now().strftime("%Y-%m")
#=======

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
