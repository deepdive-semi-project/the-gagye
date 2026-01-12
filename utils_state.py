from datetime import datetime
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy import text


load_dotenv() 

USER_ID = 2  # 임시 사용자 ID

def get_db_engine():
    try:
        cfg = st.secrets["mysql"]
    except Exception:
        return None

    db_url = (
        f"mysql+pymysql://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    return create_engine(db_url)



def init_state():
    # 1) budgets 기본값
    if "budgets" not in st.session_state:
        st.session_state.budgets = {}
        engine = get_db_engine()

        if engine is not None:
            try:
                with engine.connect() as conn:
                    query = 'SELECT year, month, category_name, amount ' \
                    'FROM Budget b JOIN Category c on b.category_id = c.id ' \
                    'WHERE b.user_id = :user_id ' \
                    'ORDER BY year, month, category_name;'
                    result = conn.execute(text(query), {"user_id": USER_ID})

                for row in result:
                    year, month, category_name, amount = row
                    st.session_state.budgets.setdefault(f"{year}-{month:02d}", {})[category_name] = amount

            except Exception as e:
                print("init_state budgets load error:", e)

    # 2) month_input 기본값
    if "month_input" not in st.session_state:
        st.session_state["month_input"] = datetime.now().strftime("%Y-%m")

    # 3) 기타 상태값
    if "pending_month" not in st.session_state:
        st.session_state["pending_month"] = None
    if "last_json_upload_key" not in st.session_state:
        st.session_state["last_json_upload_key"] = None

    # 4) spend_df: DB에서 로드 시도, 실패하면 빈 DF
    if "spend_df" not in st.session_state:
        engine = get_db_engine()
        table = os.getenv("DB_TABLE", "Transactions")

        if engine is not None:
            try:
                st.session_state.spend_df = pd.read_sql(f"SELECT * FROM {table}", engine)
            except Exception:
                st.session_state.spend_df = pd.DataFrame(
                    columns=["id", "transaction_date", "merchant_name", "description", "category_name", "amount"]
                )
        else:
            # DB 설정이 없으면 로컬 모드
            st.session_state.spend_df = pd.DataFrame(
                columns=["id", "transaction_date", "merchant_name", "description", "category_name", "amount"]
            )

    # 5) row_id_seq 갱신
    st.session_state["row_id_seq"] = len(st.session_state.spend_df)
