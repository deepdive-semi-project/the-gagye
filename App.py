from datetime import datetime
import pandas as pd
import streamlit as st
# 추가
import os
from dotenv import load_dotenv
from sqlalchemy import text # 추가
from utils_state import init_state, get_db_engine #추가
init_state()

# 사이드바 텍스트 변경
import streamlit as st

st.set_page_config(
    page_title="THE 가계",
    page_icon="💳",
    layout="wide",
)

pg = st.navigation([
    st.Page("pages/0_home.py", title="홈", icon="🏠"),
    st.Page("pages/1_main.py", title="영수증 입력", icon="🧾"),
    st.Page("pages/2_visualization.py", title="지출 현황", icon="📊"),
    st.Page("pages/3_DB.py", title="DB 관리", icon="🗄️"),
    st.Page("pages/4_prediction.py", title="지출 예측", icon="🔮"),
])
pg.run()



# 경로 수정====
current_dir = os.path.dirname(os.path.abspath(__file__)) # 현재 파일이 있는 폴더 경로
env_path = os.path.join(current_dir, ".env") # 현재 폴더의 .env 파일 지정
load_dotenv(dotenv_path=env_path, override=True)
# =====
st.set_page_config(page_title="THE 가계", layout="wide")

# ---- global session init (모든 페이지 공유) ----
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




