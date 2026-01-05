from datetime import datetime
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from utils_state import init_state
init_state()


load_dotenv(dotenv_path=r"C:\project_ocr\.env", override=True)

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

st.title("THE 가계")
st.caption("영수증/문자 결제내역을 입력하면 자동으로 지출 데이터로 변환하고, 예산 대비 사용 현황을 시각화합니다.")

c1, c2 = st.columns([1.2, 1])
with c1:
    st.markdown(
        """
### 사용 흐름
1) **[1_영수증_입력]** 페이지에서 종이영수증/문자 결제내역 업로드  
2) OCR/LLM 파싱 결과 확인 → **지출 데이터로 적재**  
3) **[2_예산_지출_현황]** 페이지에서 월 예산 설정 + 사용률 확인
"""
    )

with c2:
    st.info(
        "왼쪽 사이드바에서 페이지를 이동하세요.\n\n"
        "- 1_영수증_입력\n"
        "- 2_예산_지출_현황",
        icon="🧾"
    )
