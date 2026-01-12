from datetime import datetime
import pandas as pd
import streamlit as st
# 추가
import os
from dotenv import load_dotenv
from sqlalchemy import text # 추가
from utils_state import init_state, get_db_engine #추가
init_state()

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

# -----------------------
# 예산 초과 알림 로직
# -----------------------
def show_budget_alert():
    engine = get_db_engine()
    if engine is None: return

    now = datetime.now()
    current_month = now.strftime("%Y-%m")
    
    user_budgets = st.session_state.get("budgets", {}).get(current_month, {})
    total_budget = sum(user_budgets.values()) if user_budgets else 0

    if total_budget <= 0:
        return

    try:
        with engine.connect() as conn:
            sd_str = now.strftime("%Y-%m-01 00:00:00")
            
            spend_sql = text("""
                SELECT SUM(amount) FROM Transactions 
                WHERE user_id = 1 
                  AND (type = 'E' OR type IS NULL OR type = '') 
                  AND transaction_date >= :sd
            """)
            
            # 파라미터
            total_spent = conn.execute(spend_sql, {"sd": sd_str}).scalar() or 0

        # [디버그] 
        st.write(f"🔍 [시스템 확인] 이번 달 지출: {total_spent:,.0f}원 / 설정 예산: {total_budget:,.0f}원")

        if total_spent > total_budget:
            over_amount = total_spent - total_budget
            percent = (total_spent / total_budget) * 100
            
            # [추가]큰 제목 ====
            st.divider() 
            st.header("📢 월간 예산 초과 알림") 
            # ==================

            st.error(f"🚨 **{now.month}월 예산 초과 경고**")
            c1, c2 = st.columns([2, 1])
            with c1:
                st.write(f"현재 총 지출: **{total_spent:,.0f}원**")
                st.write(f"설정된 예산: **{total_budget:,.0f}원**")
                st.progress(1.0)
            with c2:
                st.metric("소진율", f"{percent:.1f}%", f"{over_amount:,.0f}원 초과", delta_color="inverse")
            st.divider()
            
    except Exception as e:
        st.sidebar.error(f"알림 시스템 오류: {e}")

show_budget_alert()





