import streamlit as st
from datetime import datetime
import pandas as pd
from sqlalchemy import text # 추가

from utils_state import init_state, get_db_engine #추가
init_state()

st.title("THE 가계")
st.caption("영수증/문자 결제내역을 입력하면 자동으로 지출 데이터로 변환하고, 예산 대비 사용 현황을 시각화합니다.")

c1, c2 = st.columns([1.2, 1])
with c1:
    st.markdown("""
### 🔄 사용 흐름
1) **[🧾 영수증 입력]** 페이지에서 종이영수증 또는 문자 결제내역 업로드  
2) OCR / LLM 파싱 결과 확인 후 → **지출 데이터로 DB 적재**  
3) **[📊 지출 현황]** 페이지에서 월별 예산 설정 및 카테고리별 사용률 확인  
4) **[🔮 지출 예측]** 페이지에서 과거 소비 패턴 기반 다음 달 지출 예측 확인  
""")


with c2:
    st.info("왼쪽 메뉴에서 기능을 선택하세요.", icon="🏠")
    st.markdown("""
    - 🧾 **영수증 입력**: OCR + 파싱 후 DB 저장  
    - 📊 **지출 현황**: 월별/카테고리별 시각화  
    - 🔮 **지출 예측**: 다음 달 소비 예측
    """)


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
        # st.write(f"🔍 [시스템 확인] 이번 달 지출: {total_spent:,.0f}원 / 설정 예산: {total_budget:,.0f}원")

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