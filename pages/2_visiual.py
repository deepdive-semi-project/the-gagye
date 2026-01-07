from datetime import datetime
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sqlalchemy import text  # [추가] SQL 명령 실행을 위해 필요

# [수정] get_db_engine 추가 임포트
from utils_state import init_state, get_db_engine 
init_state()

# -----------------------
# [추가] DB 데이터 로드 및 컬럼명 바꾸기
# -----------------------
engine = get_db_engine()

try:
    df_db = pd.read_sql("SELECT * FROM Transactions", engine)
    
    df_db = df_db.rename(columns={
        'transaction_date': 'date_time',
        'description': 'item',
        'merchant_name': 'merchant',
        'category_name': 'category'
    })
    st.session_state.spend_df = df_db
except Exception as e:
    st.error(f"DB 데이터를 불러오는 중 오류 발생: {e}")

st.write("spend_df rows:", len(st.session_state.spend_df))
st.dataframe(st.session_state.spend_df.tail(20), width="stretch")

# 카테고리명 재 분류 완료
DEFAULT_CATEGORIES = [
    "식비", "장보기", "교통/차량", "쇼핑/취미", 
    "생활/주거", "교육", "의료비", "기타"
]

def normalize_month(dt):
    try:
        # [수정] DB에서 날짜 객체(Datetime)로 넘어올 경우를 위해 처리 추가
        if isinstance(dt, datetime):
            return dt.strftime("%Y-%m")
        return datetime.strptime(str(dt), "%Y-%m-%d %H:%M").strftime("%Y-%m")
    except:
        return "Unknown"

st.title("📊 예산 & 지출 현황")

# -----------------------
# 월 선택
# -----------------------
month = st.sidebar.text_input(
    "대상 월 (YYYY-MM)",
    value=st.session_state.get("month_input", datetime.now().strftime("%Y-%m"))
)
st.session_state["month_input"] = month

# -----------------------
# 예산 설정
# -----------------------
# [수정] 예산 설정 부분 - 카테고리 바뀌었으니까 그거에 맞게 수정
st.sidebar.header("월간 예산 설정")
if month not in st.session_state.budgets:
    st.session_state.budgets[month] = {c: 0 for c in DEFAULT_CATEGORIES}

for c in DEFAULT_CATEGORIES:
    st.session_state.budgets[month][c] = st.sidebar.number_input(
        f"{c}",
        min_value=0,
        step=1000,
        value=st.session_state.budgets[month].get(c, 0)
    )

# -----------------------
# 데이터 필터
# -----------------------
df = st.session_state.spend_df.copy()

# [수정] 데이터가 비어있지 않을 때만 실행하도록 조건 추가
if not df.empty:
    df["month"] = df["date_time"].apply(normalize_month)
    df = df[df["month"] == month]

st.subheader(f"지출 내역 ({month})")
st.dataframe(df, width="stretch")

# -----------------------
# [수정] 집계 로직 정상화
# -----------------------
if not df.empty:
    # 1. [추가] 금액 데이터 타입 강제 형변환 (에러 방지)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)

    # 2. [수정] 카테고리별 그룹화 및 합계 계산
    summary = df.groupby("category")["amount"].sum().reset_index()

    # 3. [추가] 예산 데이터 매핑 및 계산
    summary["budget"] = summary["category"].map(st.session_state.budgets.get(month, {}))
    summary["budget"] = summary["budget"].fillna(0) # 예산 설정 안 된 경우 0 처리
    summary["remain"] = summary["budget"] - summary["amount"]

    # 4. [수정] 가독성을 위해 숫자에 천 단위 콤마 추가하여 출력
    st.subheader(f"📈 {month} 카테고리별 요약 현황")
    
    # 표시용 데이터프레임 가공
    display_summary = summary.copy()
    for col in ["amount", "budget", "remain"]:
        display_summary[col] = display_summary[col].apply(lambda x: f"{int(x):,}원")
    
    st.table(display_summary)
else:
    st.info(f"📅 {month}월에 적재된 데이터가 없습니다.")



