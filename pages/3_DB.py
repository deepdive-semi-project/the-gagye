import streamlit as st
import pandas as pd
from db import get_engine

st.title("📂 DB 조회 (읽기 전용)")

engine = get_engine()

# --- 상단: 연결 정보(민감 정보는 출력 X) ---
cfg = st.secrets["mysql"]
st.caption(f"host: {cfg['host']}  |  port: {cfg.get('port', 3306)}  |  db: {cfg['database']}")

st.divider()

# 1) 테이블 목록 보기
with st.expander("🔍 테이블 목록 보기", expanded=True):
    try:
        with engine.connect() as conn:
            dbname = conn.exec_driver_sql("SELECT DATABASE()").scalar()
            tables = conn.exec_driver_sql("SHOW TABLES").fetchall()
        st.success(f"현재 DB: {dbname}")
        st.write([t[0] for t in tables])
    except Exception as e:
        st.error(e)

st.divider()

# 2) expenses 미리보기
st.subheader("📊 expenses 테이블 미리보기")

col1, col2 = st.columns([1, 1])
with col1:
    limit = st.number_input("조회 개수", min_value=10, max_value=500, step=10, value=50)
with col2:
    order = st.selectbox("정렬", ["최신순(date DESC)", "오래된순(date ASC)"])

order_sql = "date DESC" if "DESC" in order else "date ASC"

if st.button("조회 실행", use_container_width=True):
    try:
        q = f"SELECT * FROM expenses ORDER BY {order_sql} LIMIT %s"
        df = pd.read_sql(q, con=engine, params=(int(limit),))
        st.success(f"조회 성공: {len(df)}건")
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(e)

st.divider()

# 3) 테이블 구조 보기
with st.expander("🧱 expenses 테이블 구조(DESCRIBE)"):
    try:
        df_desc = pd.read_sql("DESCRIBE expenses", con=engine)
        st.dataframe(df_desc, use_container_width=True)
    except Exception as e:
        st.error(e)
