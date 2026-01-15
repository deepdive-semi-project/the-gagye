import streamlit as st
import pandas as pd
from utils_state import get_db_engine



st.title("🗄️ DB 조회 (읽기 전용)")

engine = get_db_engine()

# --- 상단: 연결 정보(민감 정보는 출력 X) ---
cfg = st.secrets["mysql"]
st.caption(f"host: {cfg['host']}  |  port: {cfg.get('port', 3306)}  |  db: {cfg['database']}")

st.divider()

# -------------------------------------------------
# 1) 테이블 목록 불러오기
# -------------------------------------------------
with engine.connect() as conn:
    dbname = conn.exec_driver_sql("SELECT DATABASE()").scalar()
    tables = [t[0] for t in conn.exec_driver_sql("SHOW TABLES").fetchall()]

st.success(f"현재 DB: {dbname}")

# 👉 테이블 선택
table = st.selectbox("📌 조회할 테이블 선택", tables)

st.divider()

# -------------------------------------------------
# 2) 선택한 테이블 미리보기
# -------------------------------------------------
st.subheader(f"📊 `{table}` 테이블 미리보기")

col1, col2 = st.columns([1, 1])
with col1:
    limit = st.number_input("조회 개수", min_value=10, max_value=500, step=10, value=50)
with col2:
    order = st.selectbox("정렬", ["최신순", "오래된순"])

# 🔎 정렬 컬럼 자동 탐색 (date / created_at / id 우선)
def find_order_column(engine, table):
    cols = pd.read_sql(f"DESCRIBE `{table}`", engine)["Field"].tolist()
    for c in ["date", "date_time", "created_at", "updated_at", "id"]:
        if c in cols:
            return c
    return cols[0]  # fallback

order_col = find_order_column(engine, table)
order_sql = f"`{order_col}` {'DESC' if order == '최신순' else 'ASC'}"

if st.button("조회 실행", use_container_width=True):
    try:
        q = f"SELECT * FROM `{table}` ORDER BY {order_sql} LIMIT %s"
        df = pd.read_sql(q, con=engine, params=(int(limit),))
        st.success(f"조회 성공: {len(df)}건 (정렬 기준: {order_col})")
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(e)

st.divider()

# -------------------------------------------------
# 3) 테이블 구조 보기
# -------------------------------------------------
with st.expander(f"🧱 `{table}` 테이블 구조 (DESCRIBE)"):
    try:
        df_desc = pd.read_sql(f"DESCRIBE `{table}`", con=engine)
        st.dataframe(df_desc, use_container_width=True)
    except Exception as e:
        st.error(e)
