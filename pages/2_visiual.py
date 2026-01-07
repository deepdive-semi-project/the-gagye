from datetime import datetime
import pandas as pd
import streamlit as st
<<<<<<< HEAD
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
# [수정] 집계 로직 누적 정상화
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



=======
import matplotlib.pyplot as plt  # (원본 코드에 있었던 import 유지)

from sqlalchemy import text  # SQL 실행용(원본 코드에 있었던 import 유지)

# ✅ utils_state에서 초기화 + DB 엔진 가져오기
from utils_state import init_state, get_db_engine

init_state()

st.title("📊 예산 & 지출 현황")

# ✅ 새 카테고리 체계
CATEGORY_SCHEMA = {
    "식비": ["외식", "배달", "카페", "편의점", "간식"],
    "장보기": ["마트", "식재료", "생필품"],
    "교통/차량": ["택시", "주유", "주차", "대중교통"],
    "쇼핑/취미": ["의류", "도서", "운동", "온라인 쇼핑", "문구"],
    "생활/주거": ["관리비", "통신비", "구독료", "약국"],
    "교육": ["교육", "자기계발"],
    "의료비": ["병원", "약국"],
    "기타": ["경조사", "분류 미정 항목"],
}
DEFAULT_CATEGORIES = list(CATEGORY_SCHEMA.keys())

# -----------------------
# ✅ SessionState 안전 초기화 (예산 관련만 유지)
# -----------------------
if "budgets" not in st.session_state:
    st.session_state.budgets = {}          # { "YYYY-MM": {category: budget} }
if "total_budgets" not in st.session_state:
    st.session_state.total_budgets = {}    # { "YYYY-MM": total_budget }

# (선택) fallback 용도로만 유지 - 화면 표시 기준은 Transactions로 바꿈
if "spend_df" not in st.session_state:
    st.session_state.spend_df = pd.DataFrame(columns=["row_id", "date_time", "merchant", "item", "category", "amount"])

# -----------------------
# ✅ 날짜 -> YYYY-MM 정규화
# -----------------------
def normalize_month(dt) -> str:
    """date_time 문자열/값 -> YYYY-MM. 실패 시 Unknown"""
    if dt is None:
        return "Unknown"

    # DB에서 datetime 객체로 넘어오는 케이스
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m")

    s = str(dt).strip()
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y.%m.%d %H:%M:%S",
        "%Y.%m.%d %H:%M",
        "%Y.%m.%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
    ):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m")
        except Exception:
            pass
    return "Unknown"


# -----------------------
# ✅ [핵심 변경] Transactions 테이블을 “항상” 화면 기준 데이터로 사용
# -----------------------
engine = get_db_engine()

df_tx = pd.DataFrame(columns=["row_id", "date_time", "merchant", "item", "category", "amount"])

if engine is None:
    st.warning("DB 엔진을 만들 수 없습니다. (.env / secrets 설정을 확인하세요) 현재는 로컬 session_state 데이터로만 동작합니다.")
    # fallback (원하면 이 줄도 지우면 됨)
    df_tx = st.session_state.spend_df.copy()
else:
    try:
        # Transactions -> 화면 표준 컬럼명으로 매핑
        df_db = pd.read_sql("SELECT * FROM Transactions", engine)

        df_db = df_db.rename(columns={
            "transaction_date": "date_time",
            "description": "item",
            "merchant_name": "merchant",
            "category_name": "category",
        })

        # row_id는 화면용(고유키로 id가 있으면 그걸 써도 됨)
        if "row_id" not in df_db.columns:
            if "id" in df_db.columns:
                df_db = df_db.rename(columns={"id": "row_id"})
            else:
                df_db.insert(0, "row_id", range(1, len(df_db) + 1))

        # amount 숫자 보정
        if "amount" in df_db.columns:
            df_db["amount"] = pd.to_numeric(df_db["amount"], errors="coerce").fillna(0.0)
        else:
            df_db["amount"] = 0.0

        # category 보정
        if "category" in df_db.columns:
            df_db["category"] = df_db["category"].fillna("기타").astype(str).str.strip()
            df_db.loc[~df_db["category"].isin(DEFAULT_CATEGORIES), "category"] = "기타"
        else:
            df_db["category"] = "기타"

        # 필수 컬럼 확보
        for c in ["date_time", "merchant", "item"]:
            if c not in df_db.columns:
                df_db[c] = None

        df_tx = df_db[["row_id", "date_time", "merchant", "item", "category", "amount"]].copy()

    except Exception as e:
        st.error(f"DB 데이터를 불러오는 중 오류 발생: {e}")
        # fallback (원하면 이 줄도 지우면 됨)
        df_tx = st.session_state.spend_df.copy()

# -----------------------
# ✅ 이제부터는 spend_df가 아니라 df_tx(Transactions)가 기준
# -----------------------
st.caption(f"Transactions rows: {len(df_tx)}")
st.dataframe(df_tx.tail(20), width="stretch")

df_all = df_tx.copy()
if df_all.empty:
    df_all = df_all.assign(month=pd.Series(dtype="object"))
else:
    df_all["month"] = df_all["date_time"].apply(normalize_month)

# -----------------------
# ✅ 콜백: 미배분을 '기타'에 배분
# -----------------------
def alloc_to_misc(month: str, diff: int):
    """미배분(diff)을 '기타' 예산에 더한다. (on_click 콜백에서만 호출)"""
    if diff <= 0:
        return

    st.session_state.budgets.setdefault(month, {c: 0 for c in DEFAULT_CATEGORIES})

    misc_key = f"budget_{month}_기타"
    cur_misc = int(st.session_state.get(misc_key, st.session_state.budgets[month].get("기타", 0)))
    new_misc = cur_misc + int(diff)

    st.session_state[misc_key] = new_misc
    st.session_state.budgets[month]["기타"] = new_misc


# -----------------------
# ✅ 월 선택
# -----------------------
if st.session_state.get("pending_month"):
    st.session_state["month_input"] = st.session_state["pending_month"]
    st.session_state["pending_month"] = None

available_months = sorted([m for m in df_all["month"].unique().tolist() if m != "Unknown"])
default_month = st.session_state.get("month_input", datetime.now().strftime("%Y-%m"))

with st.sidebar:
    st.header("월 선택")

    if available_months:
        month = st.selectbox(
            "대상 월 (YYYY-MM)",
            options=available_months,
            index=available_months.index(default_month) if default_month in available_months else len(available_months) - 1,
            key="month_selectbox",
        )
    else:
        month = st.text_input("대상 월 (YYYY-MM)", value=default_month, key="month_text")

    st.session_state["month_input"] = month

    # ✅ 월 키 생성 (KeyError 방지)
    st.session_state.budgets.setdefault(month, {c: 0 for c in DEFAULT_CATEGORIES})
    st.session_state.total_budgets.setdefault(month, 0)

    # -----------------------
    # ✅ 총 예산(월) 설정
    # -----------------------
    st.divider()
    st.header("총 예산(월)")

    total_key = f"total_budget_{month}"
    if total_key not in st.session_state:
        st.session_state[total_key] = int(st.session_state.total_budgets.get(month, 0))

    total_budget = st.number_input(
        "총 예산",
        min_value=0,
        step=10000,
        key=total_key,
    )
    st.session_state.total_budgets[month] = int(total_budget)

    # -----------------------
    # ✅ 카테고리 예산 설정
    # -----------------------
    st.divider()
    st.header("예산 설정")

    for c in DEFAULT_CATEGORIES:
        k = f"budget_{month}_{c}"
        if k not in st.session_state:
            st.session_state[k] = int(st.session_state.budgets[month].get(c, 0))

        v = st.number_input(
            c,
            min_value=0,
            step=1000,
            key=k,
        )
        st.session_state.budgets[month][c] = int(v)

    # -----------------------
    # ✅ 합계/차이 표시 + 자동 배분 버튼
    # -----------------------
    st.divider()
    cat_budget_sum = int(sum(st.session_state.budgets[month].values()))
    diff = int(total_budget) - int(cat_budget_sum)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("카테고리 예산 합계", f"{cat_budget_sum:,.0f}원")
    with c2:
        if diff >= 0:
            st.metric("미배분", f"{diff:,.0f}원")
        else:
            st.metric("초과 배분", f"{abs(diff):,.0f}원")

    if diff > 0:
        st.button(
            "➕ 미배분을 '기타'에 배분",
            use_container_width=True,
            key=f"alloc_misc_{month}",
            on_click=alloc_to_misc,
            kwargs={"month": month, "diff": int(diff)},
        )

# -----------------------
# ✅ 데이터 필터 (Transactions 기반)
# -----------------------
df = df_all[df_all["month"] == month].copy()

if not df.empty:
    df["category"] = df["category"].fillna("기타").astype(str).str.strip()
    df.loc[~df["category"].isin(DEFAULT_CATEGORIES), "category"] = "기타"

st.subheader(f"지출 내역 ({month})  — Transactions 기준")
st.dataframe(df, width="stretch")

# -----------------------
# ✅ 집계 (Transactions 기반)
# -----------------------
if df.empty:
    spend_sum = pd.Series(0.0, index=DEFAULT_CATEGORIES)
else:
    spend_sum = df.groupby("category")["amount"].sum().reindex(DEFAULT_CATEGORIES, fill_value=0.0)

budget_map = st.session_state.budgets.get(month, {c: 0 for c in DEFAULT_CATEGORIES})
budget_series = pd.Series({c: float(budget_map.get(c, 0)) for c in DEFAULT_CATEGORIES})

summary = pd.DataFrame({
    "category": DEFAULT_CATEGORIES,
    "amount": spend_sum.values,
    "budget": budget_series.values,
})
summary["remain"] = summary["budget"] - summary["amount"]

st.subheader("카테고리별 요약")
st.dataframe(summary, width="stretch")

# -----------------------
#  게이지바 기본 형태
# -----------------------
st.subheader("📈 예산 사용 게이지")

for _, r in summary.iterrows():
    cat = r["category"]
    spent = float(r["amount"])
    budget = float(r["budget"])

    if budget <= 0:
        st.write(f"**{cat}** — 예산이 0원이라 게이지를 표시할 수 없어요.")
        continue

    ratio = spent / budget
    pct = int(min(max(ratio, 0), 1) * 100)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(f"**{cat}**  ({spent:,.0f} / {budget:,.0f}원, {ratio*100:,.1f}%)")
        st.progress(pct)
    with col2:
        if ratio >= 1:
            st.error(f"초과 {spent - budget:,.0f}원")
        elif ratio >= 0.8:
            st.warning(f"주의 {budget - spent:,.0f}원 남음")
        else:
            st.success(f"잔액 {budget - spent:,.0f}원")
>>>>>>> origin/receipt-ocr
