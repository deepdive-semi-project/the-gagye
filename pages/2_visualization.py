from datetime import datetime
import pandas as pd
import numpy as np
import re
import streamlit as st
import matplotlib.pyplot as plt 
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter

from prophet import Prophet

from sqlalchemy import text  

plt.rcParams["font.family"] = "Malgun Gothic"  # Windows
plt.rcParams["axes.unicode_minus"] = False

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
# ✅ [핵심 수정] Transactions + Category JOIN으로 category_name 가져오기
# -----------------------
engine = get_db_engine()

def map_to_main_category(cat_raw: str) -> str:
    """DB에서 가져온 카테고리(소분류/대분류 혼재)를 대분류로 정규화"""
    if cat_raw is None:
        return "기타"
    cat_raw = str(cat_raw).strip()
    if not cat_raw:
        return "기타"

    # 1) 이미 대분류라면 그대로
    if cat_raw in CATEGORY_SCHEMA.keys():
        return cat_raw

    # 2) 소분류면 소속 대분류 찾기
    for main, subs in CATEGORY_SCHEMA.items():
        if cat_raw in subs:
            return main

    # 3) 어디에도 없으면 기타
    return "기타"


df_tx = pd.DataFrame(columns=["row_id", "date_time", "merchant", "item", "category", "amount"])

if engine is None:
    st.warning("DB 엔진을 만들 수 없습니다. (.env / secrets 설정을 확인하세요) 현재는 로컬 session_state 데이터로만 동작합니다.")
    df_tx = st.session_state.spend_df.copy()

else:
    try:
        sql = """
        SELECT
            t.id AS row_id,
            t.transaction_date AS date_time,
            t.merchant_name AS merchant,
            t.description AS item,
            COALESCE(c.category_name, '기타') AS category_raw,
            t.amount
        FROM Transactions t
        LEFT JOIN Category c
            ON t.category_id = c.id
        -- 필요하면 여기서 지출만 필터링 가능
        -- WHERE t.type = 'E'
        """
        df_db = pd.read_sql(sql, engine)

        df_db["amount"] = pd.to_numeric(df_db["amount"], errors="coerce").fillna(0.0)

        df_db["category"] = df_db["category_raw"].apply(map_to_main_category)

        for c in ["date_time", "merchant", "item"]:
            if c not in df_db.columns:
                df_db[c] = None

        df_tx = df_db[["row_id", "date_time", "merchant", "item", "category", "amount"]].copy()

    except Exception as e:
        st.error(f"가계부 데이터를 불러오는 중 오류 발생: {e}")
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

    # 예산 설정시 카테고리명 조회
    category_list = pd.read_sql("SELECT id, category_name FROM Category ORDER BY id", engine)
    category_name_to_id = dict(zip(category_list["category_name"], category_list["id"]))

    # 예산 설정 DB 저장
    def save_budget(key):
        arr_key = key.split("_")  # budget_{month}_{category} 형태
        arr_month = arr_key[1].split("-")
        category_nm = arr_key[2]

        with engine.connect() as conn:
            query = '''
INSERT Budget (user_id, year, month, category_id, amount, description)
VALUES (:user_id, :year, :month, :category_id, :amount, '')
ON DUPLICATE KEY UPDATE amount = :amount '''
        
            params = {"user_id": USER_ID, 
                        "year": int(arr_month[0]), 
                        "month": int(arr_month[1]), 
                        "category_id": int(category_name_to_id.get(category_nm, category_name_to_id.get("기타", 8))), 
                        "amount": int(st.session_state[key])}
            result = conn.execute(text(query), params)
            # print("result", result.rowcount)
            conn.commit()
    
    # 예산 설정 변경 이벤트            
    def budget_changed(key):
        try:
            save_budget(key)
        except Exception as e:
            print("Error:", e)

    for c in DEFAULT_CATEGORIES:
        k = f"budget_{month}_{c}"
        if k not in st.session_state:
            st.session_state[k] = int(st.session_state.budgets[month].get(c, 0))

        v = st.number_input(
            c,
            min_value=0,
            step=1000,
            key=k,
            on_change=budget_changed,
            args=[k]
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

st.subheader(f"지출 내역 ({month})")
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


# -----------------------
#  월별 카테고리 내역 시각화 (Category JOIN: category_name)
# -----------------------
import pandas as pd
import streamlit as st

st.markdown("### 카테고리별 지출 (금액/비율)")

if engine is None:
    st.error("DB 엔진(engine)이 None 입니다.")
    st.stop()

# ✅ E(지출)만 + 필요하면 user_id까지 필터
USER_ID = 2  

query = """
SELECT
    t.transaction_date,
    t.type,
    t.user_id,
    COALESCE(c.category_name, '기타') AS category,
    t.amount
FROM Transactions t
LEFT JOIN Category c
    ON t.category_id = c.id
WHERE t.type = 'E'
"""
df = pd.read_sql(query, engine, params={"uid": USER_ID})


# 날짜 → YYYY-MM
df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
df["month"] = df["transaction_date"].dt.strftime("%Y-%m")

# 금액/카테고리 정리
df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
df["category"] = df["category"].fillna("기타").astype(str).str.strip()

# 월 선택
available_months = sorted(df["month"].dropna().unique().tolist())
if not available_months:
    st.warning("표시할 지출(E) 데이터가 없습니다.")
    st.stop()

month = st.selectbox("조회할 월", options=available_months, index=len(available_months) - 1)

# 월 필터
df_m = df[df["month"] == month].copy()

# ✅ 지출인데 amount가 음수(환불)도 있을 수 있음 → 포함/제외 옵션
include_refund = st.checkbox("환불/취소(음수 금액) 포함", value=True)
if not include_refund:
    df_m = df_m[df_m["amount"] >= 0]

# 집계
summary = (
    df_m.groupby("category", as_index=False)["amount"]
        .sum()
)

# 0원 제거 + 금액 정렬
summary = summary[summary["amount"] != 0].sort_values("amount", ascending=False)

if summary.empty:
    st.info("선택한 월에 표시할 지출 합계가 없습니다.")
    st.stop()

total = float(summary["amount"].sum())
summary["ratio_pct"] = (summary["amount"] / total * 100).round(1)

# KPI + 표
c1, c2 = st.columns([1, 2])
with c1:
    st.metric("총 지출(E)", f"{total:,.0f}원")
with c2:
    st.caption("금액 기준 내림차순 정렬 (비율은 해당 월 총지출 대비)")

st.dataframe(
    summary.rename(columns={"category": "카테고리", "amount": "지출(원)", "ratio_pct": "비율(%)"}),
    width="stretch"
)

# 차트
st.bar_chart(
    summary.set_index("category")[["amount"]].rename(columns={"amount": "지출(원)"}),
    height=420
)


# =========================================================
# 🧩 DB 직접 관리 패널 (Transactions CRUD: 추가/수정/삭제)
#   - Category(category_name) ↔ Transactions.category_id 매핑 포함
# =========================================================
st.markdown("---")
st.subheader("🛠️ 가계부 지출 데이터 직접 관리 (추가/수정/삭제)")

if engine is None:
    st.warning("DB 엔진이 없어 CRUD 기능을 사용할 수 없습니다.")
    st.stop()

# ---- Category 매핑 로드 (id <-> name)
cat_df = pd.read_sql("SELECT id, category_name FROM Category ORDER BY id", engine)
cat_id_to_name = dict(zip(cat_df["id"], cat_df["category_name"]))
cat_name_to_id = dict(zip(cat_df["category_name"], cat_df["id"]))
cat_names = cat_df["category_name"].tolist()

order_oldest_first = st.checkbox(
    "오래된 순(처음 데이터부터) 보기",
    value=False,
    help="체크하면 가장 오래된 거래(id가 작은 것)부터 표시합니다."
)

order_sql = "t.id ASC" if order_oldest_first else "t.transaction_date DESC, t.id DESC"

crud_query = f"""
SELECT
    t.id,
    t.user_id,
    t.transaction_date,
    t.merchant_name,
    t.description,
    t.amount,
    t.type,
    t.category_id
FROM Transactions t
ORDER BY {order_sql}
"""

crud_df = pd.read_sql(crud_query, engine, params={"uid": USER_ID})

# category_id -> category_name 표시용 컬럼
crud_df["category_name"] = crud_df["category_id"].map(cat_id_to_name).fillna("기타")
crud_df["transaction_date"] = pd.to_datetime(crud_df["transaction_date"], errors="coerce")
crud_df["amount"] = pd.to_numeric(crud_df["amount"], errors="coerce").fillna(0.0)

tab_add, tab_edit, tab_delete = st.tabs(["➕ 추가", "✏️ 수정", "🗑️ 삭제"])


# -----------------------------
# 1) 추가 (INSERT)
# -----------------------------
with tab_add:
    st.caption("가계부에 지출 1건을 직접 추가합니다.")

    with st.form("tx_add_form", clear_on_submit=False):
        c1, c2 = st.columns([1, 1])
        with c1:
            dt = st.text_input("일시 (예: 2026-01-05 09:30)", value=datetime.now().strftime("%Y-%m-%d %H:%M"))
        with c2:
            cat_name = st.selectbox("카테고리", options=cat_names, index=cat_names.index("기타") if "기타" in cat_names else 0)

        c3, c4 = st.columns([1, 1])
        with c3:
            merchant = st.text_input("상호명(merchant_name)", value="")
        with c4:
            amount = st.number_input("금액(amount)", min_value=-10_000_000.0, max_value=10_000_000.0, step=100.0, value=0.0)

        desc = st.text_input("설명(description)", value="")

        submitted = st.form_submit_button("✅ 가계부에 추가", use_container_width=True)

    if submitted:
        # date 파싱
        dt_parsed = pd.to_datetime(dt, errors="coerce")
        if pd.isna(dt_parsed):
            st.error("일시 형식이 올바르지 않습니다. 예: 2026-01-05 09:30")
        else:
            cat_id = int(cat_name_to_id.get(cat_name, cat_name_to_id.get("기타", 8)))

            insert_sql = text("""
                INSERT INTO Transactions
                (user_id, category_id, transaction_date, merchant_name, amount, type, description)
                VALUES
                (:user_id, :category_id, :transaction_date, :merchant_name, :amount, 'E', :description)
            """)
            try:
                with engine.begin() as conn:
                    conn.execute(insert_sql, {
                        "user_id": int(USER_ID),
                        "category_id": cat_id,
                        "transaction_date": dt_parsed.to_pydatetime(),
                        "merchant_name": merchant,
                        "amount": float(amount),
                        "description": desc
                    })
                st.success("추가 완료! (가계부에 저장됨)")
                st.rerun()
            except Exception as e:
                st.error(f"추가 실패: {e}")


# -----------------------------
# 2) 수정 (UPDATE) - data_editor
# -----------------------------
DT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}$")  # YYYY-MM-DD HH:MM

with tab_edit:
    st.caption("표에서 수정한 뒤 ‘저장’ 버튼을 누르면 DB에 반영됩니다. (날짜는 YYYY-MM-DD HH:MM 형식)")

    # ✅ 화면 편집용 df (표시 컬럼 정리)
    edit_view = crud_df[[
    "id", "user_id", "transaction_date", "merchant_name", "description",
    "category_name", "amount"
    ]].copy()

    # ✅ data_editor에서 TextColumn으로 편집하려면 dtype을 str로 바꿔야 함
    edit_view["transaction_date"] = pd.to_datetime(
        edit_view["transaction_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d %H:%M")
    edit_view["transaction_date"] = edit_view["transaction_date"].fillna("")

    # ✅ 안전하게 타입 정리
    edit_view["merchant_name"] = edit_view["merchant_name"].fillna("").astype(str)
    edit_view["description"] = edit_view["description"].fillna("").astype(str)
    edit_view["category_name"] = edit_view["category_name"].fillna("기타").astype(str)
    edit_view["amount"] = pd.to_numeric(edit_view["amount"], errors="coerce").fillna(0.0)

    edited = st.data_editor(
    edit_view,
    key="tx_editor",
    use_container_width=True,
    num_rows="fixed",
    column_config={
        "id": st.column_config.NumberColumn("id", disabled=True),
        "user_id": st.column_config.NumberColumn("user_id", disabled=True),
        "transaction_date": st.column_config.TextColumn(
            "transaction_date (YYYY-MM-DD HH:MM)",
            help="예: 2026-01-05 09:30"
        ),
        "merchant_name": st.column_config.TextColumn("merchant_name"),
        "description": st.column_config.TextColumn("description"),
        "amount": st.column_config.NumberColumn("amount", step=100.0),
        "category_name": st.column_config.SelectboxColumn("category", options=cat_names),
    },
)

    c1, c2 = st.columns([1, 1])

    with c1:
        a = st.session_state.get("tx_save_alert")
        if a:
            (st.success if a["level"]=="success" else st.warning)(a["msg"])
            if st.button("알림 닫기", key="tx_save_alert_close"):
                del st.session_state["tx_save_alert"]
                st.rerun()

        if st.button("💾 수정 저장(가계부 반영)", use_container_width=True):
            try:
                before = edit_view.set_index("id")
                after = edited.set_index("id")

                def norm(x):
                    return "" if pd.isna(x) else str(x).strip()

                changed_ids = []
                for rid in after.index:
                    if rid not in before.index:
                        continue
                    b = before.loc[rid]
                    a = after.loc[rid]

                    if (
                        norm(b.get("transaction_date")) != norm(a.get("transaction_date")) or
                        norm(b.get("merchant_name")) != norm(a.get("merchant_name")) or
                        norm(b.get("description")) != norm(a.get("description")) or
                        norm(b.get("category_name")) != norm(a.get("category_name")) or
                        float(b.get("amount", 0.0)) != float(a.get("amount", 0.0))
                    ):
                        changed_ids.append(int(rid))

                if not changed_ids:
                    st.toast("변경된 내용이 없습니다.", icon="ℹ️")
                    st.info("변경된 내용이 없습니다.")
                else:
                    update_sql = text("""
                        UPDATE Transactions
                        SET transaction_date = :transaction_date,
                            merchant_name = :merchant_name,
                            description = :description,
                            amount = :amount,
                            category_id = :category_id
                        WHERE id = :id
                        AND user_id = :user_id
                        AND type = 'E'
                    """)

                    bad_rows = []
                    zero_updates = []
                    success_count = 0  # ✅ 실제 DB 반영 건수

                    with engine.begin() as conn:
                        for rid in changed_ids:
                            row = after.loc[rid]

                            # 날짜 검증
                            sdt = str(row.get("transaction_date", "")).strip()
                            if not DT_RE.match(sdt):
                                bad_rows.append((rid, sdt))
                                continue

                            dt_val = pd.to_datetime(sdt, errors="coerce")
                            if pd.isna(dt_val):
                                bad_rows.append((rid, sdt))
                                continue

                            # user_id 안전 변환
                            uid_val = pd.to_numeric(row.get("user_id"), errors="coerce")
                            if pd.isna(uid_val):
                                zero_updates.append((rid, "user_id가 비어있음/변환불가"))
                                continue
                            uid_val = int(uid_val)

                            cat_name = str(row.get("category_name", "기타")).strip()
                            cat_id = int(cat_name_to_id.get(cat_name, cat_name_to_id.get("기타", 8)))

                            res = conn.execute(update_sql, {
                                "id": int(rid),
                                "user_id": uid_val,
                                "transaction_date": dt_val.to_pydatetime(),
                                "merchant_name": str(row.get("merchant_name", "")).strip(),
                                "description": str(row.get("description", "")).strip(),
                                "amount": float(pd.to_numeric(row.get("amount", 0.0), errors="coerce") or 0.0),
                                "category_id": cat_id,
                            })

                            if res.rowcount > 0:
                                success_count += res.rowcount
                            else:
                                zero_updates.append((rid, f"업데이트 0건 (user_id={uid_val}, type='E' 조건 확인)"))

                    # 상세 오류 표시
                    if bad_rows:
                        st.error("일부 행은 날짜 형식 오류로 저장되지 않았습니다. (YYYY-MM-DD HH:MM)")
                        st.dataframe(
                            pd.DataFrame(bad_rows, columns=["id", "입력된 transaction_date"]),
                            use_container_width=True
                        )

                    if zero_updates:
                        st.warning("일부 행은 조건 불일치로 가계부에 반영되지 않았습니다.")
                        st.dataframe(
                            pd.DataFrame(zero_updates, columns=["id", "사유"]),
                            use_container_width=True
                        )

                    if success_count > 0:
                        st.session_state["tx_save_alert"] = {
                            "level": "success",
                            "msg": f"✅ 수정 완료! ({success_count}건 가계부 반영)"
                        }
                    else:
                        st.session_state["tx_save_alert"] = {
                            "level": "warning",
                            "msg": "⚠️ 가계부에 반영된 항목이 없습니다. (user_id/type/날짜 형식 확인)"
                        }

                    st.cache_data.clear()
                    st.rerun()

            except Exception as e:
                st.error(f"수정 저장 실패: {e}")


# -----------------------------
# 3) 삭제 (DELETE) - 체크 선택
# -----------------------------
with tab_delete:
    st.caption("삭제는 되돌리기 어렵습니다. 신중히 진행하세요.")

    with engine.begin() as conn:
        crud_df = pd.read_sql(
            text("""
                SELECT
                    t.id,
                    t.user_id,
                    t.transaction_date,
                    t.merchant_name,
                    t.description,
                    c.category_name,
                    t.amount
                FROM Transactions t
                LEFT JOIN Category c ON t.category_id = c.id
                WHERE t.type = 'E'
                ORDER BY t.id DESC
            """),
            conn
        )

    del_view = crud_df[[
        "id", "user_id", "transaction_date", "merchant_name", "description", "category_name", "amount"
    ]].copy()
    del_view.insert(0, "delete?", False)

    picked = st.data_editor(
        del_view,
        key="tx_delete_picker_all",  
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "delete?": st.column_config.CheckboxColumn("delete?"),
            "id": st.column_config.NumberColumn("id", disabled=True),
            "user_id": st.column_config.NumberColumn("user_id", disabled=True),
            "transaction_date": st.column_config.DatetimeColumn("transaction_date", disabled=True),
            "amount": st.column_config.NumberColumn("amount", disabled=True),
            "category_name": st.column_config.TextColumn("category", disabled=True),
        },
    )

    ids_to_delete = picked.loc[picked["delete?"] == True, "id"].astype(int).tolist()

    if st.button(f"🗑️ 선택 {len(ids_to_delete)}건 삭제(가계부)",
                 use_container_width=True,
                 disabled=(len(ids_to_delete) == 0)):
        try:
            delete_sql = text("""
                DELETE FROM Transactions
                WHERE id = :id
                  AND type = 'E'
            """)
            with engine.begin() as conn:
                for rid in ids_to_delete:
                    conn.execute(delete_sql, {"id": int(rid)})  

            st.success("삭제 완료! (가계부 반영)")
            st.cache_data.clear()  
            st.rerun()

        except Exception as e:
            st.error(f"삭제 실패: {e}")







