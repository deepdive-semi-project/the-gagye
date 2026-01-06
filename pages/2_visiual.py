from datetime import datetime
import pandas as pd
import streamlit as st

from utils_state import init_state
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
# ✅ SessionState 안전 초기화
# -----------------------
if "budgets" not in st.session_state:
    st.session_state.budgets = {}          # { "YYYY-MM": {category: budget} }
if "total_budgets" not in st.session_state:
    st.session_state.total_budgets = {}    # { "YYYY-MM": total_budget }
if "spend_df" not in st.session_state:
    st.session_state.spend_df = pd.DataFrame(columns=["row_id","date_time","merchant","item","category","amount"])


def normalize_month(dt) -> str:
    """date_time 문자열/값 -> YYYY-MM. 실패 시 Unknown"""
    if dt is None:
        return "Unknown"
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

    # ✅ 위젯 key 값은 콜백에서 업데이트 가능
    st.session_state[misc_key] = new_misc
    st.session_state.budgets[month]["기타"] = new_misc


# -----------------------
# spend_df 확인
# -----------------------
st.caption(f"spend_df rows: {len(st.session_state.spend_df)}")
st.dataframe(st.session_state.spend_df.tail(20), width="stretch")

df_all = st.session_state.spend_df.copy()
if df_all.empty:
    df_all = df_all.assign(month=pd.Series(dtype="object"))
else:
    df_all["month"] = df_all["date_time"].apply(normalize_month)


# -----------------------
# 월 선택 (pending_month 자동 반영)
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

        # 최초 1회만 초기값 세팅
    if total_key not in st.session_state:
        st.session_state[total_key] = int(st.session_state.total_budgets.get(month, 0))

    total_budget = st.number_input(
        "총 예산",
        min_value=0,
        step=10000,
        key=total_key,   # ✅ value= 제거
    )
    st.session_state.total_budgets[month] = int(total_budget)



    # -----------------------
    # ✅ 카테고리 예산 설정
    # -----------------------
    st.divider()
    st.header("예산 설정")

    # number_input key 상태가 우선이므로, key가 없을 때만 budgets 값으로 동기화
    for c in DEFAULT_CATEGORIES:
        k = f"budget_{month}_{c}"
        if k not in st.session_state:
            st.session_state[k] = int(st.session_state.budgets[month].get(c, 0))

        v = st.number_input(
            c,
            min_value=0,
            step=1000,
            value=int(st.session_state[k]),
            key=k,
        )
        st.session_state.budgets[month][c] = int(v)

    # -----------------------
    # ✅ 합계/차이 표시 + 자동 배분 버튼
    # -----------------------
    st.divider()
    cat_budget_sum = int(sum(st.session_state.budgets[month].values()))
    diff = int(total_budget) - int(cat_budget_sum)  # ✅ 방금 입력한 total_budget 기준

    c1, c2 = st.columns(2)
    with c1:
        st.metric("카테고리 예산 합계", f"{cat_budget_sum:,.0f}원")
    with c2:
        if diff >= 0:
            st.metric("미배분", f"{diff:,.0f}원")
        else:
            st.metric("초과 배분", f"{abs(diff):,.0f}원")

    # ✅ on_click 콜백으로 처리 (위젯 key 수정 에러 방지)
    if diff > 0:
        st.button(
            "➕ 미배분을 '기타'에 배분",
            use_container_width=True,
            key=f"alloc_misc_{month}",
            on_click=alloc_to_misc,
            kwargs={"month": month, "diff": int(diff)},
        )


# -----------------------
# 데이터 필터
# -----------------------
df = df_all[df_all["month"] == month].copy()

# category 값이 이상하면 기타로 보정
if not df.empty:
    df["category"] = df["category"].fillna("기타").astype(str).str.strip()
    df.loc[~df["category"].isin(DEFAULT_CATEGORIES), "category"] = "기타"

st.subheader("지출 내역")
st.dataframe(df, width="stretch")


def _ensure_columns():
    needed = ["row_id", "date_time", "merchant", "item", "category", "amount"]
    for c in needed:
        if c not in st.session_state.spend_df.columns:
            st.session_state.spend_df[c] = None
    st.session_state.spend_df["amount"] = pd.to_numeric(
        st.session_state.spend_df["amount"], errors="coerce"
    ).fillna(0.0)


def _next_row_id(n: int = 1):
    base = int(st.session_state.get("row_id_seq", 0))
    st.session_state["row_id_seq"] = base + n
    return list(range(base + 1, base + 1 + n))


def add_one_row(date_time: str, merchant: str, item: str, category: str, amount: float):
    _ensure_columns()
    rid = _next_row_id(1)[0]
    df_new = pd.DataFrame([{
        "row_id": rid,
        "date_time": date_time,
        "merchant": merchant,
        "item": item,
        "category": category,
        "amount": float(amount),
    }])
    st.session_state.spend_df = pd.concat([st.session_state.spend_df, df_new], ignore_index=True)


def delete_rows_by_ids(ids: list[int]):
    _ensure_columns()
    if not ids:
        return
    st.session_state.spend_df = (
        st.session_state.spend_df[~st.session_state.spend_df["row_id"].isin(ids)]
        .reset_index(drop=True)
    )


# -----------------------------
# 🧩 사용자 편집 패널
# -----------------------------
st.markdown("---")
st.subheader("🛠️ 지출 데이터 직접 관리 (추가/수정/삭제)")

_ensure_columns()

tab_add, tab_edit, tab_delete = st.tabs(["➕ 추가", "✏️ 수정", "🗑️ 삭제"])

# -----------------------------
# 1) 추가
# -----------------------------
with tab_add:
    st.caption("직접 1건 추가합니다.")
    with st.form("add_row_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            date_time = st.text_input("일시 (예: 2026-01-05 09:30)", value=datetime.now().strftime("%Y-%m-%d %H:%M"))
        with c2:
            merchant = st.text_input("상호명", value="")
        with c3:
            category = st.selectbox(
                "카테고리",
                options=DEFAULT_CATEGORIES,
                index=DEFAULT_CATEGORIES.index("기타") if "기타" in DEFAULT_CATEGORIES else 0
            )

        c4, c5 = st.columns([2, 1])
        with c4:
            item = st.text_input("품목", value="")
        with c5:
            amount = st.number_input("금액", min_value=0.0, step=100.0, value=0.0)

        submitted = st.form_submit_button("✅ 추가하기", use_container_width=True)

    if submitted:
        add_one_row(date_time=date_time, merchant=merchant, item=item, category=category, amount=amount)
        st.success("추가 완료!")
        st.rerun()

# -----------------------------
# 2) 수정 (data_editor)
# -----------------------------
with tab_edit:
    st.caption("표에서 직접 수정한 뒤, 아래 저장 버튼을 누르면 반영됩니다.")
    df_view = st.session_state.spend_df.copy()

    edited = st.data_editor(
        df_view,
        key="spend_editor",
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "row_id": st.column_config.NumberColumn("row_id", disabled=True),
            "amount": st.column_config.NumberColumn("amount", min_value=0.0, step=100.0),
            "category": st.column_config.SelectboxColumn("category", options=DEFAULT_CATEGORIES),
        },
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("💾 수정 저장", use_container_width=True):
            _ensure_columns()
            edited2 = edited.copy()
            edited2["amount"] = pd.to_numeric(edited2["amount"], errors="coerce").fillna(0.0)

            if "row_id" not in edited2.columns:
                edited2.insert(0, "row_id", _next_row_id(len(edited2)))

            st.session_state.spend_df = edited2[["row_id","date_time","merchant","item","category","amount"]].reset_index(drop=True)
            st.success("저장 완료!")
            st.rerun()

    with c2:
        if st.button("↩️ 수정 취소(원복)", use_container_width=True):
            if "spend_editor" in st.session_state:
                del st.session_state["spend_editor"]
            st.info("편집 화면을 원복했어요.")
            st.rerun()

# -----------------------------
# 3) 삭제 (선택 삭제 / row_id 삭제)
# -----------------------------
with tab_delete:
    st.caption("삭제는 되돌리기 어려우니 주의!")

    df_del = st.session_state.spend_df.copy()
    df_del["delete?"] = False

    df_pick = st.data_editor(
        df_del[["delete?", "row_id", "date_time", "merchant", "item", "category", "amount"]],
        key="delete_picker",
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "delete?": st.column_config.CheckboxColumn("delete?"),
            "row_id": st.column_config.NumberColumn("row_id", disabled=True),
            "category": st.column_config.SelectboxColumn("category", options=DEFAULT_CATEGORIES, disabled=True),
        },
    )

    ids_to_delete = df_pick.loc[df_pick["delete?"] == True, "row_id"].astype(int).tolist()

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button(f"🗑️ 선택 {len(ids_to_delete)}건 삭제", use_container_width=True, disabled=(len(ids_to_delete) == 0)):
            delete_rows_by_ids(ids_to_delete)
            st.success("삭제 완료!")
            st.rerun()

    with c2:
        rid = st.number_input("row_id로 1건 삭제", min_value=0, step=1, value=0)
        if st.button("🗑️ row_id 삭제", use_container_width=True, disabled=(rid == 0)):
            delete_rows_by_ids([int(rid)])
            st.success("삭제 완료!")
            st.rerun()


# -----------------------
# 집계 (✅ 예산 0이어도 카테고리 모두 노출)
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

st.subheader("📊 카테고리별 예산 사용 현황")

cols_per_row = 3
cards = st.columns(cols_per_row)

def status_box(text: str, bg: str, color: str):
    st.markdown(
        f"""
        <div style="
            height:44px;
            border-radius:10px;
            padding:10px 14px;
            background:{bg};
            color:{color};
            font-size:14px;
            font-weight:600;
            display:flex;
            align-items:center;
        ">
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )

for i, (_, r) in enumerate(summary.iterrows()):
    col = cards[i % cols_per_row]

    category = r["category"]
    spent = float(r["amount"])
    budget = float(r["budget"])

    with col:
        st.markdown(
            f"""
            <div style="
                border-radius:16px;
                padding:16px;
                background-color:#111;
                box-shadow:0 4px 12px rgba(0,0,0,0.3);
            ">
                <h4 style="
                    margin-bottom:6px;
                    color:#FFFFFF;
                    font-weight:700;
                    letter-spacing:-0.3px;
                ">{category}</h4>
                <div style="font-size:14px; color:#aaa; margin-bottom:10px;">
                    {spent:,.0f} / {budget:,.0f} 원
                </div>
            """,
            unsafe_allow_html=True
        )

        def status_box(text: str, bg: str, color: str):
            st.markdown(
                f"""
                <div style="
                    height:44px;
                    border-radius:10px;
                    padding:10px 14px;
                    background:{bg};
                    color:{color};
                    font-size:14px;
                    font-weight:500;
                    display:flex;
                    align-items:center;
                    margin-bottom:70px;
                ">
                    {text}
                </div>
                """,
                unsafe_allow_html=True
            )

        # ✅ progress는 무조건 한 번 그려서 '자리'를 맞춘다
        if budget > 0:
            ratio = spent / budget
            pct = int(min(max(ratio, 0), 1) * 100)
        else:
            ratio = 0
            pct = 0

        st.progress(pct)  # ✅ 항상 출력(의료비도 0%로 자리 확보)

        # ✅ 상태 박스도 항상 동일 높이로 출력
        if budget <= 0:
            status_box("예산 미설정", "#e8f2ff", "#2563eb")
        elif ratio >= 1:
            status_box(f"초과 {spent-budget:,.0f}원", "#fee2e2", "#dc2626")
        elif ratio >= 0.8:
            status_box(f"잔여 {budget-spent:,.0f}원", "#fef9c3", "#a16207")
        else:
            status_box(f"여유 {budget-spent:,.0f}원", "#ecfdf5", "#059669")







