from datetime import datetime
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from utils_state import init_state
init_state()


st.write("spend_df rows:", len(st.session_state.spend_df))
st.dataframe(st.session_state.spend_df.tail(20), width="stretch")

# 카테고리명 재 분류 필요
DEFAULT_CATEGORIES = [
    "식비", "카페·간식", "교통·차량", "주거·통신", "생활용품",
    "쇼핑·의류", "의료·건강", "교육·자기계발", "문화·여가", "기타"
]

def normalize_month(dt):
    try:
        return datetime.strptime(dt, "%Y-%m-%d %H:%M").strftime("%Y-%m")
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
st.sidebar.header("예산 설정")

if month not in st.session_state.budgets:
    st.session_state.budgets[month] = {c: 0 for c in DEFAULT_CATEGORIES}

for c in DEFAULT_CATEGORIES:
    st.session_state.budgets[month][c] = st.sidebar.number_input(
        c,
        min_value=0,
        step=1000,
        value=st.session_state.budgets[month].get(c, 0)
    )

# -----------------------
# 데이터 필터
# -----------------------
df = st.session_state.spend_df.copy()
df["month"] = df["date_time"].apply(normalize_month)
df = df[df["month"] == month]

st.subheader("지출 내역")
st.dataframe(df, width="stretch")

# -----------------------
# 집계
# -----------------------
summary = df.groupby("category")["amount"].sum().reset_index()
summary["budget"] = summary["category"].map(st.session_state.budgets[month])
summary["remain"] = summary["budget"] - summary["amount"]

st.subheader("카테고리별 요약")
st.dataframe(summary, width="stretch")


