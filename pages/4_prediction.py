# pages/4_prediction.py
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from prophet import Prophet
from matplotlib.ticker import FuncFormatter

# ✅ utils_state에서 초기화 + DB 엔진 가져오기
from utils_state import init_state, get_db_engine

# -----------------------
# 기본 설정
# -----------------------
init_state()
engine = get_db_engine()

plt.rcParams["font.family"] = "Malgun Gothic"   # Windows
plt.rcParams["axes.unicode_minus"] = False

st.title("🔮 지출 예측 (월별 · 카테고리별)")

if engine is None:
    st.error("DB 엔진(engine)이 None 입니다. (.env / secrets / utils_state 설정 확인)")
    st.stop()

# -----------------------
# 데이터 로드 (월별·카테고리별 집계)
# - type='E' (지출)만
# - user_id 필터(필요시)
# -----------------------
USER_ID = 2  # 필요하면 로그인/세션 값으로 대체

@st.cache_data(ttl=300)
def load_monthly_category(_engine, user_id: int):
    sql = """
    SELECT
        DATE_FORMAT(t.transaction_date, '%%Y-%%m') AS ym,
        COALESCE(c.category_name, '기타') AS category,
        SUM(t.amount) AS monthly_amount
    FROM Transactions t
    LEFT JOIN Category c ON c.id = t.category_id
    WHERE t.type = 'E'
      AND t.user_id = %(uid)s
    GROUP BY ym, category
    ORDER BY ym, category;
    """
    df = pd.read_sql(sql, engine, params={"uid": int(user_id)})
    df["ym"] = pd.to_datetime(df["ym"] + "-01", errors="coerce")
    df["monthly_amount"] = pd.to_numeric(df["monthly_amount"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["ym"])
    return df

df = load_monthly_category(engine, USER_ID)

if df.empty:
    st.error("지출(E) 데이터가 없습니다.")
    st.stop()

# -----------------------
# Sidebar: 설정 + 설명 UI
# -----------------------
st.sidebar.header("🧠 지출 예측 설정")

categories_all = sorted(df["category"].unique().tolist())

st.sidebar.subheader("카테고리 선택 (학습 대상)")
st.sidebar.caption("선택한 카테고리만 예측 모델에 포함됩니다.")
selected_categories = st.sidebar.multiselect(
    "카테고리",
    categories_all,
    default=categories_all
)
with st.sidebar.expander("ℹ️ 의미"):
    st.markdown(
        "- 예측 모델이 **학습하고 결과를 보여줄 카테고리**를 선택합니다.\n"
        "- 선택하지 않은 카테고리는 **예측에서 제외**됩니다.\n"
        "- 데이터가 적은 카테고리는 제외하면 **예측이 안정적**입니다."
    )

st.sidebar.subheader("최소 유효(지출>0) 월 수")
min_active_months = st.sidebar.slider("최소 유효 월 수", 4, 18, 6)
with st.sidebar.expander("ℹ️ 의미"):
    st.markdown(
        "- 해당 카테고리에서 **지출이 실제로 발생(>0)한 달 수** 기준입니다.\n"
        "- 이 값보다 적으면 예측을 수행하지 않습니다.\n"
        "- 데이터가 너무 적을 때 생기는 **엉뚱한 예측을 방지**합니다."
    )

st.sidebar.subheader("추세 민감도 (작을수록 완만)")
cps = st.sidebar.select_slider(
    "추세 민감도",
    options=[0.005, 0.01, 0.03, 0.05],
    value=0.01
)
with st.sidebar.expander("ℹ️ 의미"):
    st.markdown(
        "- 지출 추세 변화에 **얼마나 민감하게 반응할지**를 조절합니다.\n"
        "- 작을수록 곡선이 부드럽고(안정적), 클수록 최근 변화에 민감하지만 과적합 위험이 커집니다."
    )

st.sidebar.subheader("연간 계절성")
yearly_opt = st.sidebar.selectbox("연간 계절성", ["Auto", "끄기(False)", "켜기(True)"], index=0)
if yearly_opt == "Auto":
    yearly_seasonality = "auto"
elif yearly_opt == "끄기(False)":
    yearly_seasonality = False
else:
    yearly_seasonality = True
with st.sidebar.expander("ℹ️ 의미"):
    st.markdown(
        "- **매년 반복되는 소비 패턴**(명절, 계절요인 등)을 반영할지 설정합니다.\n"
        "- 데이터가 1년 미만이면 켜면 과적합될 수 있어 Auto/False가 유리할 때가 많습니다."AAA
    )

st.sidebar.subheader("log 변환 사용 (추천)")
use_log = st.sidebar.checkbox("log 변환 사용", value=True)
with st.sidebar.expander("ℹ️ 의미"):
    st.markdown(
        "- 큰 금액(이상치) 때문에 예측이 튀는 것을 줄이기 위해 **log 스케일로 학습**합니다.\n"
        "- 대부분의 가계부 지출에서는 켜두는 것이 안정적입니다."
    )

st.sidebar.subheader("이상치 상한 캡(quantile)")
cap_q = st.sidebar.select_slider("상한 캡", options=[0.90, 0.95, 0.98, 0.99], value=0.95)
with st.sidebar.expander("ℹ️ 의미"):
    st.markdown(
        "- 상위 일부 극단적인 지출을 제한(clip)해 **한 번의 큰 소비가 예측을 망치지 않게** 합니다.\n"
        "- 일반 가계부는 0.95가 무난합니다."
    )

st.sidebar.subheader("0원 달은 학습에서 제외")
drop_zero_months = st.sidebar.checkbox("0원 달은 학습에서 제외", value=True)
with st.sidebar.expander("ℹ️ 의미"):
    st.markdown(
        "- 지출이 없던 달을 0으로 학습하면 평균이 낮아져 예측이 왜곡될 수 있어 제외합니다.\n"
        "- 여행/문화비처럼 **간헐 지출**에 특히 효과적입니다."
    )

st.sidebar.divider()
run_forecast = st.sidebar.button("🚀 예측 실행(학습)")
st.sidebar.caption("버튼을 눌렀을 때만 모델 학습/예측을 수행합니다. (결과는 저장되어 탐색은 즉시 가능)")

# -----------------------
# Pivot 만들기(월×카테고리)
# -----------------------
pivot = (
    df[df["category"].isin(selected_categories)]
    .pivot_table(index="ym", columns="category", values="monthly_amount", aggfunc="sum")
    .fillna(0.0)
    .sort_index()
)

# 결측 월 보정(축 유지)
full_idx = pd.date_range(pivot.index.min(), pivot.index.max(), freq="MS")
pivot = pivot.reindex(full_idx).fillna(0.0)
pivot.index.name = "ds"

# -----------------------
# Prophet 예측 함수
# -----------------------
def prophet_predict(series: pd.Series):
    s = series.copy()

    # 상한 캡
    cap = float(np.quantile(s.values, cap_q)) if len(s) else 0.0
    if cap > 0:
        s = s.clip(upper=cap)

    # 학습용 시계열 (0원 제외 옵션)
    s_train = s[s > 0] if drop_zero_months else s

    active_months = int((s > 0).sum())
    if active_months < int(min_active_months) or s.sum() <= 0 or len(s_train) < 2:
        return None, None, active_months

    df_p = s_train.reset_index()
    df_p.columns = ["ds", "y"]

    # log 변환
    if use_log:
        df_p["y"] = np.log1p(df_p["y"])

    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=float(cps),
    )
    model.fit(df_p)

    future = model.make_future_dataframe(periods=1, freq="MS")
    fcst = model.predict(future)

    # 역변환
    if use_log:
        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            fcst[col] = np.expm1(fcst[col])

    # 음수 방지
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        fcst[col] = fcst[col].clip(lower=0)

    return s, fcst, active_months

# -----------------------
# 예측 실행(버튼) → 결과를 session_state에 저장
# -----------------------
def build_forecast(pivot_df: pd.DataFrame):
    results = []
    cache = {}

    for cat in pivot_df.columns:
        s = pivot_df[cat].rename(cat)
        s_raw, fcst, active = prophet_predict(s)
        if fcst is None:
            continue

        next_row = fcst.iloc[-1]
        cache[cat] = (s, fcst)

        results.append({
            "category": cat,
            "next_month": pd.to_datetime(next_row["ds"]).strftime("%Y-%m"),
            "predicted": float(next_row["yhat"]),
            "lower": float(next_row["yhat_lower"]),
            "upper": float(next_row["yhat_upper"]),
            "last_month": float(s.iloc[-1]),
            "active_months": int(active),
        })

    pred_df = pd.DataFrame(results)
    if not pred_df.empty:
        pred_df = pred_df.sort_values("predicted", ascending=False).reset_index(drop=True)

    return pred_df, cache

if "pred_df" not in st.session_state:
    st.session_state.pred_df = pd.DataFrame()
if "pred_cache" not in st.session_state:
    st.session_state.pred_cache = {}

# 버튼 눌렀을 때만 학습/예측
if run_forecast:
    with st.spinner("예측 모델 학습 중..."):
        pred_df, cache = build_forecast(pivot)
        st.session_state.pred_df = pred_df
        st.session_state.pred_cache = cache

    if pred_df.empty:
        st.warning("예측 가능한 카테고리가 없습니다. (유효 월 수/카테고리 선택을 조정해보세요)")
    else:
        st.success("예측 완료! 아래에서 카테고리별로 탐색할 수 있어요.")

# -----------------------
# 결과 표시(탐색): 버튼 없이도 session_state 결과로 탐색 가능
# -----------------------
pred_df = st.session_state.pred_df
cache = st.session_state.pred_cache

if pred_df.empty:
    st.info("왼쪽에서 설정을 조정한 뒤 **🚀 예측 실행(학습)** 버튼을 눌러 결과를 생성하세요.")
    st.stop()

# KPI
total_pred = float(pred_df["predicted"].sum())
total_last = float(pred_df["last_month"].sum())

c1, c2, c3 = st.columns(3)
c1.metric("예측 월", pred_df["next_month"].iloc[0])
c2.metric("다음 달 총 지출 예측", f"{total_pred:,.0f}원", f"{(total_pred-total_last):,.0f}원")
c3.metric("예측 카테고리 수", len(pred_df))

st.divider()

# -----------------------
# 차트: 다음 달 카테고리별 예측 (만원 단위)
# -----------------------
st.subheader("📌 다음 달 카테고리별 예측 (금액)")

fig = plt.figure(figsize=(9, 4))
plt.bar(pred_df["category"], pred_df["predicted"])
plt.xticks(rotation=45, ha="right")

ax = plt.gca()
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x/10000):,}"))

plt.ylabel("예측 지출 (만원)")
plt.tight_layout()
st.pyplot(fig)

# 표 (원 + 만원 같이)
view_df = pred_df.copy()
view_df["predicted_만원"] = (view_df["predicted"] / 10000).round(1)
view_df["last_month_만원"] = (view_df["last_month"] / 10000).round(1)
view_df["lower_만원"] = (view_df["lower"] / 10000).round(1)
view_df["upper_만원"] = (view_df["upper"] / 10000).round(1)

st.dataframe(
    view_df.rename(columns={
        "category": "카테고리",
        "next_month": "예측월",
        "predicted": "예측(원)",
        "predicted_만원": "예측(만원)",
        "lower": "하한(원)",
        "lower_만원": "하한(만원)",
        "upper": "상한(원)",
        "upper_만원": "상한(만원)",
        "last_month": "전월(원)",
        "last_month_만원": "전월(만원)",
        "active_months": "유효월수(지출>0)",
    }),
    use_container_width=True
)

st.divider()

# -----------------------
# 상세: 카테고리별 실제 vs 예측 (만원 단위)
# -----------------------
st.subheader("🔍 카테고리별 실제 vs 예측")

selected = st.selectbox("카테고리 선택", pred_df["category"].tolist(), index=0)

series, fcst = cache[selected]
actual_df = pd.DataFrame({"ds": series.index, "actual": series.values})

plot_df = fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]].merge(actual_df, on="ds", how="left")

fig2 = plt.figure(figsize=(9, 4))
plt.plot(plot_df["ds"], plot_df["yhat"], label="예측")
plt.fill_between(plot_df["ds"], plot_df["yhat_lower"], plot_df["yhat_upper"], alpha=0.2)
plt.plot(plot_df["ds"], plot_df["actual"], label="실제")

ax2 = plt.gca()
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x/10000):,}"))

plt.ylabel("지출 (만원)")
plt.legend()
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig2)
