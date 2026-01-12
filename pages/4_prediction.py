from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import os
import json
from dotenv import load_dotenv

from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.statespace.sarimax import SARIMAX

from utils_state import init_state, get_db_engine


current_dir = os.path.dirname(os.path.abspath(__file__)) # 현재 파일이 있는 폴더 경로
env_path = os.path.join(current_dir, ".env") # 현재 폴더의 .env 파일 지정
load_dotenv(dotenv_path=env_path, override=True)


# ==================================================
# 기본 설정
# ==================================================
init_state()
engine = get_db_engine()

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="THE 가계 - 지출 예측(SARIMAX)", layout="wide")
st.title("🔮 지출 예측")

if engine is None:
    st.error("DB 엔진(engine)이 None 입니다. (.env / secrets / utils_state 설정 확인)")
    st.stop()

USER_ID = 2  # 필요하면 로그인/세션 값으로 대체


# ==================================================
# 데이터 로드 (일별·카테고리별)
# ==================================================
@st.cache_data(ttl=300)
def load_daily_category(_engine, user_id: int):
    sql = """
    SELECT
        DATE(t.transaction_date) AS d,
        COALESCE(c.category_name, '기타') AS category,
        SUM(t.amount) AS daily_amount
    FROM Transactions t
    LEFT JOIN Category c ON c.id = t.category_id
    WHERE t.type = 'E'
      AND t.user_id = %(uid)s
    GROUP BY d, category
    ORDER BY d, category;
    """
    df = pd.read_sql(sql, _engine, params={"uid": int(user_id)})
    df["d"] = pd.to_datetime(df["d"], errors="coerce")
    df["daily_amount"] = pd.to_numeric(df["daily_amount"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["d"])
    return df


df = load_daily_category(engine, USER_ID)
if df.empty:
    st.error("지출(E) 데이터가 없습니다.")
    st.stop()


# ==================================================
# 유틸: 다음달 기간 계산
# ==================================================
def next_month_range(last_date: pd.Timestamp):
    last_date = pd.to_datetime(last_date).normalize()
    next_month_start = (last_date + pd.offsets.MonthBegin(1)).normalize()
    next_month_end = (next_month_start + pd.offsets.MonthEnd(0)).normalize()
    horizon = (next_month_end - next_month_start).days + 1
    return next_month_start, next_month_end, int(horizon)


# ==================================================
# 예측 모델 (SARIMAX)
# ==================================================
def fallback_forecast_next_month(series: pd.Series, nm_start: pd.Timestamp, horizon: int):
    """
    SARIMAX가 어려운 카테고리(데이터 부족/간헐 지출)도 무조건 결과를 내기 위한 대체 예측.
    - 최근 30일 평균(0 포함) × 다음달 일수
    - 구간은 ±30% (간단한 불확실성 표시)
    """
    s = series.copy().astype(float).sort_index()
    recent = s.tail(30)
    avg = float(recent.mean()) if len(recent) else 0.0

    idx = pd.date_range(nm_start, periods=horizon, freq="D")
    yhat = np.full(horizon, max(avg, 0.0))
    lo = yhat * 0.7
    hi = yhat * 1.3

    fcst = pd.DataFrame({"ds": idx, "yhat": yhat, "yhat_lower": lo, "yhat_upper": hi})

    total = float(fcst["yhat"].sum())
    total_lo = float(fcst["yhat_lower"].sum())
    total_hi = float(fcst["yhat_upper"].sum())

    return (total, total_lo, total_hi), fcst, "fallback_30d_mean"


def sarimax_forecast_next_month(series: pd.Series, nm_start: pd.Timestamp, horizon: int, params: dict):
    """
    일별 → 다음달(일별 경로) 예측 후, 월합으로 합산.
    실패하거나 데이터가 부족하면 None 반환 (caller에서 fallback 처리)

    ※ SARIMAX는 statsmodels에서 ARIMA/SARIMA/ARIMAX를 모두 포함하는 통합 모델 엔진입니다.
    """
    s = series.copy().astype(float)

    cap_q = float(params["cap_q"])
    use_log = bool(params["use_log"])
    drop_zero_days = bool(params["drop_zero_days"])
    use_smoothing = bool(params["use_smoothing"])
    min_active_days = int(params["min_active_days"])
    min_train_len = int(params["min_train_len"])
    use_weekly_season = bool(params["use_weekly_season"])
    order_p, order_d, order_q = int(params["p"]), int(params["d"]), int(params["q"])

    # 상한 캡(너무 강하면 월합 왜곡 가능 → 0.98 이상 권장)
    cap = float(np.quantile(s.values, max(cap_q, 0.98))) if len(s) else 0.0
    if cap > 0:
        s = s.clip(upper=cap)

    # 학습용
    s_train = s[s > 0] if drop_zero_days else s

    active_days = int((s > 0).sum())
    if active_days < min_active_days or s.sum() <= 0 or len(s_train) < min_train_len:
        return None, None, active_days, "insufficient_data"

    y = s_train.sort_index()

    # 월합 목적: 일별 노이즈 완화(추천)
    if use_smoothing:
        y = y.rolling(7, min_periods=1).mean()

    if use_log:
        y = np.log1p(y)

    # 주간 계절성(안정형)
    if use_weekly_season and len(y) >= 45:
        seasonal_order = (1, 1, 1, 7)
    else:
        seasonal_order = (0, 0, 0, 0)

    try:
        model = SARIMAX(
            y,
            order=(order_p, order_d, order_q),
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False)

        pred = fit.get_forecast(steps=horizon)
        mean = pred.predicted_mean
        ci = pred.conf_int(alpha=0.10)  # 90% CI

        idx = pd.date_range(nm_start, periods=horizon, freq="D")
        mean.index = idx
        ci.index = idx

        yhat = mean.astype(float).values
        lo = ci.iloc[:, 0].astype(float).values
        hi = ci.iloc[:, 1].astype(float).values

        if use_log:
            yhat = np.expm1(yhat)
            lo = np.expm1(lo)
            hi = np.expm1(hi)

        # 음수 방지
        yhat = np.clip(yhat, 0, None)
        lo = np.clip(lo, 0, None)
        hi = np.clip(hi, 0, None)

        # 안전장치: 예측이 과하게 커지면 최근 분포 기준으로 캡
        recent_cap = float(np.quantile(s.tail(90).values, 0.995)) if len(s) >= 30 else float(np.quantile(s.values, 0.995))
        if recent_cap > 0:
            yhat = np.clip(yhat, 0, recent_cap)
            lo = np.clip(lo, 0, recent_cap)
            hi = np.clip(hi, 0, recent_cap)

        fcst = pd.DataFrame({"ds": idx, "yhat": yhat, "yhat_lower": lo, "yhat_upper": hi})

        total = float(fcst["yhat"].sum())
        total_lo = float(fcst["yhat_lower"].sum())
        total_hi = float(fcst["yhat_upper"].sum())

        return (total, total_lo, total_hi), fcst, active_days, "sarimax"

    except Exception as e:
        return None, None, active_days, f"sarimax_error:{type(e).__name__}"


def build_forecast_next_month(pivot_df: pd.DataFrame, params: dict):
    results = []
    cache = {}

    last_day = pivot_df.index.max()
    nm_start, nm_end, horizon = next_month_range(last_day)

    for cat in pivot_df.columns:
        s = pivot_df[cat].rename(cat)

        totals, fcst, active, method = sarimax_forecast_next_month(s, nm_start, horizon, params)

        if totals is None:
            totals, fcst, fb_method = fallback_forecast_next_month(s, nm_start, horizon)
            method = fb_method

        total, lo, hi = totals
        cache[cat] = (s, fcst)

        last_month_sum = float(s.loc[s.index.to_period("M") == last_day.to_period("M")].sum())

        results.append({
            "category": cat,
            "next_month": nm_start.strftime("%Y-%m"),
            "predicted": total,
            "lower": lo,
            "upper": hi,
            "last_month": last_month_sum,
            "active_days": int(active),
        })

    pred_df = pd.DataFrame(results)
    if not pred_df.empty:
        pred_df = pred_df.sort_values("predicted", ascending=False).reset_index(drop=True)

    meta = {
        "last_day": last_day,
        "next_month_start": nm_start,
        "next_month_end": nm_end,
        "horizon_days": horizon,
    }
    return pred_df, cache, meta


# ==================================================
# 1) 사이드바: "처음엔 접힌" 예측 방식 설정 패널
# 2) 페이지 진입 시: 기본 파라미터로 자동 예측 1회 수행
# 3) 사용자가 조정 후 버튼 누르면: 그 파라미터로 재예측
# ==================================================

DEFAULT_PARAMS = {
    "min_active_days": 25,
    "p": 1,
    "d": 1,
    "q": 1,
    "use_weekly_season": True,
    "use_log": True,
    "cap_q": 0.98,
    "drop_zero_days": False,
    "use_smoothing": True,
    "min_train_len": 30,
}

if "nm_params" not in st.session_state:
    st.session_state.nm_params = DEFAULT_PARAMS.copy()

if "pred_df_nm" not in st.session_state:
    st.session_state.pred_df_nm = pd.DataFrame()
if "pred_cache_nm" not in st.session_state:
    st.session_state.pred_cache_nm = {}
if "pred_meta_nm" not in st.session_state:
    st.session_state.pred_meta_nm = {}

# 사이드바 UI (처음엔 접힘)
st.sidebar.header("⚙️ 설정")
st.sidebar.caption("기본값으로 예측이 먼저 표시됩니다. 필요하면 아래 패널을 펼쳐 조정하고 실행하세요.")

categories_all = sorted(df["category"].unique().tolist())
selected_categories = st.sidebar.multiselect("카테고리(예측 포함)", categories_all, default=categories_all)

with st.sidebar.expander("🔧 예측 방식 설정 (고급)", expanded=False):
    st.markdown("#### 데이터 포함 기준")
    min_active_days = st.slider("최소 유효(지출>0) 일 수", 5, 365, st.session_state.nm_params["min_active_days"])
    with st.expander("ℹ️ 의미", expanded=False):
        st.markdown(
            "- 이 카테고리는 **지출이 있었던 날이 최소 몇 번 이상** 있어야 예측할지 정하는 값이에요.\n"
            "- 값을 **낮추면** 지출 기록이 적은 카테고리도 예측에 포함되지만, 결과가 **덜 믿을 수** 있어요.\n"
            "- 값을 **높이면** 기록이 충분한 카테고리만 예측해서 결과가 **더 안정적**이에요.\n"
            "- 참고: 기록이 너무 적으면, 이 앱은 SARIMAX 대신 **간단한 방식(Fallback)** 으로 예측값을 보여줄 수 있어요."
        )

    min_train_len = st.slider("최소 학습 길이(일)", 7, 90, st.session_state.nm_params["min_train_len"])
    with st.expander("ℹ️ 의미", expanded=False):
        st.markdown(
            "- 모델이 예측을 하기 전에, **최소 며칠치 데이터를 보고 배울지** 정하는 값이에요.\n"
            "- 값이 너무 작으면 패턴을 제대로 못 배워서 예측이 **튀거나 불안정**해질 수 있어요.\n"
            "- 보통은 **30일 이상**을 추천해요."
        )

    st.markdown("#### SARIMAX 구조(p,d,q)")
    p = st.selectbox("p (AR)", [0, 1, 2], index=[0, 1, 2].index(st.session_state.nm_params["p"]))
    with st.expander("ℹ️ 의미", expanded=False):
        st.markdown(
            "- 예측할 때 **최근 며칠의 흐름(관성)**을 얼마나 참고할지 정하는 값이에요.\n"
            "- 값이 커질수록 최근 패턴을 더 따라가지만, 너무 크면 **요즘 데이터에 과하게 끌려** 예측이 흔들릴 수 있어요.\n"
            "- 잘 모르겠으면 **1**이 무난해요."
        )

    d = st.selectbox("d (차분)", [0, 1], index=[0, 1].index(st.session_state.nm_params["d"]))
    with st.expander("ℹ️ 의미", expanded=False):
        st.markdown(
            "- 지출이 시간이 지날수록 **전반적으로 늘거나(상승) 줄어드는(하락) 경향**이 있으면 예측이 어려울 수 있어요.\n"
            "- d=1은 이런 ‘전체적인 기울기’를 한 번 정리해서 모델이 **더 안정적으로 학습**하도록 도와줘요.\n"
            "- 보통은 **1**을 많이 사용해요."
        )

    q = st.selectbox("q (MA)", [0, 1, 2], index=[0, 1, 2].index(st.session_state.nm_params["q"]))
    with st.expander("ℹ️ 의미", expanded=False):
        st.markdown(
            "- 예측할 때 **최근의 흔들림(오차/변동)**을 얼마나 반영할지 정하는 값이에요.\n"
            "- 값이 커질수록 단기 변동을 더 따라가지만, 너무 크면 **예측이 불안정**해질 수 있어요.\n"
            "- 잘 모르겠으면 **1**이 무난해요."
        )

    st.markdown("#### 계절/전처리 옵션")
    use_weekly_season = st.checkbox("주간(7일) 계절성 사용", value=st.session_state.nm_params["use_weekly_season"])
    with st.expander("ℹ️ 의미", expanded=False):
        st.markdown(
            "- 월요일~일요일처럼 **요일에 따라 소비가 달라지는 패턴**이 있으면 켜는 것이 좋아요.\n"
            "- 예: 주말에 외식/카페 지출이 늘어나는 경우\n"
            "- 데이터가 아주 적으면 효과가 크지 않을 수 있어요."
        )

    use_smoothing = st.checkbox("7일 이동평균으로 노이즈 완화(추천)", value=st.session_state.nm_params["use_smoothing"])
    with st.expander("ℹ️ 의미", expanded=False):
        st.markdown(
            "- 일별 지출은 원래 **들쭉날쭉**해서 예측이 튈 수 있어요.\n"
            "- 이 옵션을 켜면, 최근 7일을 평균 내서 **그래프를 부드럽게** 만들고 예측을 더 안정적으로 해줘요.\n"
            "- 다음달 ‘월 합계’ 예측이 목적이면 **켜두는 편이 보통 더 좋아요**."
        )

    use_log = st.checkbox("log 변환 사용(추천)", value=st.session_state.nm_params["use_log"])
    with st.expander("ℹ️ 의미", expanded=False):
        st.markdown(
            "- 가끔 **한 번에 큰 지출**(예: 가전, 여행, 병원비)이 있으면 예측이 그쪽으로 크게 흔들릴 수 있어요.\n"
            "- log 변환은 큰 값을 ‘완만하게’ 만들어서 **예측이 과하게 튀는 것을 줄여줘요**.\n"
            "- 대부분의 가계부 데이터에서는 **켜두는 것이 안정적**이에요."
        )

    cap_q = st.select_slider("이상치 상한 캡(quantile)", options=[0.90, 0.95, 0.98, 0.99], value=st.session_state.nm_params["cap_q"])
    with st.expander("ℹ️ 의미", expanded=False):
        st.markdown(
            "- 아주 큰 지출이 한 번 있으면 예측이 ‘그 정도가 계속 나갈 것’처럼 커질 수 있어요.\n"
            "- 이 옵션은 **상위 몇 %의 큰 지출을 적당히 제한**해서 예측이 망가지지 않게 해줘요.\n"
            "- 값이 **0.99**면 거의 안 자르고, **0.90**이면 많이 자르는 편이에요.\n"
            "- 월 합계 예측은 보통 **0.98~0.99**가 무난해요."
        )

    drop_zero_days = st.checkbox("0원 날은 학습에서 제외", value=st.session_state.nm_params["drop_zero_days"])
    with st.expander("ℹ️ 의미", expanded=False):
        st.markdown(
            "- 지출이 없는 날(0원)이 많으면 평균이 낮아져서 예측이 작게 나올 수 있어요.\n"
            "- 이 옵션을 켜면 **지출이 있었던 날만 보고** 패턴을 학습해요.\n"
            "- 다만 ‘가끔만 쓰는 카테고리(간헐 지출)’는 0원이 중요한 정보라서,\n"
            "  이 옵션을 켜면 오히려 예측이 **과하게 커질 수 있어요**. 잘 모르겠으면 **끄는 것을 추천**해요."
        )

st.sidebar.divider()
rerun = st.sidebar.button("🚀 예측 방식 설정을 적용해 다시 예측")

current_params = st.session_state.nm_params.copy()

if rerun:
    st.session_state.nm_params = {
        "min_active_days": int(min_active_days),
        "p": int(p),
        "d": int(d),
        "q": int(q),
        "use_weekly_season": bool(use_weekly_season),
        "use_log": bool(use_log),
        "cap_q": float(cap_q),
        "drop_zero_days": bool(drop_zero_days),
        "use_smoothing": bool(use_smoothing),
        "min_train_len": int(min_train_len),
    }
    current_params = st.session_state.nm_params.copy()

# ==================================================
# 페이지 진입 시 자동 예측 1회
# ==================================================
pivot = (
    df[df["category"].isin(selected_categories)]
    .pivot_table(index="d", columns="category", values="daily_amount", aggfunc="sum")
    .fillna(0.0)
    .sort_index()
)

full_idx = pd.date_range(pivot.index.min(), pivot.index.max(), freq="D")
pivot = pivot.reindex(full_idx).fillna(0.0)
pivot.index.name = "ds"

cats_key = tuple(pivot.columns.tolist())
if "cats_key_nm" not in st.session_state:
    st.session_state.cats_key_nm = None

need_auto_run = False
if st.session_state.pred_df_nm.empty:
    need_auto_run = True
elif st.session_state.cats_key_nm != cats_key:
    need_auto_run = True
elif rerun:
    need_auto_run = True

if need_auto_run:
    with st.spinner("다음달 예측 계산 중..."):
        pred_df_nm, cache_nm, meta_nm = build_forecast_next_month(pivot, current_params)
        st.session_state.pred_df_nm = pred_df_nm
        st.session_state.pred_cache_nm = cache_nm
        st.session_state.pred_meta_nm = meta_nm
        st.session_state.cats_key_nm = cats_key

pred_df = st.session_state.pred_df_nm
cache = st.session_state.pred_cache_nm
meta = st.session_state.pred_meta_nm

if pred_df.empty:
    st.warning("예측 결과가 없습니다. (데이터/카테고리 선택을 확인하세요)")
    st.stop()

# ==================================================
# 결과 표시
# ==================================================
nm_label = meta["next_month_start"].strftime("%Y-%m")
horizon = meta["horizon_days"]

total_pred = float(pred_df["predicted"].sum())
total_last = float(pred_df["last_month"].sum())

c1, c2, c3 = st.columns(3)
c1.metric("예측 대상 월", nm_label, f"{horizon}일")
c2.metric("다음달 총 지출 예측", f"{total_pred:,.0f}원", f"{(total_pred-total_last):,.0f}원")
c3.metric("예측 카테고리 수", len(pred_df))

st.divider()

st.subheader("📌 다음달 카테고리별 예측 (월 합계)")

fig = plt.figure(figsize=(9, 4))
plt.bar(pred_df["category"], pred_df["predicted"])
plt.xticks(rotation=45, ha="right")
ax = plt.gca()
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x/10000):,}"))
plt.ylabel("예측 지출 (만원)")
plt.tight_layout()
st.pyplot(fig)

view_df = pred_df.copy()
for col in ["predicted", "lower", "upper", "last_month"]:
    view_df[f"{col}_만원"] = (view_df[col] / 10000).round(1)

view_df = view_df.drop(columns=["method"], errors="ignore")

st.dataframe(
    view_df.rename(columns={
        "category": "카테고리",
        "next_month": "예측월",
        "predicted": "예측(원, 월합)",
        "predicted_만원": "예측(만원)",
        "lower": "하한(원, 월합)",
        "lower_만원": "하한(만원)",
        "upper": "상한(원, 월합)",
        "upper_만원": "상한(만원)",
        "last_month": "전월(원, 월합)",
        "last_month_만원": "전월(만원)",
        "active_days": "유효일수(지출>0)",
    }),
    use_container_width=True
)

st.divider()

# ==================================================
# LLM 피드백 (예측 결과 기반) - 버튼 눌렀을 때만 생성
# ==================================================
st.subheader("💬 AI 피드백 (예측 결과 기반)")

# --- 사용자에게 보여줄 입력 요약 만들기 ---
def build_feedback_payload_from_pred(pred_df: pd.DataFrame, nm_label: str):
    df_send = pred_df.copy()

    # LLM이 헷갈리지 않게 정수로 정리(원 단위)
    for col in ["predicted", "lower", "upper", "last_month"]:
        if col in df_send.columns:
            df_send[col] = df_send[col].fillna(0).round(0).astype(int)

    total_pred = int(df_send["predicted"].sum()) if "predicted" in df_send.columns else 0
    total_last = int(df_send["last_month"].sum()) if "last_month" in df_send.columns else 0
    delta = total_pred - total_last

    payload = {
        "target_month": nm_label,
        "total_pred": total_pred,
        "total_last": total_last,
        "delta": delta,
        "currency": "KRW",
        "rows": df_send[[
            "category", "predicted", "lower", "upper", "last_month", "active_days"
        ]].to_dict(orient="records")
    }
    return payload



# --- 규칙 기반 fallback (API 실패/키 없음 대비) ---
def rule_based_feedback(view_df: pd.DataFrame, nm_label: str) -> str:
    df2 = view_df.copy()
    if "예측(원, 월합)" not in df2.columns or "전월(원, 월합)" not in df2.columns:
        return f"{nm_label} 예측 결과가 준비됐어요. (전월 비교 컬럼이 없어 간단 요약만 제공해요.)"

    df2["증감(원)"] = df2["예측(원, 월합)"] - df2["전월(원, 월합)"]
    denom = df2["전월(원, 월합)"].replace(0, np.nan)
    df2["증감(%)"] = (df2["증감(원)"] / denom * 100).round(1)

    total_pred = float(df2["예측(원, 월합)"].sum())
    total_last = float(df2["전월(원, 월합)"].sum())
    delta = total_pred - total_last
    delta_pct = (delta / total_last * 100) if total_last > 0 else np.nan

    top_up = df2.sort_values("증감(원)", ascending=False).head(3)
    top_down = df2.sort_values("증감(원)", ascending=True).head(3)

    lines = []
    if not np.isnan(delta_pct):
        lines.append(f"- **{nm_label} 총 지출 예측**: {total_pred:,.0f}원 (전월 대비 {delta:,.0f}원, {delta_pct:.1f}%)")
    else:
        lines.append(f"- **{nm_label} 총 지출 예측**: {total_pred:,.0f}원 (전월 대비 {delta:,.0f}원)")

    lines.append("")
    lines.append("**증가가 큰 카테고리 TOP 3**")
    for _, r in top_up.iterrows():
        cat = r.get("카테고리", "-")
        lines.append(f"- {cat}: {r['예측(원, 월합)']:,.0f}원 (전월 대비 +{r['증감(원)']:,.0f}원)")

    lines.append("")
    lines.append("**감소가 큰 카테고리 TOP 3**")
    for _, r in top_down.iterrows():
        cat = r.get("카테고리", "-")
        lines.append(f"- {cat}: {r['예측(원, 월합)']:,.0f}원 (전월 대비 {r['증감(원)']:,.0f}원)")

    lines.append("")
    lines.append("**추천 액션(간단)**")
    lines.append("- 증가 TOP 카테고리 1~2개에 **다음달 예산 상한**을 먼저 정하고, 주 1회만 점검해보세요.")
    lines.append("- ‘간헐 지출’(전월이 0에 가까움)은 예측이 튈 수 있어 **참고용**으로 보고 계획 소비와 함께 판단하세요.")
    return "\n".join(lines)


# --- OpenAI 호출 ---
def llm_feedback_openai(payload: dict) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    system = (
        "너는 개인 가계부 예측 결과를 보고, 사용자가 이해하기 쉬운 한국어로 피드백을 주는 재무 코치다. "
        "과도한 확신은 피하고, 예측의 불확실성을 분명히 말한다. "
        "출력은 마크다운으로, 짧고 실행 가능한 조언 위주로 작성한다."
    )

    user = f"""
다음은 사용자의 {payload["target_month"]} 지출 예측 결과다.

중요 규칙:
- **총 지출 예측은 payload.total_pred 값을 그대로 사용해라. 절대 다시 합산/추정하지 마라.**
- 전월 총합도 payload.total_last 값을 그대로 사용해라.
- 카테고리별 값은 rows의 predicted/lower/upper/last_month를 사용해라.

요청:
1) 총 지출 변화 요약(전월 대비 증가/감소) — 반드시 total_pred, total_last, delta 사용
2) 증가 위험이 큰 카테고리 TOP 3(증감 큰 순)와 가능한 해석
3) 줄이기 쉬운 카테고리/전략 제안 3개 (행동 단위)
4) 예측 한계(간헐 지출/데이터 부족 등) 2~3줄

데이터(JSON):
{json.dumps(payload, ensure_ascii=False)}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content



# =========================
#        AI 피드백 
# =========================
payload = build_feedback_payload_from_pred(view_df, nm_label)

payload_key = (
    nm_label,
    tuple(view_df.get("카테고리", pd.Series([], dtype=str)).astype(str).tolist()),
    float(view_df.get("예측(원, 월합)", pd.Series([0.0])).sum()),
    float(view_df.get("전월(원, 월합)", pd.Series([0.0])).sum()) if "전월(원, 월합)" in view_df.columns else 0.0
)

if "llm_feedback" not in st.session_state:
    st.session_state.llm_feedback = ""
if "llm_feedback_key" not in st.session_state:
    st.session_state.llm_feedback_key = None

cbtn1, cbtn2 = st.columns([1, 1])
with cbtn1:
    generate_feedback = st.button("📝 AI 피드백 생성", type="primary")
    st.caption(
    "AI가 예측 데이터를 기반으로 피드백을 제공합니다."
    )
with cbtn2:
    regen = st.button("🔄 피드백 다시 생성")
    st.caption(
    "현재 화면의 예측 숫자는 그대로 두고, "
    "AI가 설명과 조언만 새로 제공합니다."
    )

if generate_feedback or regen:
    with st.spinner("AI 피드백 생성 중..."):
        try:
            text = llm_feedback_openai(payload)
        except Exception:
            text = rule_based_feedback(view_df, nm_label)

        st.session_state.llm_feedback = text
        st.session_state.llm_feedback_key = payload_key

if st.session_state.llm_feedback and st.session_state.llm_feedback_key != payload_key:
    st.warning("예측 결과가 변경되었습니다. 최신 예측 기준으로 피드백을 보려면 'AI 피드백 생성'을 눌러주세요.")

if not st.session_state.llm_feedback:
    st.info("버튼을 누르면 현재 예측 결과를 바탕으로 AI 피드백을 생성해드려요.")
else:
    st.markdown(st.session_state.llm_feedback)










