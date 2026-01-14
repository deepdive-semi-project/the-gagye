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


current_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 파일이 있는 폴더 경로
env_path = os.path.join(current_dir, ".env")              # 현재 폴더의 .env 파일 지정
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
# 예측 대상 월 선택 (예: 2026-01)
# - 선택한 월의 1일을 기준으로, 그 전날까지의 데이터만 학습에 사용
# ==================================================
max_d = df["d"].max().normalize()
default_target_month = (max_d + pd.offsets.MonthBegin(1)).strftime("%Y-%m")  # 기본: 다음달

target_month = st.sidebar.text_input(
    "예측할 월 (YYYY-MM)",
    value=default_target_month,
    help="예: 2026-01 을 입력하면, 2025-12-31 까지 데이터로 2026-01을 예측합니다."
)

try:
    target_start = pd.to_datetime(f"{target_month}-01").normalize()
except Exception:
    st.sidebar.error("예측할 월 형식이 올바르지 않습니다. 예: 2026-01")
    st.stop()

target_end = (target_start + pd.offsets.MonthEnd(0)).normalize()
horizon = int((target_end - target_start).days + 1)

# 학습 데이터는 target_start 전날까지만
train_end = (target_start - pd.Timedelta(days=1)).normalize()

if train_end < df["d"].min().normalize():
    st.sidebar.warning("선택한 예측월이 너무 과거라 학습 데이터가 부족합니다.")


# ==================================================
# 유틸: 다음달 기간 계산
# ==================================================
def target_month_range(target_start: pd.Timestamp):
    target_start = pd.to_datetime(target_start).normalize()
    target_end = (target_start + pd.offsets.MonthEnd(0)).normalize()
    horizon = (target_end - target_start).days + 1
    return target_start, target_end, int(horizon)


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
    """
    s = series.copy().astype(float)

    # ✅ params 안전 접근 + 기본값(추천값)으로 폴백
    cap_q = float(params.get("cap_q", 0.98))

    use_log = bool(params.get("use_log", True))
    drop_zero_days = bool(params.get("drop_zero_days", False))
    use_smoothing = bool(params.get("use_smoothing", True))

    min_active_days = int(params.get("min_active_days", 10))  # ✅ 디폴트 10으로
    min_train_len = int(params.get("min_train_len", 30))
    use_weekly_season = bool(params.get("use_weekly_season", True))

    order_p = int(params.get("p", 1))
    order_d = int(params.get("d", 1))
    order_q = int(params.get("q", 1))

    # 상한 캡
    cap = float(np.quantile(s.values, max(cap_q, 0.98))) if len(s) else 0.0
    if cap > 0:
        s = s.clip(upper=cap)

    # 학습용
    s_train = s[s > 0] if drop_zero_days else s

    active_days = int((s > 0).sum())
    if active_days < min_active_days or s.sum() <= 0 or len(s_train) < min_train_len:
        return None, None, active_days, "insufficient_data"

    y = s_train.sort_index()

    # 노이즈 완화(추천)
    if use_smoothing:
        y = y.rolling(7, min_periods=1).mean()

    if use_log:
        y = np.log1p(y)

    # 주간 계절성
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

        # 안전 캡
        recent_cap = float(np.quantile(s.tail(90).values, 0.995)) if len(s) >= 15 else float(np.quantile(s.values, 0.995))
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


def build_forecast_for_target_month(pivot_df: pd.DataFrame, params: dict, target_start: pd.Timestamp):
    results = []
    cache = {}

    nm_start, nm_end, horizon = target_month_range(target_start)

    for cat in pivot_df.columns:
        s = pivot_df[cat].rename(cat)

        totals, fcst, active, method = sarimax_forecast_next_month(s, nm_start, horizon, params)

        if totals is None:
            totals, fcst, fb_method = fallback_forecast_next_month(s, nm_start, horizon)
            method = fb_method

        total, lo, hi = totals
        cache[cat] = (s, fcst)

        # "전월"은 예측 월의 직전 월 합계로 계산
        prev_month = (nm_start - pd.offsets.MonthBegin(1)).to_period("M")
        last_month_sum = float(s.loc[s.index.to_period("M") == prev_month].sum())

        results.append({
            "category": cat,
            "next_month": nm_start.strftime("%Y-%m"),
            "predicted": total,
            "lower": lo,
            "upper": hi,
            "last_month": last_month_sum,
            "active_days": int(active),
            "method": method,
        })

    pred_df = pd.DataFrame(results)
    if not pred_df.empty:
        pred_df = pred_df.sort_values("predicted", ascending=False).reset_index(drop=True)

    meta = {
        "train_end": train_end,
        "target_month_start": nm_start,
        "target_month_end": nm_end,
        "horizon_days": horizon,
    }
    return pred_df, cache, meta



# =========================
# 추천(기본) 예측 설정
# =========================
DEFAULT_NM_PARAMS = {
    "min_active_days": 10,
    "min_train_len": 30,
    "use_weekly_season": True,
    "use_smoothing": True,
    "use_log": True,
    "cap_q": 0.98,
    "drop_zero_days": False,
    "p": 1,
    "d": 1,
    "q": 1,
}

# nm_params 초기화/보정
if "nm_params" not in st.session_state or not isinstance(st.session_state.nm_params, dict):
    st.session_state.nm_params = DEFAULT_NM_PARAMS.copy()
else:
    for k, v in DEFAULT_NM_PARAMS.items():
        st.session_state.nm_params.setdefault(k, v)

# ✅ 위젯 상태도 "처음"에만 세팅 (위젯 생성 전에!)
st.session_state.setdefault("ui_min_active_days", int(st.session_state.nm_params["min_active_days"]))
st.session_state.setdefault("ui_min_train_len", int(st.session_state.nm_params["min_train_len"]))
st.session_state.setdefault("ui_use_weekly_season", bool(st.session_state.nm_params["use_weekly_season"]))
st.session_state.setdefault("ui_use_smoothing", bool(st.session_state.nm_params["use_smoothing"]))
st.session_state.setdefault("ui_use_log", bool(st.session_state.nm_params["use_log"]))

# ✅ 리셋 콜백 (여기서 ui_* 를 바꿔도 안전)
def reset_to_defaults():
    st.session_state.nm_params.update(DEFAULT_NM_PARAMS)

    st.session_state.ui_min_active_days = DEFAULT_NM_PARAMS["min_active_days"]
    st.session_state.ui_min_train_len = DEFAULT_NM_PARAMS["min_train_len"]
    st.session_state.ui_use_weekly_season = DEFAULT_NM_PARAMS["use_weekly_season"]
    st.session_state.ui_use_smoothing = DEFAULT_NM_PARAMS["use_smoothing"]
    st.session_state.ui_use_log = DEFAULT_NM_PARAMS["use_log"]

    st.session_state["just_reset_defaults"] = True  # (선택) 메시지용 플래그


# =========================
# 사이드바 UI
# =========================
st.sidebar.header("⚙️ 예측 설정")

categories_all = sorted(df["category"].unique().tolist())

selected_categories = st.sidebar.multiselect(
    "예측에 포함할 카테고리",
    categories_all,
    default=categories_all,
    help="선택한 카테고리만 다음 달 지출을 예측합니다.",
    key="ui_selected_categories",
)

with st.sidebar.expander("🔧 예측 방법 세부 설정 (선택)", expanded=False):
    st.info(
        "이 설정은 예측 방식을 직접 조정하고 싶은 분들을 위한 옵션입니다.\n\n"
        "설정을 변경하지 않아도, 기본값으로 대부분의 경우에 적합한 예측이 진행됩니다."
    )

    min_active_days = st.slider(
        "지출 기록이 최소 몇 번 이상 있어야 예측할까요?",
        1, 365,
        value=st.session_state.ui_min_active_days,
        key="ui_min_active_days",
    )

    min_train_len = st.slider(
        "예측 전에 최소 며칠치 기록을 볼까요?",
        7, 90,
        value=st.session_state.ui_min_train_len,
        key="ui_min_train_len",
    )

    use_weekly_season = st.checkbox(
        "요일별 소비 패턴 반영하기 (추천)",
        value=st.session_state.ui_use_weekly_season,
        key="ui_use_weekly_season",
    )

    use_smoothing = st.checkbox(
        "소비 그래프를 부드럽게 만들기 (추천)",
        value=st.session_state.ui_use_smoothing,
        key="ui_use_smoothing",
    )

    use_log = st.checkbox(
        "큰 지출로 인한 예측 흔들림 줄이기 (추천)",
        value=st.session_state.ui_use_log,
        key="ui_use_log",
    )

    st.markdown("---")

    st.button(
        "🔁 추천 설정으로 되돌리기",
        use_container_width=True,
        on_click=reset_to_defaults,
    )


if st.session_state.get("just_reset_defaults"):
    st.sidebar.success("추천 설정으로 되돌렸어요.")
    st.session_state["just_reset_defaults"] = False


# 적용 버튼
apply_btn = st.sidebar.button("🚀 설정을 적용해 다시 예측", key="btn_apply")
if apply_btn:
    st.session_state.nm_params.update({
        "min_active_days": int(st.session_state.ui_min_active_days),
        "min_train_len": int(st.session_state.ui_min_train_len),
        "use_weekly_season": bool(st.session_state.ui_use_weekly_season),
        "use_smoothing": bool(st.session_state.ui_use_smoothing),
        "use_log": bool(st.session_state.ui_use_log),
    })


current_params = st.session_state.nm_params.copy()


# ========================
# 선택 카테고리 필터
# ========================
min_active_days_now = int(current_params.get("min_active_days", 10))

# 선택된 카테고리 중, active_days(지출>0 일수)가 기준 이상인 것만 통과
active_days_by_cat = (
    df[df["category"].isin(selected_categories)]
    .groupby("category")["daily_amount"]
    .apply(lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) > 0).sum()))
    .to_dict()
)

eligible_categories = [c for c in selected_categories if active_days_by_cat.get(c, 0) >= min_active_days_now]
excluded_categories = [c for c in selected_categories if c not in eligible_categories]

if excluded_categories:
    st.sidebar.warning(
        f"현재 '최소 지출 기록({min_active_days_now}일)' 기준을 충족하지 못해 제외된 카테고리: "
        + ", ".join(excluded_categories)
    )

if not eligible_categories:
    st.warning(f"선택된 카테고리 중 '최소 지출 기록({min_active_days_now}일)' 기준을 충족하는 항목이 없습니다.")
    st.stop()


# ==================================================
# pivot 생성 (eligible_categories만) + ✅ 학습데이터는 train_end까지만
# ==================================================
df_train = df[
    (df["category"].isin(eligible_categories)) &
    (df["d"] <= train_end)
].copy()

pivot = (
    df_train
    .pivot_table(index="d", columns="category", values="daily_amount", aggfunc="sum")
    .fillna(0.0)
    .sort_index()
)

if pivot.empty or pivot.index.isna().all():
    st.warning("예측에 사용할 학습 데이터가 없습니다. (선택 월/데이터 기간을 확인하세요)")
    st.stop()

full_idx = pd.date_range(pivot.index.min(), pivot.index.max(), freq="D")
pivot = pivot.reindex(full_idx).fillna(0.0)
pivot.index.name = "ds"


cats_key = tuple(pivot.columns.tolist())

# 필터키(최소지출기록/선택카테고리)도 캐시 무효화 조건에 넣기
filter_key = (tuple(sorted(eligible_categories)), int(min_active_days_now), int(current_params.get("min_train_len", 30)),
              bool(current_params.get("use_weekly_season", True)), bool(current_params.get("use_smoothing", True)),
              bool(current_params.get("use_log", True)))

need_auto_run = False
if st.session_state.pred_df_nm.empty:
    need_auto_run = True
elif st.session_state.cats_key_nm != cats_key:
    need_auto_run = True
elif st.session_state.filter_key_nm != filter_key:
    need_auto_run = True
elif apply_btn:
    need_auto_run = True

if need_auto_run:
    with st.spinner("예측 계산 중..."):
        pred_df_nm, cache_nm, meta_nm = build_forecast_for_target_month(pivot, current_params, target_start)
        st.session_state.pred_df_nm = pred_df_nm
        st.session_state.pred_cache_nm = cache_nm
        st.session_state.pred_meta_nm = meta_nm
        st.session_state.cats_key_nm = cats_key
        st.session_state.filter_key_nm = filter_key


pred_df = st.session_state.pred_df_nm
cache = st.session_state.pred_cache_nm
meta = st.session_state.pred_meta_nm

# ✅ 최종 안전: pred_df도 eligible_categories만 남기기 (혹시 모를 상태 꼬임 방지)
if not pred_df.empty:
    pred_df = pred_df[pred_df["category"].isin(eligible_categories)].reset_index(drop=True)

if pred_df.empty:
    st.warning("예측 결과가 없습니다. (데이터/카테고리/기준 설정을 확인하세요)")
    st.stop()


# ==================================================
# 결과 표시
# ==================================================
nm_start = meta.get("target_month_start") or meta.get("next_month_start")
nm_label = nm_start.strftime("%Y-%m")


total_pred = float(pred_df["predicted"].sum())
total_last = float(pred_df["last_month"].sum())

c1, c2, c3 = st.columns(3)
c1.metric("예측 대상 월", nm_label)
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


def build_feedback_payload_from_pred(pred_df_in: pd.DataFrame, nm_label_in: str):
    df_send = pred_df_in.copy()
    for col in ["predicted", "lower", "upper", "last_month"]:
        if col in df_send.columns:
            df_send[col] = df_send[col].fillna(0).round(0).astype(int)

    total_pred_in = int(df_send["predicted"].sum()) if "predicted" in df_send.columns else 0
    total_last_in = int(df_send["last_month"].sum()) if "last_month" in df_send.columns else 0
    delta_in = total_pred_in - total_last_in

    payload = {
        "target_month": nm_label_in,
        "total_pred": total_pred_in,
        "total_last": total_last_in,
        "delta": delta_in,
        "currency": "KRW",
        "rows": df_send[["category", "predicted", "lower", "upper", "last_month", "active_days"]].to_dict(orient="records"),
    }
    return payload


def rule_based_feedback(view_df_in: pd.DataFrame, nm_label_in: str) -> str:
    df2 = view_df_in.copy()
    if "예측(원, 월합)" not in df2.columns or "전월(원, 월합)" not in df2.columns:
        return f"{nm_label_in} 예측 결과가 준비됐어요. (전월 비교 컬럼이 없어 간단 요약만 제공해요.)"

    df2["증감(원)"] = df2["예측(원, 월합)"] - df2["전월(원, 월합)"]
    denom = df2["전월(원, 월합)"].replace(0, np.nan)
    df2["증감(%)"] = (df2["증감(원)"] / denom * 100).round(1)

    total_pred_in = float(df2["예측(원, 월합)"].sum())
    total_last_in = float(df2["전월(원, 월합)"].sum())
    delta_in = total_pred_in - total_last_in
    delta_pct = (delta_in / total_last_in * 100) if total_last_in > 0 else np.nan

    top_up = df2.sort_values("증감(원)", ascending=False).head(3)
    top_down = df2.sort_values("증감(원)", ascending=True).head(3)

    lines = []
    if not np.isnan(delta_pct):
        lines.append(f"- **{nm_label_in} 총 지출 예측**: {total_pred_in:,.0f}원 (전월 대비 {delta_in:,.0f}원, {delta_pct:.1f}%)")
    else:
        lines.append(f"- **{nm_label_in} 총 지출 예측**: {total_pred_in:,.0f}원 (전월 대비 {delta_in:,.0f}원)")

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
# AI 피드백
# =========================
payload = build_feedback_payload_from_pred(view_df, nm_label)

payload_key = (
    nm_label,
    tuple(view_df.get("카테고리", pd.Series([], dtype=str)).astype(str).tolist()),
    float(view_df.get("예측(원, 월합)", pd.Series([0.0])).sum()),
    float(view_df.get("전월(원, 월합)", pd.Series([0.0])).sum()) if "전월(원, 월합)" in view_df.columns else 0.0,
    int(min_active_days_now),
)

if "llm_feedback" not in st.session_state:
    st.session_state.llm_feedback = ""
if "llm_feedback_key" not in st.session_state:
    st.session_state.llm_feedback_key = None

cbtn1, cbtn2 = st.columns([1, 1])
with cbtn1:
    generate_feedback = st.button("📝 AI 피드백 생성", type="primary", key="btn_feedback")
    st.caption("AI가 예측 데이터를 기반으로 피드백을 제공합니다.")
with cbtn2:
    regen = st.button("🔄 피드백 다시 생성", key="btn_feedback_regen")
    st.caption("현재 화면의 예측 숫자는 그대로 두고, AI가 설명과 조언만 새로 제공합니다.")

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










