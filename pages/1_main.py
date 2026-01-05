import os, io, json
from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st
from PIL import Image
from google.cloud import vision
from openai import OpenAI

from utils_state import init_state
init_state()

# -------------------------
# Session vars
# -------------------------
if "last_parsed" not in st.session_state:
    st.session_state["last_parsed"] = None

if "last_ocr_text" not in st.session_state:
    st.session_state["last_ocr_text"] = None

# =========================
# Constants
# =========================
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# 카테고리 재 분류 필요
DEFAULT_CATEGORIES = [
    "식비", "카페·간식", "교통·차량", "주거·통신", "생활용품",
    "쇼핑·의류", "의료·건강", "교육·자기계발", "문화·여가", "기타"
]

# =========================
# Helpers
# =========================
def normalize_month(dt_str: Optional[str]) -> str:
    if not dt_str:
        return "Unknown"
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(dt_str).strip(), fmt).strftime("%Y-%m")
        except Exception:
            pass
    return "Unknown"


def add_rows_to_spend_df(df_new: pd.DataFrame):
    if df_new is None or df_new.empty:
        return

    base = int(st.session_state.get("row_id_seq", 0))
    df_new = df_new.copy()

    if "row_id" not in df_new.columns:
        df_new.insert(0, "row_id", range(base + 1, base + 1 + len(df_new)))
        st.session_state["row_id_seq"] = base + len(df_new)

    for col in ["row_id", "date_time", "merchant", "item", "category", "amount"]:
        if col not in df_new.columns:
            df_new[col] = None

    df_new["amount"] = pd.to_numeric(df_new["amount"], errors="coerce").fillna(0.0)
    df_new["category"] = df_new["category"].fillna("기타")

    st.session_state.spend_df = pd.concat(
        [st.session_state.spend_df, df_new[["row_id","date_time","merchant","item","category","amount"]]],
        ignore_index=True
    )

# =========================
# Clients
# =========================
@st.cache_resource
def get_vision_client():
    return vision.ImageAnnotatorClient()

@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# OCR
# =========================
def run_vision_ocr(img_bytes: bytes) -> str:
    client = get_vision_client()
    image = vision.Image(content=img_bytes)
    res = client.document_text_detection(image=image)
    return res.full_text_annotation.text if res.full_text_annotation else ""

# =========================
# LLM (수정 필요(좀 더 업그레이드 된 것으로))
# =========================
def parse_receipt_llm(text: str) -> dict:
    client = get_openai_client()
    prompt = f"""
너는 가계부용 영수증 파서다.
아래 OCR 텍스트를 JSON으로 구조화해라.

[출력 규칙]
- 코드펜스 없이 JSON만 출력
- 가능하면 아래 스키마를 따라라

[스키마]
{{
  "merchant_name": "string | null",
  "transaction_datetime": "YYYY-MM-DD HH:MM | null",
  "items": [
    {{
      "name": "string",
      "net_amount": "number"
    }}
  ],
  "totals": {{
    "amount_paid": "number | null"
  }}
}}

[OCR TEXT]
\"\"\"{text}\"\"\"
""".strip()

    r = client.responses.create(
        model=MODEL,
        input=prompt,
        max_output_tokens=1500
    )

    raw = (r.output_text or "").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM output is not valid JSON:\n{raw}")

    return json.loads(raw[start:end+1])

# =========================
# UI  (추가 수정 필요)
# =========================
st.title("🧾 영수증 입력")
st.caption("종이 영수증 이미지를 업로드하면 자동으로 지출 데이터로 변환합니다.")

img = st.file_uploader("영수증 이미지 업로드", type=["jpg", "png", "jpeg"], key="receipt_img")

if img:
    img_bytes = img.read()
    st.image(img_bytes, width=350)

    colA, colB = st.columns([1, 1])
    with colA:
        run_parse = st.button("🚀 OCR + 파싱 실행", type="primary", use_container_width=True)
    with colB:
        clear = st.button("🧹 파싱 결과 지우기", use_container_width=True)

    if clear:
        st.session_state["last_parsed"] = None
        st.session_state["last_ocr_text"] = None
        st.success("초기화 완료")
        st.rerun()

    if run_parse:
        with st.spinner("OCR 처리 중..."):
            ocr_text = run_vision_ocr(img_bytes)
        st.session_state["last_ocr_text"] = ocr_text

        with st.spinner("LLM 파싱 중..."):
            parsed = parse_receipt_llm(ocr_text)

        st.session_state["last_parsed"] = parsed
        st.success("파싱 완료! 아래에서 적재할 수 있어요.")

# =========================
# 파싱 결과 
# =========================
parsed = st.session_state.get("last_parsed")

st.markdown("---")
st.subheader("파싱 결과")

if parsed is None:
    st.info("이미지를 업로드한 뒤 **OCR + 파싱 실행**을 눌러주세요.")
else:
    st.json(parsed)

    with st.expander("OCR 텍스트 보기(디버그)", expanded=False):
        st.text((st.session_state.get("last_ocr_text") or "")[:20000])

    # 적재 rows 만들기 (items가 없으면 totals로 1건이라도 적재)
    rows = []
    for it in (parsed.get("items") or []):
        rows.append({
            "date_time": parsed.get("transaction_datetime"),
            "merchant": parsed.get("merchant_name"),
            "item": it.get("name") or "",
            "category": "기타",
            "amount": float(it.get("net_amount", 0.0) or 0.0),
        })

    if not rows:
        amt = (parsed.get("totals") or {}).get("amount_paid")
        if amt is not None:
            rows = [{
                "date_time": parsed.get("transaction_datetime"),
                "merchant": parsed.get("merchant_name"),
                "item": "(총 결제금액)",
                "category": "기타",
                "amount": float(amt),
            }]

    st.subheader("✅ 지출 데이터로 적재")
    c1, c2 = st.columns([1, 1])
    with c1:
        do_load = st.button("✅ 적재하기", use_container_width=True, disabled=(len(rows) == 0))


    if do_load:
        df_add = pd.DataFrame(rows)
        add_rows_to_spend_df(df_add)

        #  자동 월 이동 (추출된 날짜 데이터를 통해서)
        uploaded_month = normalize_month(parsed.get("transaction_datetime"))
        if uploaded_month != "Unknown":
            st.session_state["pending_month"] = uploaded_month
            st.session_state["month_input"] = uploaded_month

        st.success(f"적재 완료! 현재 spend_df rows: {len(st.session_state.spend_df)}")
        st.dataframe(st.session_state.spend_df.tail(20), width="stretch")


