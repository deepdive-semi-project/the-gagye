import os, io, json
from datetime import datetime
from typing import Optional, Tuple, List

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

# layout 기반 디버그용
if "last_lines" not in st.session_state:
    st.session_state["last_lines"] = None
if "last_img_size" not in st.session_state:
    st.session_state["last_img_size"] = None

# =========================
# Constants
# =========================
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


# 카테고리 재분류 필요
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
    # OPENAI_API_KEY 환경변수 필요
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# OCR + Layout extraction (업그레이드 버전)
# =========================
def _bbox_to_xyxy_norm(bounding_poly, W, H):
    xs = [v.x for v in bounding_poly.vertices]
    ys = [v.y for v in bounding_poly.vertices]
    x0, x1 = max(0, min(xs)), min(W, max(xs))
    y0, y1 = max(0, min(ys)), min(H, max(ys))
    return {"x0": round(x0 / W, 6), "y0": round(y0 / H, 6), "x1": round(x1 / W, 6), "y1": round(y1 / H, 6)}

def _merge_boxes(boxes):
    x0 = min(b["x0"] for b in boxes)
    y0 = min(b["y0"] for b in boxes)
    x1 = max(b["x1"] for b in boxes)
    y1 = max(b["y1"] for b in boxes)
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}

def ocr_fulltext_and_lines_with_bbox_from_bytes(img_bytes: bytes) -> Tuple[str, list, int, int]:
    """
    Streamlit 업로드 bytes -> (full_text, lines, W, H)
    lines: [{"text": "...", "bbox": {"x0","y0","x1","y1"}} ...]
    """
    client_vision = get_vision_client()

    image = vision.Image(content=img_bytes)
    response = client_vision.document_text_detection(image=image)

    if response.error.message:
        raise RuntimeError(response.error.message)

    annotation = response.full_text_annotation
    full_text = annotation.text if annotation and annotation.text else ""

    img = Image.open(io.BytesIO(img_bytes))
    W, H = img.size

    lines = []
    cur_words, cur_boxes = [], []

    if annotation and annotation.pages:
        for page in annotation.pages:
            for block in page.blocks:
                for para in block.paragraphs:
                    for word in para.words:
                        word_text = "".join(sym.text for sym in word.symbols)
                        word_box = _bbox_to_xyxy_norm(word.bounding_box, W, H)

                        cur_words.append(word_text)
                        cur_boxes.append(word_box)

                        last_sym = word.symbols[-1]
                        brk = (
                            last_sym.property.detected_break.type
                            if last_sym.property and last_sym.property.detected_break
                            else None
                        )

                        if brk in (
                            vision.TextAnnotation.DetectedBreak.BreakType.LINE_BREAK,
                            vision.TextAnnotation.DetectedBreak.BreakType.EOL_SURE_SPACE,
                        ):
                            line_text = " ".join(cur_words).strip()
                            if line_text:
                                lines.append({"text": line_text, "bbox": _merge_boxes(cur_boxes)})
                            cur_words, cur_boxes = [], []

    if cur_words:
        line_text = " ".join(cur_words).strip()
        if line_text:
            lines.append({"text": line_text, "bbox": _merge_boxes(cur_boxes)})

    return full_text, lines, W, H

# =========================
# LLM (layout_hint 기반 업그레이드 버전)
# =========================
def parse_receipt_llm_with_layout(full_text: str, lines: list, W: int, H: int) -> dict:
    client_llm = get_openai_client()
    layout_hint = {"image_size": {"width": W, "height": H}, "lines": lines}

    prompt = f"""
너는 “가계부용 영수증 파서”다.
입력은 (1) OCR 전체 텍스트(full_text)와 (2) OCR 라인별 bbox(layout_hint)다.
이미지는 제공되지 않는다. 대신 bbox를 근거로 “위치”를 활용해 구조화하라.

[중요]
- bbox 목록은 위에서 아래로 읽힌 순서대로 처리한다.
- 음수 금액(할인)은 새로운 품목이 아니며, 할인 라인은 바로 이전에 처리된 품목 1개에만 귀속한다.
- item_discount와 order_discount는 항상 0 이하(음수 또는 0)로 출력한다.
- 결제단(하단)에 있는 결제할인/총결제금액은 totals로 처리한다.
- 출력은 코드펜스 없이 JSON만 출력한다.

[출력 스키마]
{{
  "merchant_name": "string | null",
  "transaction_datetime": "YYYY-MM-DD HH:MM | null",
  "items": [
    {{
      "name": "string",
      "qty": "number | null",
      "unit_price": "number | null",
      "gross_amount": "number | null",
      "item_discount": "number",
      "net_amount": "number",
      "memo": "string | null"
    }}
  ],
  "totals": {{
    "total_before_order_discount": "number | null",
    "order_discount": "number | null",
    "unassigned_item_discount": "number | null",
    "amount_paid": "number | null"
  }},
  "confidence": "number",
  "notes": "string | null"
}}

[OCR 텍스트(full_text)]
\"\"\"{full_text}\"\"\"

[레이아웃 힌트(layout_hint)]
{json.dumps(layout_hint, ensure_ascii=False)}
""".strip()

    resp = client_llm.responses.create(
        model=MODEL,
        max_output_tokens=2500,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
    )

    text_out = (resp.output_text or "").strip()
    start = text_out.find("{")
    end = text_out.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM output is not valid JSON:\n{text_out}")

    return json.loads(text_out[start:end + 1])

# =========================
# UI (수정 필요)
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
        st.session_state["last_lines"] = None
        st.session_state["last_img_size"] = None
        st.success("초기화 완료")
        st.rerun()

    if run_parse:
        with st.spinner("OCR + 레이아웃 추출 중..."):
            full_text, lines, W, H = ocr_fulltext_and_lines_with_bbox_from_bytes(img_bytes)

        st.session_state["last_ocr_text"] = full_text
        st.session_state["last_lines"] = lines
        st.session_state["last_img_size"] = (W, H)

        with st.spinner("LLM 파싱 중..."):
        
            parsed = parse_receipt_llm_with_layout(full_text=full_text, lines=lines, W=W, H=H)

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

    with st.expander("레이아웃 라인 보기(디버그)", expanded=False):
        lines = st.session_state.get("last_lines") or []
        size = st.session_state.get("last_img_size")
        st.caption(f"lines={len(lines)}, image_size={size}")
        st.json(lines[:50]) 

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

        uploaded_month = normalize_month(parsed.get("transaction_datetime"))
        if uploaded_month != "Unknown":
            st.session_state["pending_month"] = uploaded_month
            st.session_state["month_input"] = uploaded_month

        st.success(f"적재 완료! 현재 spend_df rows: {len(st.session_state.spend_df)}")
        st.dataframe(st.session_state.spend_df.tail(20), width="stretch")


