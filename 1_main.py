# 추가 설치
import os, io, json
from datetime import datetime
from typing import Optional, Tuple, List

import pandas as pd
import streamlit as st
from PIL import Image
from google.cloud import vision
from openai import OpenAI

current_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.dirname(current_dir) 

key_path = os.path.join(current_dir, "ocr-service-482801-c279b4cf4d9f.json")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

from utils_state import init_state, get_db_engine
init_state()
#========================

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


# 카테고리 재분류 완료
DEFAULT_CATEGORIES = [
    "식비", "장보기", "교통/차량", "쇼핑/취미", 
    "생활/주거", "교육", "의료비", "기타"
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

    # [추가] 카테고리 리스트
    CATEGORIES = "식비, 장보기, 교통/차량, 쇼핑/취미, 생활/주거, 교육, 의료비, 기타"

    prompt = f"""
너는 “가계부용 영수증 파서”다.
입력은 (1) OCR 전체 텍스트(full_text)와 (2) OCR 라인별 bbox(layout_hint)다.
이미지는 제공되지 않는다. 대신 bbox를 근거로 “위치”를 활용해 구조화하라.

[추가: 카테고리 분류 엄격 규칙]
각 아이템의 'category'는 반드시 아래 8개 중 하나로만 매핑하라. 
영수증에 '정육', '식품', '공산'이라고 적혀 있어도 무시하고 이 규칙을 따라라:
1. 식비: 외식, 배달, 카페, 편의점 간식
2. 장보기: 마트에서 산 식재료(정육, 채소, 과일, 식품 등), 생필품
3. 교통/차량: 택시, 주유, 주차, 대중교통
4. 쇼핑/취미: 의류, 도서, 운동, 온라인 쇼핑
5. 생활/주거: 관리비, 통신비, 구독료
6. 교육: 교육, 자기계발
7. 의료비: 병원, 약국
8. 기타: 경조사, 분류 미정 항목

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
      "category": "string",  # [수정] 위 8개 카테고리 명칭 중 하나를 엄격히 선택
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

    # 적재 rows 만들기
    rows = []
    for it in (parsed.get("items") or []):
        rows.append({
            "date_time": parsed.get("transaction_datetime"),
            "merchant": parsed.get("merchant_name"),
            "item": it.get("name") or "",
            "category": it.get("category") or "기타", # [수정] "기타" 하드코딩 대신 AI 결과 반영
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

    st.subheader("🚀 데이터 적재")
    c1, c2 = st.columns([1, 1])
    
    with c1:
        # [수정] 버튼 정의
        do_load = st.button("✅ 데이터 적재하기", use_container_width=True, type="primary")

    # [수정] 적재 실행 로직 (edited_df 대신 파싱 결과인 rows를 직접 사용)
    if do_load:
        if rows: # [추가] rows 데이터가 있을 때만 실행
            df_add = pd.DataFrame([{
                "user_id": 1,
                "transaction_date": r["date_time"],
                "merchant_name": r["merchant"],
                "description": r["item"],
                "category_name": r["category"],
                "amount": int(r["amount"])
            } for r in rows]) # [수정] edited_df 대신 위에서 만든 rows를 직접 참조
            
            try:
                engine = get_db_engine()
                df_add.to_sql("Transactions", con=engine, if_exists="append", index=False)
                
                # [수정] 세션 업데이트: 읽은 목록 누락 현상 있음... 특정 영수증에 대해서 일어나기 때문에 추가 수정 필요
                st.session_state.spend_df = pd.read_sql("SELECT * FROM Transactions", engine)
                
                st.success(f"성공! AWS RDS에 {len(df_add)}건 적재 완료")
                st.rerun() 

            except Exception as e:
                st.error(f"❌ DB 연동 오류: {e}")
        else:
            st.warning("적재할 데이터가 없습니다. 먼저 영수증을 파싱해 주세요.")