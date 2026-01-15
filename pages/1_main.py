import os, io, json
from datetime import datetime
from typing import Optional, Tuple
import numpy as np
import cv2

import pandas as pd
import streamlit as st
from PIL import Image
from google.cloud import vision
from openai import OpenAI
from sqlalchemy import text
from utils_state import init_state, get_db_engine

import streamlit as st

st.set_page_config(
    page_title="영수증 입력",
    page_icon="🧾",
)

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

# =========================
# Helpers
# =========================
def to_db_datetime(dt_str: Optional[str]) -> Optional[str]:
    """LLM이 준 날짜를 DB용 'YYYY-MM-DD HH:MM:SS'로 정규화"""
    if not dt_str:
        return None
    s = str(dt_str).strip()

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
            d = datetime.strptime(s, fmt)
            return d.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

    # 파싱 실패 시 원문 그대로 반환
    return s
def normalize_signature(s: str) -> str:
    """중복 체크용 description 서명 정규화"""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = " ".join(s.split()).strip()
    return s


# ===========================================
# 이미지 전처리 (노이즈 감소 + 대비 강화 + 샤프닝)
# ===========================================
def preprocess_receipt_bytes(img_bytes: bytes) -> bytes:
    """
    전처리: 노이즈 감소 + 대비 강화(CLAHE) + 샤프닝
    반환: Google Vision에 넣을 JPEG bytes
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 1) 노이즈 감소 (배경은 부드럽게, 글자 획 유지)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # 2) 대비 강화 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 3) 샤프닝
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    gray = cv2.filter2D(gray, -1, kernel)

    ok, buf = cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return buf.tobytes() if ok else img_bytes


# =====================
# 중복 체크
# =====================
def add_rows_to_spend_df(df_new: pd.DataFrame):
    if df_new is None or df_new.empty:
        return

    #------------------------------------------
    # st.session_state.spend_df 에 이미 등록된 항목은 df_new 에서 제외
    df_new = check_duplicate(df_new)

    if df_new is None or df_new.empty:
        st.toast(f"모든 항목이 중복되어 적재할 항목이 없습니다.", icon="⚠️")
        return
    #------------------------------------------
    
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

# 중복 체크를 위한 정규화 함수
def norm_text(val):
    if pd.isna(val):
        return ""
    
    # 공백 제거
    return " ".join(str(val).split()).strip()

# 영수증 등록시 중복 체크
def check_duplicate(df_new):
    df_new = df_new.copy()

    # merchant 비교용 정규화만 유지
    df_new["_merchant_norm"] = df_new["merchant"].apply(norm_text)

    for _, row in st.session_state.spend_df.iterrows():
        row_dt = row.get("date_time")
        row_amount = float(row.get("amount") or 0.0)
        row_merchant_norm = norm_text(row.get("merchant"))

        mask = (
            (df_new["date_time"] == row_dt) &
            (df_new["_merchant_norm"] == row_merchant_norm) &
            (pd.to_numeric(df_new["amount"], errors="coerce").fillna(0.0) == row_amount)
        )

        df_new = df_new.loc[~mask]

    return df_new.drop(columns=["_merchant_norm"], errors="ignore")


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
# OCR + Layout extraction
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
# LLM (Transactions 1행 형태)
# =========================
def parse_receipt_llm_to_transactions(full_text: str, lines: list, W: int, H: int) -> dict:
    client_llm = get_openai_client()
    layout_hint = {"image_size": {"width": W, "height": H}, "lines": lines}
    allowed_categories = " | ".join(DEFAULT_CATEGORIES)

    prompt = f"""
너는 “가계부용 영수증 파서”다.
입력은 (1) OCR 전체 텍스트(full_text)와 (2) OCR 라인별 bbox(layout_hint)다.
이미지는 제공되지 않는다. 대신 bbox를 근거로 “위치(상단/중단/하단)”를 활용해 구조화하라.

# 목표
- 아래 DB 테이블 Transactions 컬럼에 바로 INSERT 가능한 형태로 1건(JSON 1개)만 출력한다.
- 품목(item) 단위로 여러 줄을 만들지 말고, 영수증 1장 = 거래 1건으로 요약(description)에 압축한다.

# DB Transactions 컬럼
- user_id: 숫자(앱에서 채울 예정이므로 여기서는 null 고정)
- category_name: 상위 카테고리명(아래 허용 목록 중 하나)
- transaction_date: 거래일시(가능하면 YYYY-MM-DD HH:MM:SS)
- merchant_name: 상호명
- amount: 최종 결제 금액(정수)
- description: 대표 품목/할인/결제수단 요약(한 줄)

# 위치 기반 규칙 (매우 중요)
- 상단(상호/사업자/주소/전화): merchant 후보가 많다. 가장 그럴듯한 상호명을 merchant_name으로 선택.
- 중단(품목 영역): 품목명/수량/금액이 반복되는 영역. 여기서 대표 품목들을 description에 요약.
- 하단(합계/총액/결제): 최종 결제금액(amount)과 거래일시(transaction_date)가 주로 나온다.
- bbox 목록은 위에서 아래로 읽힌 순서대로 처리한다.

# 금액 규칙 (매우 중요)
- amount는 “사용자가 실제로 결제한 최종 금액(실결제/받은금액/결제금액/합계/총 결제금액/승인금액)”을 우선한다.
- 할인/쿠폰/행사/즉시할인 등은 amount가 아니라 계산 근거일 수 있다. (amount는 최종 결제 기준)
- 음수 금액(할인) 라인은 품목이 아니라 할인이다.
- 최종 결제 금액 후보가 여러 개면, “하단”에 있고, 키워드(결제금액/총액/합계/받은금액/승인금액/결제대상금액)에 가장 가까운 값을 채택한다.
- 통화단위(원) 표기나 콤마는 제거하고 정수로 만든다.

# 날짜/시간 규칙
- transaction_date는 가능하면 "YYYY-MM-DD HH:MM:SS"로 출력한다.
- 초가 없으면 ":00"을 붙인다.
- 날짜만 있으면 시간은 "00:00:00"으로 둔다.
- 완전히 못 찾으면 null.

# 카테고리 분류 규칙 — 반드시 준수
- category_name은 다음 중 하나만: {allowed_categories}
- "이마트", "홈플러스", "롯데마트", "코스트코", "GS더프레시", "노브랜드" 등 대형마트·식료품 중심 매장은 **반드시 "장보기"**
- "쿠팡", "네이버쇼핑", "11번가", "G마켓", "옥션", "SSG", "무신사" 등 온라인 쇼핑몰은 **반드시 "쇼핑/취미"**
- 카페/베이커리/편의점은 **"식비"**
- 약국/병원/의원은 **"의료비"**
- 위 규칙으로 판단 불가한 경우만 **"기타"**
- merchant 기준 규칙이 품목 추정보다 우선이다.

# description은 "중복검사용 표준 서명"으로 작성한다 (매우 중요)
- 절대 자연어 문장으로 쓰지 말고, 아래 포맷을 강제한다.
- 토큰 구분자는 반드시 " " 만 사용한다.
- 품목은 OCR lines를 기준으로 위→아래 순서로 최대 6개만 뽑는다.
- 품목 표기는 반드시 "{{name}} {{amount}}x{{qty}}" 로 고정한다. (qty 없으면 x1)
- 할인/쿠폰은 반드시 "DISCOUNT {{abs_amount}}" 로 표기한다.
- 결제수단은 반드시 "PAY=card|PAY=cash|PAY=easy" 중 하나로 표기(모르면 PAY=unknown)
- 마지막에 반드시 "TOTAL={{amount}}" 와 "DT={{transaction_date}}" 를 포함한다.
- 공백, 괄호, 쉼표, "원" 같은 단위는 절대 넣지 않는다.


# 출력 스키마 (반드시 이 JSON 1개만 출력, 추가 설명 금지)
{{
  "user_id": null,
  "category_name": "{allowed_categories} 중 하나 또는 null",
  "transaction_date": "YYYY-MM-DD HH:MM:SS | null",
  "merchant_name": "string | null",
  "amount": "number | null",
  "description": "string | null",
  "confidence": "number (0~1)",
  "notes": "string | null"
}}

[OCR 텍스트(full_text)]
\"\"\"{full_text}\"\"\"

[레이아웃 힌트(layout_hint)]
{json.dumps(layout_hint, ensure_ascii=False)}
""".strip()

    resp = client_llm.responses.create(
        model=MODEL,
        max_output_tokens=1600,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
    )

    text_out = (resp.output_text or "").strip()
    start = text_out.find("{")
    end = text_out.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM output is not valid JSON:\n{text_out}")

    return json.loads(text_out[start:end + 1])


# =========================
# UI
# =========================
st.title("🧾 영수증 입력")

img = st.file_uploader("영수증 이미지 업로드", type=["jpg", "png", "jpeg"], key="receipt_img")

if img:
    img_bytes = img.read()
    st.image(img_bytes, width=350)

    colA, colB = st.columns([1, 1])
    with colA:
        run_parse = st.button("🔍 영수증 인식하기", type="primary", use_container_width=True)
    with colB:
        clear = st.button("🧹 인식 결과 지우기", use_container_width=True)

    if clear:
        st.session_state["last_parsed"] = None
        st.session_state["last_ocr_text"] = None
        st.session_state["last_lines"] = None
        st.session_state["last_img_size"] = None
        st.success("초기화 완료")
        st.rerun()



    if run_parse:
        # 전처리 적용 (노이즈 감소 + 대비 강화 + 샤프닝)
        img_bytes_pp = preprocess_receipt_bytes(img_bytes)

        with st.spinner("OCR + 레이아웃 추출 중..."):
            full_text, lines, W, H = ocr_fulltext_and_lines_with_bbox_from_bytes(img_bytes_pp)

        st.session_state["last_ocr_text"] = full_text
        st.session_state["last_lines"] = lines
        st.session_state["last_img_size"] = (W, H)

        with st.spinner("LLM 파싱 중...(Transactions 1행 형태)"):
            parsed = parse_receipt_llm_to_transactions(full_text=full_text, lines=lines, W=W, H=H)

        st.session_state["last_parsed"] = parsed
        st.success("영수증 인식 완료! 이제 영수증을 가계부에 저장할 수 있어요!")


# =========================
# 파싱 결과
# =========================
parsed = st.session_state.get("last_parsed")

st.markdown("---")
st.subheader("영수증 인식 결과")



if parsed is None:
    st.info("이미지를 업로드한 뒤 **OCR + 파싱 실행**을 눌러주세요.")
else:
    # ✅ 일반 사용자용: 핵심만
    st.success("✅ 영수증이 인식되었습니다. 아래 내용을 확인한 뒤 저장하세요.")


    # 일반 사용자에게 보여줄 핵심 요약(필요한 것만)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("상호", parsed.get("merchant_name") or "-")
        st.metric("카테고리", (parsed.get("category_name") or "기타"))
    with col2:
        st.metric("금액", int(float(parsed.get("amount") or 0)))
        st.metric("거래일시", to_db_datetime(parsed.get("transaction_date")) or "-")

    st.caption(parsed.get("description") or "")


if st.session_state.get("save_success_msg"):
    st.success(st.session_state["save_success_msg"])
    st.session_state["save_success_msg"] = None

# -------------------------
# Parsed -> DB row (안전 처리)
# -------------------------
# ✅ parsed가 None/비dict이면 dict로 치환
if not isinstance(parsed, dict):
    parsed = {}

cat = (parsed.get("category_name") or "기타").strip()
if cat not in DEFAULT_CATEGORIES:
    cat = "기타"

date_time_db = to_db_datetime(parsed.get("transaction_date"))

amt = parsed.get("amount")
amt = float(amt) if amt is not None else 0.0

# ✅ 실제 저장되는 amount 로직을 함수처럼 고정 (미리보기/저장 동일)
amt_int = int(amt) if amt == int(amt) else int(round(amt))

st.markdown("---")
st.subheader("💾 영수증 저장")
st.info(
    "위에서 영수증 인식 결과를 확인하셨다면, "
    "아래 버튼을 눌러 가계부에 저장해 주세요.\n\n"
    "인식 결과에 오류가 있더라도 이후에 수정할 수 있습니다."
)

do_load_db = st.button("💾 영수증을 가계부에 저장하기", use_container_width=True, type="primary")

if do_load_db:
    engine = get_db_engine()

    try:
        with st.spinner("저장 중..."):
            with engine.begin() as conn:
                cat_df = pd.read_sql("SELECT id, category_name FROM Category", con=conn)
                cat_map = dict(zip(cat_df["category_name"], cat_df["id"]))

                target_id = cat_map.get(cat)
                if target_id is None:
                    target_id = list(cat_map.values())[0] if cat_map else 1

                # -----------------------------
                # 중복 체크 +(표준 문자 서명)
                # -----------------------------
                query = """
                SELECT COUNT(*) FROM Transactions
                WHERE user_id = :user_id
                AND type = 'E'
                AND amount = :amount
                AND DATE_FORMAT(transaction_date, '%Y-%m-%d %H:%i')
                    = DATE_FORMAT(:transaction_date, '%Y-%m-%d %H:%i')
                AND description = :description
                """
                param = {
                    "user_id": 1,
                    "transaction_date": date_time_db,
                    "amount": amt_int,
                    "description": normalize_signature(parsed.get("description") or "(영수증)")
                }


                result = conn.execute(text(query), param)
                count = result.scalar()

                if count > 0:
                    st.warning("⚠️ 동일한 영수증이 이미 가계부에 등록되어 있습니다.")
                    st.stop()
                # -----------------------------

                df_add_db = pd.DataFrame([{
                    "user_id": 1,
                    "transaction_date": date_time_db,
                    "merchant_name": parsed.get("merchant_name"),
                    "description": parsed.get("description") or "(영수증)",
                    "amount": amt_int,
                    "category_id": int(target_id),
                    "type": "E",
                }])

                df_add_db.to_sql("Transactions", con=conn, if_exists="append", index=False)

        st.session_state["save_success_msg"] = f"✅ 저장 완료! ({cat}, ID:{target_id})"
        st.success(st.session_state["save_success_msg"])
        st.rerun()

    except Exception as e:
        st.error(f"❌ DB 적재 오류: {e}")


st.markdown(
    """
    ---
    #### 🧪 고급 정보 (선택 사항)
    아래 내용은 **개발자·관리자·검증용** 정보입니다.  
    일반적인 사용에는 **확인하지 않아도 무방**합니다.
    """
)

with st.expander("🧪 고급 정보(파싱 JSON / OCR / bbox) 보기", expanded=False):

    # 1️⃣ 파싱 결과 JSON
    st.subheader("파싱 결과 JSON")
    st.json(parsed)

    st.markdown("---")

    # 2️⃣ DB(Transactions)로 저장될 값 미리보기
    # ❌ parsed를 덮어쓰지 말고 별도 변수로!
    parsed_dbg = st.session_state.get("last_parsed")
    if not isinstance(parsed_dbg, dict):
        parsed_dbg = {}

    cat_dbg = (parsed_dbg.get("category_name") or "기타").strip()
    if cat_dbg not in DEFAULT_CATEGORIES:
        cat_dbg = "기타"

    date_time_dbg = to_db_datetime(parsed_dbg.get("transaction_date"))

    amt_dbg = parsed_dbg.get("amount")
    amt_dbg = float(amt_dbg) if amt_dbg is not None else 0.0
    amt_dbg_int = int(amt_dbg) if amt_dbg == int(amt_dbg) else int(round(amt_dbg))

    st.subheader("🧩 DB에 저장될 값 미리보기")
    st.code(
        json.dumps(
            {
                "user_id": 1,
                "category_name": cat_dbg,
                "transaction_date": date_time_dbg,
                "merchant_name": parsed_dbg.get("merchant_name"),
                "amount": amt_dbg_int,
                "description": parsed_dbg.get("description") or "(영수증)",
            },
            ensure_ascii=False,
            indent=2,
        ),
        language="json",
    )



    st.markdown("---")

    # 3️⃣ OCR 원문 텍스트
    with st.expander("OCR 텍스트 보기(디버그)", expanded=False):
        st.text((st.session_state.get("last_ocr_text") or "")[:20000])

    # 4️⃣ 레이아웃 라인(bbox) 정보
    with st.expander("레이아웃 라인 보기(디버그)", expanded=False):
        lines = st.session_state.get("last_lines") or []
        size = st.session_state.get("last_img_size")
        st.caption(f"lines={len(lines)}, image_size={size}")
        st.json(lines[:50])


