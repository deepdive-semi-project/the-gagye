import easyocr
import re
import os
import cv2

def advanced_parsing(img_path):
    # 1. OCR 엔진 초기화 및 실행
    reader = easyocr.Reader(['ko', 'en'])
    result = reader.readtext(img_path)
    
    # AI가 읽은 순서대로 텍스트만 리스트로 만듭니다.
    texts = [res[1] for res in result]
    
    # --- [데이터 파싱(선별) 시작] ---
    
    parsed_result = {
        '상호명': '미확인',
        '날짜': '0000-00-00',
        '품목': [],
        '총금액': 0
    }

    # 1. 상호명 추출 (리스트의 가장 앞부분에서 찾기)
    if texts:
        parsed_result['상호명'] = texts[0]

    # 2. 날짜 추출 (정규표현식 사용)
    for t in texts:
        date_match = re.search(r'\d{4}[-/.]\d{2}[-/.]\d{2}', t)
        if date_match:
            parsed_result['날짜'] = date_match.group()
            break

    # 3. 품목 및 총금액 추출 (키워드 기반 추적)
    item_start = False
    for i, t in enumerate(texts):
        # 품목 시작 지점 찾기 (이마트 영수증 기준)
        if '상품명' in t or '단가' in t:
            item_start = True
            continue
        
        # 품목 끝 지점 및 총금액 찾기
        if '대상금액' in t or '합계' in t or '결제금액' in t:
            item_start = False
            # 총금액은 키워드 근처의 숫자를 가져옴
            for j in range(i, min(i+3, len(texts))):
                clean_val = re.sub(r'[^0-9]', '', texts[j])
                if clean_val and len(clean_val) >= 4:
                    parsed_result['총금액'] = int(clean_val)
                    break
            continue

        # 품목 수집 (상품명과 합계 사이의 글자들)
        if item_start:
            # 숫자가 섞여 있고 너무 짧지 않은 문장을 품목으로 간주
            if len(t) > 2 and any(char.isdigit() for char in t):
                parsed_result['품목'].append(t)

    return parsed_result

# 실행 및 결과 확인
img_file = 'receipt.jpg' # 영수증 파일명
if os.path.exists(img_file):
    print(f"--- '{img_file}' 파싱 시작 ---")
    data = advanced_parsing(img_file)
    
    print("\n[최종 파싱 결과]")
    print(f"🏢 상호명: {data['상호명']}")
    print(f"📅 날짜: {data['날짜']}")
    print(f"💰 총금액: {data['총금액']:,}원")
    print(f"🛒 품목 리스트:")
    for item in data['품목']:
        print(f"   - {item}")
else:
    print("파일이 없습니다!")