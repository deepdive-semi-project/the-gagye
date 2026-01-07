import pandas as pd
import sqlite3

def load_data():
    # 1. 파일 읽기 (인코딩 주의)
    try:
        df = pd.read_csv('expenses_20251224.csv', encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv('expenses_20251224.csv', encoding='cp949')

    # 2. 컬럼명 공백 제거 (이제 ' income ' -> 'income'이 됩니다)
    df.columns = df.columns.str.strip()
    
    # 3. 금액 데이터 숫자 변환 (콤마 제거)
    # 이제 공백 없이 'income', 'expense'로 접근합니다.
    df['income'] = df['income'].str.replace(',', '').fillna(0).astype(float)
    df['expense'] = df['expense'].str.replace(',', '').fillna(0).astype(float)

    # 4. DB 연결 및 저장
    conn = sqlite3.connect('smart_ledger.db')
    
    for _, row in df.iterrows():
        # 수입/지출 중 값이 있는 것을 금액으로 선택
        amount = row['income'] if row['type'] == 'income' else row['expense']
        
        conn.execute('''
            INSERT INTO expenses (date, description, amount, category, source_type, group_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (row['date'], row['description'], int(amount), '미분류', 'CSV', 1))
    
    conn.commit()
    conn.close()
    print("성공!")

load_data()