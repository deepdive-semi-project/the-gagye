import pandas as pd
import sqlite3
import os

def restore_project():
    # 1. DB 연결 및 테이블 생성
    conn = sqlite3.connect('smart_ledger.db')
    cursor = conn.cursor()
    
    print("1. DB 테이블 생성 중...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            description TEXT,
            amount INTEGER,
            category TEXT,
            source_type TEXT,
            group_id INTEGER
        )
    ''')
    
    # 2. CSV 파일 읽기
    file_name = 'expenses_20251224.csv'
    if not os.path.exists(file_name):
        print(f"오류: {file_name} 파일이 폴더에 없습니다!")
        return

    try:
        df = pd.read_csv(file_name, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_name, encoding='cp949')

    # 3. 데이터 정제 (공백 제거 및 숫자 변환)
    print("2. 데이터 정제 및 DB 로드 중...")
    df.columns = df.columns.str.strip()
    
    # 금액 데이터에서 콤마 제거 후 숫자로 변환
    df['income'] = df['income'].astype(str).str.replace(',', '').replace('nan', '0').astype(float)
    df['expense'] = df['expense'].astype(str).str.replace(',', '').replace('nan', '0').astype(float)

    # 4. DB에 데이터 밀어넣기
    for _, row in df.iterrows():
        # 수입/지출 중 값이 있는 것을 금액으로 선택
        amount = row['income'] if row['type'] == 'income' else row['expense']
        
        cursor.execute('''
            INSERT INTO expenses (date, description, amount, category, source_type, group_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (row['date'], row['description'], int(amount), '미분류', 'CSV', 1))
    
    conn.commit()
    conn.close()
    print("-" * 30)
    print("✅ 복구 완료! 'smart_ledger.db' 파일이 생성되었습니다.")
    print("✅ 모든 CSV 데이터가 DB에 저장되었습니다.")

if __name__ == "__main__":
    restore_project()