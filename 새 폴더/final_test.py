import pandas as pd

data = pd.read_csv('expenses_20251224.csv')

try:
    df = pd.read_csv('expenses_20251224.csv', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('expenses_20251224.csv', encoding='cp949')

# 데이터 정제 (컬럼 공백 제거 및 금액 숫자 변환)
df.columns = df.columns.str.strip()
df['income'] = df['income'].str.replace(',', '').fillna(0).astype(float)
df['expense'] = df['expense'].str.replace(',', '').fillna(0).astype(float)

def clean_currency(column):
    # 콤마 제거 후 숫자로 변환, 빈 값은 0으로 채움
    return pd.to_numeric(df[column].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)

# 각 컬럼에 적용
df['income'] = clean_currency('income')
df['expense'] = clean_currency('expense')
df['balance'] = clean_currency('balance')

# 저장
df.to_csv('cleaned_expenses.csv', index=False, encoding='utf-8-sig')













