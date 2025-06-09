import pandas as pd

a = "12 45 as"

res = a.split()

df = pd.DataFrame([res])
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='ignore')

cat_cols = df.select_dtypes(include='object').columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
print(df[cat_cols])
print(df[num_cols])