import pandas as pd
import numpy as np

file_path = '../b.xlsx'  # Microseismic time series data
df = pd.read_excel(file_path)

# Converting time strings to datetime types
df['time'] = pd.to_datetime(df['time'])

grouped = df.groupby('time')

b_values = {}

# 循环处理每个分组
for name, group in grouped:
    m = group['M']
    N = len(m)

    M = m.values
    M_min = M.min()
    b = (0.4343*N) / (np.sum(M - M_min) + 0.05)
    A_b = (1 / b) * np.log10(np.sum(10**(b * M)))

    # Storing results
    b_values[name] = b

for date, b_value in b_values.items():
    print(f"{date}: {b_value}")