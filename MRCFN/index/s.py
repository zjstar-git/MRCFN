import pandas as pd
import numpy as np

file_path = '../b.xlsx'
df = pd.read_excel(file_path)

df['time'] = pd.to_datetime(df['time'])
grouped = df.groupby('time')
b_values = {}

for name, group in grouped:
    m = group['M']
    N = len(m)
    M = m.values
    M_min = M.min()
    b = (0.4343*N) / (np.sum(M - M_min) + 0.05)
    A_b = (1 / b) * np.log10(np.sum(10**(b * M)))
    s = 0.117 * np.log10(N + 1) + 0.029 * (np.log10(np.sum(10**(1.5 * M) + 0.015 * M.max())))
    b_values[name] = b

for date, b_value in b_values.items():
    print(f"{date}: {b_value}")