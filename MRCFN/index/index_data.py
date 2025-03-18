import pandas as pd
import numpy as np

def index(path):
    df = pd.read_excel(path)
    df['time'] = pd.to_datetime(df['time'])

    grouped = df.groupby('time')

    b_values = {}
    A_b_values = {}
    s_values = {}
    F_values = {}
    AC_values = {}

    for name, group in grouped:
        m = group['M']
        N = len(m)
        M = m.values
        M_min = M.min()
        b = (0.4343*N) / (np.sum(M - M_min) + 0.05)
        A_b = (1 / b) * np.log10(np.sum(10**(b * M)))
        s = 0.117 * np.log10(N + 1) + 0.029 * (np.log10(np.sum(10**(1.5 * M) + 0.015 * M.max())))
        F = np.log10((np.sum(10**(6.11 + 1.09 * M))))

        changes = np.sum(group['energy'].values[1:] != group['energy'].values[:-1])
        if changes == 0:
            changes = changes + 1
        AC = np.log(changes) / (changes * np.log(M.max() - M.min() + 1))

        # Storing results
        b_values[name] = b
        A_b_values[name] = A_b
        s_values[name] = s
        F_values[name] = F
        AC_values[name] = AC

    results_df = pd.DataFrame({
        'date': list(b_values.keys()),
        'b_value': list(b_values.values()),
        'A_b_value': list(A_b_values.values()),
        's_value': list(s_values.values()),
        'F_value': list(F_values.values()),
        'AC_values': list(AC_values.values()),
    })
    return results_df

if __name__ == '__main__':
    file_path = '../b.xlsx'
    results_df = index(file_path)
    results_df.set_index('date', inplace=True)
    # 打印结果
    print(results_df)
    results_df.to_excel('index.xlsx')



