import pandas as pd

def creat_data(path):
    data = pd.read_excel(path)

    results = []
    for date, group in data.groupby('time'):
        energy_values = group['energy'].values

        results.append({
            'data': date,
            'max_energy': group['energy'].max(),
            'avg_energy': group['energy'].mean(),
            'event_frequency': len(group),
            'energy_sum': group['energy'].sum()
        })
    result_df = pd.DataFrame(results)
    return result_df


if __name__ == '__main__':
    file_path = '../data_set/result_data.xlsx'
    result_df = creat_data(file_path)
    output_path = '../module/results.xlsx'
    result_df.to_excel(output_path, index=False)
