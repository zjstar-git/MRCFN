from index import index_data
from index import create_data
import pandas as pd

def timeseries_data(TF_file_path, index_file_path):
    tf_data = create_data.creat_data(TF_file_path)
    index = index_data.index(index_file_path)
    b_dropped = index.iloc[:, 1:]
    combined = pd.concat([tf_data, b_dropped], axis=1)
    return combined