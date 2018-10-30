# Thakur, Nishant
# 1001-544-591
# 2018-10-29
# Assignment-04-01
import numpy as np

def read_csv_as_matrix(file_name):
    # Each row of data in the file becomes a row in the matrix
    # So the resulting matrix has dimension [num_samples x sample_dimension]
    data = np.loadtxt(file_name, skiprows=1, delimiter=',', dtype=np.float32)
    return data

def normalize_data(data):
    res = data.T
    for i in range(len(res)):
        min_val = np.min(res[i])
        max_val = np.max(res[i])
        res[i] = 2 * ((res[i] - min_val) / (max_val - min_val)) - 1
    return res.T

def load_and_normalize_data(filename):
    raw_data = read_csv_as_matrix(filename)
    normalized_data = normalize_data(raw_data)
    return normalized_data


if __name__ == "__main__":
    filename = "data_set_2.csv"
    result = load_and_normalize_data(filename)
    print(result)