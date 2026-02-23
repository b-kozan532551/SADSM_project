import read_data as rd
import statistics
import numpy as np
import pandas as pd
import os
import sys


def check_if_numerical(data):
    for element in data:
        if type(element) is not float:
            return False
    return True


def return_text_stats(list):
    values = list.value_counts()
    stats = values.reset_index()
    stats.columns = ['name', 'number']

    empty_values = pd.DataFrame({'name': ['Empty'], 'number': [len(list) - len(list.loc[list.notna()])]})
    stats = pd.concat([stats, empty_values], ignore_index=True)

    return stats


def return_numerical_stats(list):
    original_size = len(list)
    list = list.loc[list.notna()]

    stats = {
        'avg': sum(list) / len(list),
        'median': statistics.median(list),
        'min': min(list),
        'max': max(list),
        'deviation': np.std(list),
        '5th percentile': np.percentile(list, 5),
        '95th percentile': np.percentile(list, 95),
        'missing values': original_size - len(list)
    }

    return pd.Series(stats)


def basic_stats_to_csv(data, folder_name):
    numerical_stats = pd.DataFrame(columns=['avg', 'median', 'min', 'max', 'deviation', '5th percentile', '95th percentile', 'missing values'])

    counter = 0
    for name, values in data.items():
        try:
            float(values[0])
            numerical_stats.loc[len(numerical_stats)] = return_numerical_stats(values)
            numerical_stats.rename(index={counter: name}, inplace=True)
            counter += 1
        except ValueError:
            return_text_stats(values).to_csv(os.path.join(folder_name, name + '.csv'), index=False)

    numerical_stats.to_csv(os.path.join(folder_name, 'numerical_data.csv'), index=True, index_label='Data type')
    return numerical_stats


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception('Cannot find the filename')
    data = rd.read_data_from_file(sys.argv[1])
    basic_stats_to_csv(data, 'statistics')
