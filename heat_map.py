import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import read_data as rd
import os
import sys


def all_numeric_data_heat_map(data, file_name='user_all_numeric_data_heatmap', folder_name='plots'):
    plt.figure(figsize=(15, 6))

    data_corr = pd.DataFrame()
    for name, element in data.items():
        try:
            float(element[0])
            data_corr[name] = data[name]
        except ValueError:
            pass

    sns.heatmap(data_corr.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

    plt.title('Heatmap of every numerical data')

    plt.savefig(os.path.join(folder_name, file_name), dpi=300, bbox_inches='tight')


def main():
    arguments = sys.argv

    if len(arguments) < 2:
        raise Exception('Wrong number of arguments! Minimal number of arguments needed: 1')

    arg_list = [rd.read_data_from_file(arguments[1])]

    if len(arguments) == 3:
        arg_list.append('user_' + arguments[2])
    if len(arguments) > 3:
        raise Exception('Wrong number of arguments! Maximal number of arguments allowed: 2')

    all_numeric_data_heat_map(*arg_list)


if __name__ == '__main__':
    main()
