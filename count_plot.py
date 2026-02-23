import matplotlib.pyplot as plt
import seaborn as sns
import read_data as rd
import os
import sys


def count_plot(data, x_values, title='', x_label='', y_label='', file_name='user_countplot', hue='', folder_name='plots'):
    plt.figure(figsize=(15, 6))
    sns.countplot(x=x_values, data=data, hue=hue, palette='tab10')

    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    plt.savefig(os.path.join(folder_name, file_name), dpi=300, bbox_inches='tight')


def main():
    arguments = sys.argv

    if len(arguments) < 3:
        raise Exception('Wrong number of arguments! Minimal number of arguments needed: 3')

    arg_list = [rd.read_data_from_file(arguments[1]), arguments[2]]

    if len(arguments) >= 4:
        arg_list.append(arguments[3])
    if len(arguments) >= 5:
        arg_list.append(arguments[4])
    if len(arguments) >= 6:
        arg_list.append(arguments[5])
    if len(arguments) >= 7:
        arg_list.append('user_' + arguments[6])
    if len(arguments) == 8:
        arg_list.append(arguments[7])
    if len(arguments) > 8:
        raise Exception('Wrong number of arguments! Maximal number of arguments allowed: 7')

    count_plot(*arg_list)


if __name__ == '__main__':
    main()
