import sys
import matplotlib.pyplot as plt
import seaborn as sns
import read_data as rd
import os


def box_plot(data, x_values, y_values, title='', x_label='', y_label='', file_name='user_boxplot', folder_name='plots'):
    plt.figure(figsize=(15, 6))
    sns.boxplot(x=x_values, y=y_values, data=data)

    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    plt.grid(True)
    plt.savefig(os.path.join(folder_name, file_name), dpi=300, bbox_inches='tight')


def main():
    arguments = sys.argv

    if len(arguments) < 4:
        raise Exception('Wrong number of arguments! Minimal number of arguments needed: 3')

    arg_list = [rd.read_data_from_file(arguments[1]), arguments[2], arguments[3]]

    if len(arguments) >= 5:
        arg_list.append(arguments[4])
    if len(arguments) >= 6:
        arg_list.append(arguments[5])
    if len(arguments) >= 7:
        arg_list.append(arguments[6])
    if len(arguments) == 8:
        arg_list.append('user_' + arguments[7])
    if len(arguments) > 8:
        raise Exception('Wrong number of arguments! Maximal number of arguments allowed: 7')

    box_plot(*arg_list)


if __name__ == '__main__':
    main()
