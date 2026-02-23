import pandas as pd
import sys


def read_data_from_file(filename):
    data = pd.read_csv(filename)
    data['NObeyesdad'] = pd.Categorical(data['NObeyesdad'], categories=['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'], ordered=True)
    return convert_numeric_data(data)


def convert_numeric_data(data):
    for name, element in data.items():
        try:
            float(element[0])
            data[name] = pd.to_numeric(data[name])
        except ValueError:
            pass

    return data


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception('Cannot find the filename')
    data = read_data_from_file(sys.argv[1])
    print(data)
