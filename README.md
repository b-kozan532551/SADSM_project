##INFORMATION

This project provides basic tools for data analysis from .csv files in python

--Key features--
- Generating basic plots based on given data
- Reading the csv file
- Generating basic statistics based on given data

--Needed components--
Python version: 3.12.0
Necessary libraries:
- numpy
- seaborn
- matplotlib
- pandas



##INSTALLATION

###Open the terminal

###First create a new virtual environment:

python -m venv venv

###Next activate it:

.\venv\Scripts\Activate [\venv\Scripts\Activate - if using cmd]

###Finally install the necessary libraries:

pip install -r requirements.txt



##USAGE

!!! optional values are given in []

1.

python read_data.py <data>

Reads data from csv file and prints it to stdout
Properties:
- data - name of the file to analyse

2.

python basic_stats.py <data>

Returns different basic statistics about values as csv files in the 'statistics' folder
Properties:
- data - name of the file to analyse

3.

python box_plot.py <data> <x_values> <y_values> [title] [x_label] [y_label] [filename]

Generates a boxplot in the 'plots' folder
Properties:
- data - name of the file to analyse 
- x_values - header of the column which values are supposed to be on the X axis
- y_values - header of the column which values are supposed to be on the Y axis
- title - the title of the plot (optional)
- x_label - the label visible at the X axis (optional)
- y_label - the label visible at the Y axis (optional)
- file_name - name of the file containing the plot (optional)

4.

python violin_plot.py <data> <x_values> <y_values> [title] [x_label] [y_label] [filename]

Generates a violinplot in the 'plots' folder
Properties:
- data - name of the file to analyse 
- x_values - header of the column which values are supposed to be on the X axis
- y_values - header of the column which values are supposed to be on the Y axis
- title - the title of the plot (optional)
- x_label - the label visible at the X axis (optional)
- y_label - the label visible at the Y axis (optional)
- file_name - name of the file containing the plot (optional)

5.

python error_bars.py <data> <x_values> <y_values> [title] [x_label] [y_label] [filename]

Generates a barplot with error bars in the 'plots' folder
Properties:
- data - name of the file to analyse 
- x_values - header of the column which values are supposed to be on the X axis
- y_values - header of the column which values are supposed to be on the Y axis
- title - the title of the plot (optional)
- x_label - the label visible at the X axis (optional)
- y_label - the label visible at the Y axis (optional)
- file_name - name of the file containing the plot (optional)

6.

python hist_plot.py <data> <x_values> [title] [x_label] [y_label] [filename] [hue]

Generates a histplot in the 'plots' folder
Properties:
- data - name of the file to analyse 
- x_values - header of the column which values are supposed to be on the X axis
- title - the title of the plot (optional)
- x_label - the label visible at the X axis (optional)
- y_label - the label visible at the Y axis (optional)
- file_name - name of the file containing the plot (optional)
- hue - categorical data used to add an extra layer of differentiation (optional)

7.

python heat_map.py <data> [filename]

Generates a heatmap of all the numerical columns from data in the 'plots' folder
Properties:
- data - name of the file to analyse
- file_name - name of the file containing the plot (optional)

8.

python regression_line.py <data> <x_values> <y_values> [title] [x_label] [y_label] [filename]

Generates a pointplot with a regression line in the 'plots' folder
Properties:
- data - name of the file to analyse 
- x_values - header of the column which values are supposed to be on the X axis
- y_values - header of the column which values are supposed to be on the Y axis
- title - the title of the plot (optional)
- x_label - the label visible at the X axis (optional)
- y_label - the label visible at the Y axis (optional)
- file_name - name of the file containing the plot (optional)

9.

python count_plot.py <data> <x_values> [title] [x_label] [y_label] [filename] [hue]

Generates a countplot in the 'plots' folder
Properties:
- data - name of the file to analyse 
- x_values - header of the column which values are supposed to be on the X axis
- title - the title of the plot (optional)
- x_label - the label visible at the X axis (optional)
- y_label - the label visible at the Y axis (optional)
- file_name - name of the file containing the plot (optional)
- hue - categorical data used to add an extra layer of differentiation (optional)