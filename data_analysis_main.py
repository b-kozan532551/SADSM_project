import basic_stats as bs
import box_plot as bp
import violin_plot as vp
import error_bars as errb
import hist_plot as hp
import heat_map as hm
import regression_line as rl
import read_data as rd
import count_plot as cp
import os


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def basic_data(data, folder_name):
    bs.basic_stats_to_csv(data, folder_name)


def box_plots(data, folder_name):
    bp.box_plot(data,'NObeyesdad', 'TUE', folder_name=folder_name, file_name='obesity_TUE_boxplot', title='Impact of technology usage on obesity', x_label='Obesity level', y_label='Time spent on technological devices')
    bp.box_plot(data,'NObeyesdad', 'FAF', folder_name=folder_name, file_name='obesity_FAF_boxplot', title='Impact of physical activity on obesity', x_label='Obesity level', y_label='Physical activity')


def violin_plots(data, folder_name):
    vp.violin_plot(data, 'family_history_with_overweight', 'Weight', folder_name=folder_name, file_name='fhwo_weight_violinplot', title='Correlation between weight and family history of ilness', x_label='Family history with overweight', y_label='Weight')


def error_bars(data, folder_name):
    errb.error_bars(data, 'SMOKE', 'Weight', folder_name=folder_name, file_name='smoking_weight_errorbars', title='Correlation between smoking and weight', x_label='Smoking', y_label='Weight')
    errb.error_bars(data, 'NObeyesdad', 'Height', folder_name=folder_name, file_name='obesity_height_errorbars', title='Correlation between obesity level and height', x_label='Obesity level', y_label='Height')


def hist_plots(data, folder_name):
    hp.hist_plot(data, 'CH2O', file_name='CH2O_histplot', folder_name=folder_name, title='How many litres of H2O per day', x_label='H2O per day', y_label='Number of people')


def hue_hist_plots(data, folder_name):
    hp.hist_plot_hue(data, 'NCP', hue='NObeyesdad', folder_name=folder_name, file_name='obesity_NCP_huehistplot', title='Correlation between obesity status and number of meals', x_label='Number of meals', y_label='Number of people')
    hp.hist_plot_hue(data, 'Weight', hue='Gender', folder_name=folder_name, file_name='gender_weight_huehistplot', title='Correlation between weight and gender', x_label='Weight', y_label='Number of people')


def heat_map(data, folder_name):
    hm.all_numeric_data_heat_map(data, folder_name=folder_name)


def regression_lines(data, folder_name):
    rl.regression_line(data, 'Age', 'TUE', folder_name=folder_name, file_name='age_TUE_regressionline', title='Correlation between use of technological devices and age', x_label='Age', y_label='Time spent on technological devices')
    rl.regression_line(data, 'Weight', 'Height', folder_name=folder_name, file_name='weight_height_regressionline', title='Correlation between weight and height', x_label='Weight', y_label='Height')
    rl.lm_regression_line(data, 'Age', 'FAF', 'SMOKE', folder_name=folder_name, file_name='FAF_age_SMOKE_regressionline', title='Correlation between age and physical activity', x_label='Age', y_label='Physical activity')


def count_plots(data, folder_name):
    cp.count_plot(data, 'NObeyesdad', folder_name=folder_name, title='Influence of family history on persons obesity', x_label='Obesity level', y_label='Number of people', file_name='obesity_fhwo_countplot', hue='family_history_with_overweight')
    cp.count_plot(data, 'NObeyesdad', folder_name=folder_name, title='Correlation between gender and obesity', x_label='Obesity level', y_label='Number of people', file_name='obesity_gender_countplot', hue='Gender')


def write_all_data(data):
    stats_folder_name = 'statistics'
    plots_folder_name = 'plots'

    create_folder(stats_folder_name)
    create_folder(stats_folder_name)

    basic_data(data, stats_folder_name)

    box_plots(data, plots_folder_name)
    violin_plots(data, plots_folder_name)
    error_bars(data, plots_folder_name)
    hist_plots(data, plots_folder_name)
    hue_hist_plots(data, plots_folder_name)
    heat_map(data, plots_folder_name)
    regression_lines(data, plots_folder_name)
    count_plots(data, plots_folder_name)


if __name__ == '__main__':
    write_all_data(rd.read_data_from_file('data.csv'))
