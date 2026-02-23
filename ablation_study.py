from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingRegressor, StackingRegressor
import warnings

warnings.filterwarnings("ignore")


def cross_valid(models, X_data, Y_data):
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    for name, model in models:
        print(f'\n{name}')
        r_square_train = []
        rmse_train = []
        r_square_test = []
        rmse_test = []
        counter = 0
        for train, test in kf.split(X_data):
            counter += 1
            model.fit(X_data.iloc[train], Y_data.iloc[train])
            r_square_train.append(model.score(X_data.iloc[train], Y_data.iloc[train]))
            r_square_test.append(model.score(X_data.iloc[test], Y_data.iloc[test]))
            rmse_train.append(sqrt(mean_squared_error(Y_data.iloc[train], model.predict(X_data.iloc[train]))))
            rmse_test.append(sqrt(mean_squared_error(Y_data.iloc[test], model.predict(X_data.iloc[test]))))
        print(f'    Train R^2: {sum(r_square_train) / 3}')
        print(f'    Test R^2: {sum(r_square_test) / 3}')
        print(f'    Train RMSE: {sum(rmse_train) / 3}')
        print(f'    Test RMSE: {sum(rmse_test) / 3}')


def standard(preprocessor):
    lr = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    dtr = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor())
    ])

    svr = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', SVR())
    ])

    return [('Linear Regression', lr), ('Decision Tree', dtr), ('SVR', svr)]


def regularization(preprocessor):
    ridge = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge())
    ])

    lasso = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Lasso())
    ])

    dtr = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor())
    ])

    svr = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', SVR())
    ])

    return [('Ridge', ridge), ('Lasso', lasso), ('Decision Tree', dtr), ('SVR', svr)]


def grid_search(X_train, Y_train):
    ridge = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge())
    ])

    param_grid = {'regressor__alpha': [0.01, 0.1, 1.0, 10.0, 100.0], 'regressor__fit_intercept': [True, False]}

    ridge_grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error',
                                   n_jobs=-1)
    ridge_grid_search.fit(X_train, Y_train)

    lasso = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Lasso())
    ])

    param_grid = {'regressor__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0], 'regressor__fit_intercept': [True, False]}

    lasso_grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error',
                                   n_jobs=-1)
    lasso_grid_search.fit(X_train, Y_train)

    dtr = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor())
    ])

    param_grid = {'regressor__max_depth': [None, 5, 10, 20], 'regressor__min_samples_split': [2, 5, 10],
                  'regressor__min_samples_leaf': [1, 2, 4], 'regressor__max_features': [None, 'auto', 'sqrt']}

    dtr_grid_search = GridSearchCV(estimator=dtr, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error',
                                   n_jobs=-1)
    dtr_grid_search.fit(X_train, Y_train)

    svr = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', SVR())
    ])

    param_grid = {'regressor__C': [0.1, 1, 10, 100], 'regressor__epsilon': [0.1, 0.2, 0.5],
                  'regressor__kernel': ['linear', 'rbf'], 'regressor__gamma': ['scale', 'auto']}

    svr_grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error',
                                   n_jobs=-1)
    svr_grid_search.fit(X_train, Y_train)

    return [('Ridge', ridge_grid_search.best_estimator_), ('Lasso', lasso_grid_search.best_estimator_), ('DecisionTree', dtr_grid_search.best_estimator_), ('SVR', svr_grid_search.best_estimator_)]


def ensemble(models):
    vr = VotingRegressor(models)
    sr = StackingRegressor(estimators=models)

    return [('Voting Regressor', vr), ('Stacking Regressor', sr)]


if __name__ == '__main__':
    data = pd.read_csv('data.csv')
    data = data[['Height', 'Weight', 'NObeyesdad']]

    X_data = data.drop('Weight', axis=1)
    Y_data = data['Weight']

    num_cols = X_data.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X_data.select_dtypes(include=['object', 'category']).columns

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
    ])

    print('---------- CROSS VALIDATION ----------')
    cross_valid(standard(preprocessor), X_data, Y_data)

    print('\n---------- REGULARIZATION ----------')
    cross_valid(regularization(preprocessor), X_data, Y_data)

    gscv = grid_search(X_data, Y_data)
    print('\n---------- GRID SEARCH ----------')
    cross_valid(gscv, X_data, Y_data)

    print('\n---------- ENSEMBLE ----------')
    cross_valid(ensemble(gscv), X_data, Y_data)
