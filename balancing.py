from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import GridSearchCV


data = pd.read_csv('data.csv')
data = data[['Height', 'Weight', 'NObeyesdad']]

X_data = data.drop('Weight', axis=1)
Y_data = data['Weight']

num_cols = X_data.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X_data.select_dtypes(include=['object', 'category']).columns

preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

X_train, X_temp, Y_train, Y_temp = train_test_split(X_data, Y_data, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.4, random_state=42)

dtr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor())
])

param_grid = {'regressor__max_depth': [None, 5, 10, 20], 'regressor__min_samples_split': [2, 5, 10], 'regressor__min_samples_leaf': [1, 2, 4], 'regressor__max_features': [None, 'auto', 'sqrt']}

dtr_grid_search = GridSearchCV(estimator=dtr, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
dtr_grid_search.fit(X_train, Y_train)

dtr_GSCV = dtr_grid_search.best_estimator_

svr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SVR())
])

param_grid = {'regressor__C': [0.1, 1, 10, 100], 'regressor__epsilon': [0.1, 0.2, 0.5], 'regressor__kernel': ['linear', 'rbf'], 'regressor__gamma': ['scale', 'auto']}

svr_grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
svr_grid_search.fit(X_train, Y_train)

svr_GSCV = svr_grid_search.best_estimator_

for model, name in ((dtr, 'Decision Tree'), (dtr_GSCV, 'Decision Tree ' + str(dtr_grid_search.best_params_)), (svr, 'SVR'), (svr_GSCV, 'SVR ' + str(svr_grid_search.best_params_))):
    print(f'{name}:')
    print('1. R^2')
    model.fit(X_train, Y_train)
    print(f'    Train: {model.score(X_train, Y_train)}')
    print(f'    Validate: {model.score(X_val, Y_val)}')
    print(f'    Score: {model.score(X_test, Y_test)}')

    Y_pred = model.predict(X_test)

    print('2. RMSE')
    print(f'    Train: {sqrt(mean_squared_error(Y_train, model.predict(X_train)))}')
    print(f'    Validate: {sqrt(mean_squared_error(Y_val, model.predict(X_val)))}')
    print(f'    Score: {sqrt(mean_squared_error(Y_test, Y_pred))}\n')
