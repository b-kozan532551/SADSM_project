import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from math import sqrt


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


# Linear regression
lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Decision tree regressor
dtr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor())
])

# SVR
svr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SVR(C=100, epsilon=0.5, gamma='scale', kernel='rbf'))
])

for model, name in ((lr, 'Linear Regression'), (dtr, 'Decission Tree'), (svr, 'SVR')):
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

    #plt.figure(figsize=(8, 4))
    #plt.plot(np.arange(len(Y_test)), Y_test.values, label='Y test (prawdziwe)', marker='o')
    #plt.plot(np.arange(len(Y_test)), Y_pred, label='Y pred (przewidywane)', marker='o')
    #plt.xlabel('Numer próbki testowej')
    #plt.ylabel('Wartość wagi')
    #plt.title(f'Wartości rzeczywiste vs przewidywane')
    #plt.legend()
    #plt.tight_layout()
    #plt.show()