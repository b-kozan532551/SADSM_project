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
from sklearn.ensemble import VotingRegressor, StackingRegressor


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

vr = VotingRegressor([('lr', lr), ('dtr', dtr), ('svr', svr)])

sr = StackingRegressor(estimators=[('lr', lr), ('dtr', dtr), ('svr', svr)])

for model, name in ((vr, 'Voting Regressor'), (sr, 'Stacking Regressor')):
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
