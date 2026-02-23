import numpy as np
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
from sklearn.base import BaseEstimator, RegressorMixin


class RegLin(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.coeffs = None

    def fit(self, X, Y):
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        self.coeffs = np.linalg.pinv(X.T @ X) @ X.T @ Y
        return self

    def predict(self, X):
        return np.hstack([X, np.ones((X.shape[0], 1))]) @ self.coeffs

    def score(self, X, Y):
        return 1 - np.sum((Y - self.predict(X)) ** 2) / np.sum((Y - np.mean(Y)) ** 2)


class RegLinGD(BaseEstimator, RegressorMixin):
    def __init__(self, lr=0.01, iters=500, batch_size=64):
        self.lr = lr
        self.iters = iters
        self.batch_size = batch_size
        self.coeffs = None

    def compute_gradient(self, X_batch, Y_batch):
        Y_pred = X_batch @ self.coeffs
        return (2/X_batch.shape[0]) * X_batch.T @ (Y_pred - Y_batch)

    def fit(self, X, Y):
        Y = Y.to_numpy()
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        n_samples, n_features = X.shape
        self.coeffs = np.zeros(n_features)

        for _ in range(self.iters):
            indices = np.random.permutation(n_samples)

            X_shuffled = X[indices]
            Y_shuffled = Y[indices]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                Y_batch = Y_shuffled[i:i + self.batch_size]

                self.coeffs -= self.lr * self.compute_gradient(X_batch, Y_batch)

        return self

    def predict(self, X):
        return np.hstack([X, np.ones((X.shape[0], 1))]) @ self.coeffs

    def score(self, X, Y):
        return 1 - np.sum((Y - self.predict(X)) ** 2) / np.sum((Y - np.mean(Y)) ** 2)


data = pd.read_csv('data.csv')
data = data[['Height', 'Weight', 'NObeyesdad']]

X_data = data.drop('Weight', axis=1)
Y_data = data['Weight']

num_cols = X_data.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X_data.select_dtypes(include=['object', 'category']).columns

preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
])

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

lr_closed_form = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RegLin())
])

lr_gradient = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RegLinGD())
])

kf = KFold(n_splits=3, shuffle=True, random_state=42)

for model, name in ((lr, 'Linear Regression'), (dtr, 'Decission Tree'), (svr, 'SVR'), (lr_closed_form, 'Closed form'), (lr_gradient, 'Gradient')):
    print(f'---------- {name} ----------')
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
        print(f'Chunk number {counter}:')
        print(f'Train R^2: {r_square_train[-1]}')
        print(f'Test R^2: {r_square_test[-1]}')
        print(f'Train RMSE: {rmse_train[-1]}')
        print(f'Test RMSE: {rmse_test[-1]}')
    print(f'Mean:')
    print(f'Train R^2: {sum(r_square_train) / 3}')
    print(f'Test R^2: {sum(r_square_test) / 3}')
    print(f'Train RMSE: {sum(rmse_train) / 3}')
    print(f'Test RMSE: {sum(rmse_test) / 3}')
