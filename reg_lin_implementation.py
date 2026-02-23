import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt


class RegLin:
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


class RegLinGD:
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

X_train, X_temp, Y_train, Y_temp = train_test_split(X_data, Y_data, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.4, random_state=42)

X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)

regLin = RegLin()

regLinGD = RegLinGD()

for model, name in ((regLin, 'Closed form'), (regLinGD, 'Gradient')):
    model.fit(X_train, Y_train)
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