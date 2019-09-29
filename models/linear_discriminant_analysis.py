# Binary LDA
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .utils import evaluate_acc

class LearnDiscriminantAnalysis:
    def __init__(self):
        self.m1 = None
        self.m0 = None
        self.co_var = None
        self.w = None

    def fit(self, X, y):
        X1,X0 = X[y[:,0] == 1], X[y[:,0] == 0]

        N1,N0 = len(X1), len(X0)

        # determine P
        P1 = N1 / (N1 + N0)
        P0 = N0 / (N1 + N0)

        plog = math.log(P1/P0)
        #find mean
        mc1 = X1.mean(0)
        mc0 = X0.mean(0)

        n_feat = len(mc1)

        m1 = mc1.reshape(n_feat, 1)
        m0 = mc0.reshape(n_feat, 1)
        self.m1, self.m0 = m1, m0

        #find co-variance
        co_var = np.zeros((n_feat, n_feat))

        for row in X1:
            x = row.reshape(n_feat, 1)
            m = m1
            co_var += (x - m).dot((x - m).T / (N1 + N0 - 2))

        for row in X0:
            x = row.reshape(n_feat, 1)
            m = m0
            co_var += (x - m).dot((x - m).T) / (N1 + N0 - 2)

        self.co_var = co_var
        self.w = plog \
            - ((m1.T).dot(np.linalg.inv(co_var))).dot(m1)/2 \
            + ((m0.T).dot(np.linalg.inv(co_var))).dot(m0)/2
        
    def predict(self, X):
        pred = []
        for row in X:
            ratio = self.w + ((row.T).dot(np.linalg.inv(self.co_var))).dot(self.m1 - self.m0)
            res = 1 if ratio >= 0 else 0
            pred.append([res])
        return np.array(pred)

    def score(self, X, y_test):
        y_pred = self.predict(X)
        return evaluate_acc(y_test, y_pred)

if __name__ == "__main__":
    df = pd.read_csv("../data/winequality/winequality-red.csv", sep=";")	
    df['classified']=[1 if x>=6 else 0 for x in df["quality"] ]	
    X_train, X_test, y_train, y_test= train_test_split(df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']], df.classified, train_size=0.9)
    wine_model = LearnDiscriminantAnalysis()
    wine_model.fit(X_train, y_train)
    y_predicted = wine_model.predict(X_test)
    wine_model.score(X_test, y_test)