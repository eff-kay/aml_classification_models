# Binary LDA
import math
import numpy as np

class LearnDiscriminantAnalysis:
    def __init__(self):
        self.m1 = None
        self.m0 = None
        self.co_var = None
        self.w = None

    def fit(self, X, y):
        X1,X0 = X[y == 1], X[y == 0]
        N1,N0 = X1.count, X0.count

        # determine P
        P1 = N1 / (N1 + N0)
        P0 = N0 / (N1 + N0)

        #find mean
        m1 = X1.mean().values.reshape(n_feat, 1)
        m0 = X0.mean().values.reshape(n_feat, 1)
        self.m1, self.m0 = m1, m0

        #find co-variance
        n_feat = len(m1)
        co_var = np.zeros((n_feat, n_feat))

        for _, row in X1.iterrows():
            x = row.values.reshape(n_feat, 1)
            m = m1
            co_var += (x - m).dot((x - m).T / (N1 + N0 - 2))

        for row in X0.iterrows():
            x = row.values.reshape(n_feat, 1)
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
            pred.append(res)
        return pred
