import pandas as pd
import numpy as np
# from linear_discriminant_analysis import LearnDiscriminantAnalysis as LDA
# from logistic_regression import LogisticRegression
from sklearn.model_selection import KFold

def k_fold_validation(X, y, classifier, n_fold):
    N = len(y)
    score = 1
    kf = KFold(n_splits=n_fold)
    for train_i, test_i in kf.split(X):
        X_train, X_test = X[train_i], X[test_i]
        y_train, y_test = y[train_i], y[test_i]
        classifier.fit(X_train, y_train)
        s = classifier.score(X_test, y_test)
        if s < score:
            score = s

    return score

if __name__ == "__main__":
    df = pd.read_csv("data/winequality/winequality-red.csv", sep=";")
    df['classified']=[1 if x>=6 else 0 for x in df["quality"] ]
    X = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
    y = df[['classified']]
    model =LogisticRegression(0.00001, 100)
    score = k_fold_validation(np.array(X), np.array(y), model, 5)
    print("score", score)
