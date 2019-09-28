import pandas as pd
import numpy as np
# from linear_discriminant_analysis import LearnDiscriminantAnalysis as LDA
# from logistic_regression import LogisticRegression
from sklearn.model_selection import KFold
import time
from statistics import mean

def k_fold_validation(X, y, classifier, n_fold):
    N = len(y)
    scores = []
    times = []
    kf = KFold(n_splits=n_fold)
    for train_i, test_i in kf.split(X):
        X_train, X_test = X[train_i], X[test_i]
        y_train, y_test = y[train_i], y[test_i]
        start_t = time.time()
        classifier.fit(X_train, y_train)
        s = classifier.score(X_test, y_test)
        end_t = time.time()
        t= end_t-start_t
        print("score: " + str(s[0]))
        print("time: " + str(t))
        scores.append(s[0])
        times.append(t)

    mean_score = mean(scores)
    mean_time = mean(times)
    
    print("average score: " + str(mean_score))
    print("average time: " + str(mean_time))
    return mean_score

if __name__ == "__main__":
    df = pd.read_csv("data/winequality/winequality-red.csv", sep=";")
    df['classified']=[1 if x>=6 else 0 for x in df["quality"] ]
    X = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
    y = df[['classified']]
    model =LogisticRegression(0.00001, 100)
    score = k_fold_validation(np.array(X), np.array(y), model, 5)
    print("score", score)
