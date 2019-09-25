from logistic_regression import LogisticRegression
import pandas as pd
import numpy as np


def validate_logistic_regression_for_wine_quality():
    num_of_folds = 5
    learning_rate= 0.000001
    max_iterations = 100
    df = pd.read_csv("../data/winequality/winequality-red.csv", sep=";")
    df['classified']=[1 if x>=6 else 0 for x in df["quality"] ]
    fold_size = int(round(df.shape[0]/num_of_folds))
    for i in range(num_of_folds):
        x_test = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']][i*fold_size:fold_size+i*fold_size]
        x_train_part_1 = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']][fold_size+i*fold_size:]
        x_train_part_2 = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']][:i*fold_size]
        x_train = x_train_part_1.append(x_train_part_2)
        print(x_train)
        print(x_test)


        y_test = df[['classified']][i*fold_size:fold_size+i*fold_size]
        y_train_part_1 = df[['classified']][fold_size+i*fold_size:]
        y_train_part_2 = df[['classified']][:i*fold_size]
        y_train = y_train_part_1.append(y_train_part_2)
        print(y_train)
        print(y_test)

        model = LogisticRegression()
        model.fit(learning_rate, max_iterations, np.array(x_train), np.array(y_train))
        y_pred = model.predict(np.array(x_test))
        print(y_pred)
        print("score", model.score(np.array(x_test), np.array(y_test)))
    pass



if __name__ == "__main__":
    validate_logistic_regression_for_wine_quality()
    pass

