import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .utils import evaluate_acc

class LogisticRegression:

    def __init__(self, learning_rate, max_iter, reg_factor=0, reg_type=0):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.reg_factor = reg_factor
        self.reg_type = reg_type
    
    initial_params = np.array([0]*11)
    X_train= []
    y_train = []
    model_params = []

    def _sign(self, x):
        if x>0:
            return 1
        elif x<0:
            return -1
        elif x==0:
            return 0
        
    def _stable_sigmoid(self, x):
        "Numerically stable sigmoid function."
        if x >= 0:
            z = np.exp(-x)
            sig = 1 / (1 + z)
        else:
            # if x is less than zero then z will be small, denom can't be
            # zero because it's 1+z.
            z = np.exp(x)
            sig = z / (1 + z)
        
        return sig
    
    def _sigmoid(self, x):
        exponential_factor = np.exp(-x)
        sigmoid_result = 1.0 / (1.0 + exponential_factor)
        return sigmoid_result

    def fit(self, X_train, y_train):
        print("starting model training with learning_rate = ", self.learning_rate, "and num of iterations ", self.max_iter, "reg factor", self.reg_factor, "reg type", self.reg_type)
        ## TODO: find out if this should be a plus 1, cause of the bias
#         self.initial_params = np.array([np.random.normal(size=len(X_train[0]))])
        
        # setting this to 0 for testing purposes
        self.initial_params = np.array([0]*len(X_train[0]))
        self.X_train = X_train
        self.y_train = y_train
        
        curr_iteration = 0
        current_params = self.initial_params
        
        ## there should be another check that stops that iterations as well, we can by pass the local minima
        while (curr_iteration < self.max_iter ):
            ## assuming that the learning_rate is fixed, just get it from the global data
            next_params = self._get_next_params(current_params, self.learning_rate)
            curr_iteration+=1
            current_params = next_params
        
        self.model_params = next_params
            
    ## implement the right hand side of the equation
    def _get_next_params(self, current_params, learning_rate):
        decrement = learning_rate*self._calculate_sum_factor(current_params)
        return current_params + decrement

    def _calculate_sum_factor(self, current_params):
        ## this should be a 1 x m where m is the number of features
        total = np.array([0]*len(self.X_train[0]))
        for row in range(len(self.X_train)):
            product_for_sum = self._get_internal_product(row, current_params)
            total = total + product_for_sum
        
        total = total + self._get_regularized_params(self.reg_factor, self.reg_type, current_params)
        return total
    
    def _get_regularized_params(self, reg_factor, reg_type, params):
        if reg_type==2:
            return 2 * reg_factor * params
        elif reg_type==1:
            return reg_factor * np.array([self._sign(x) for x in params])
        else:
            return 0

    def _get_internal_product(self, row, current_params):
        ## this should be a 1xm
        classification_difference = self._get_difference(row, current_params)
        return self.X_train[row] * classification_difference

    def _get_difference(self, row, current_params):
        sigmoid_product = np.dot(current_params.T, self.X_train[row])
        sigmoid_result = self._stable_sigmoid(sigmoid_product)
        #this should be mxm
        return self.y_train[row] - sigmoid_result
    
    def predict(self, X_test):
        #returns an array of y_test
        y_test= []
        
        for row in range(len(X_test)):
            sigmoid_product = np.dot(self.model_params.T, X_test[row])
            sigmoid_result = self._stable_sigmoid(sigmoid_product)
            y_test.append([1 if sigmoid_result > 0.50000 else 0])
            # print("sigmoidProduct", sigmoid_product, "sigmoid_result", sigmoid_result, "ytest", y_test[row])
        
        return np.array(y_test)

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return evaluate_acc(y_test, y_pred)


if __name__ == "__main__":	
    learning_rate= 0.000001	
    max_iterations = 100	
    df = pd.read_csv("../data/winequality/winequality-red.csv", sep=";")	
    df['classified']=[1 if x>=6 else 0 for x in df["quality"] ]	
    X_train, X_test, y_train, y_test= train_test_split(df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']], df.classified, train_size=0.9)	

    model = LogisticRegression(learning_rate, max_iterations)	
    model.fit(np.array(X_train), np.array(y_train))	
    y_pred = model.predict(np.array(X_test))	
    print(y_pred.shape)	
    print("score", model.score(np.array(X_test), np.array(y_test)))	
    # print("score", score) 