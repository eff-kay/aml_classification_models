import numpy as np

class LogisticRegression:
    learning_rate = 0.01
    initial_params = np.array([0]*11)
    X_train= []
    y_train = []
    model_params = []
    
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

    def fit(self, learning_rate, max_iterations, X_train, y_train):
        self.learning_rate = learning_rate
        ##TODO: find out if this should be a plus 1, cause of the bias
#         self.initial_params = np.array([np.random.normal(size=len(X_train[0]))])

        # setting this to 0 for testing purposes
        self.initial_params = np.array([0]*len(X_train[0]))
        self.X_train = X_train
        self.y_train = y_train
        
        curr_iteration = 0
        current_params = self.initial_params
        
        ## there should be another check that stops that iterations as well, we can by pass the local minima
        while (curr_iteration < max_iterations):
            ## assuming that the learning_rate is fixed, just get it from the global data
            next_params = self._get_next_params(current_params, self.learning_rate)
            print("difference", abs(current_params- next_params))
            curr_iteration+=1
            current_params = next_params
            print("iteration at", curr_iteration, "params:", next_params)
        
        self.model_params = next_params
        print("Model Trained with params", self.model_params)
            
    #implement the right hand side of the equation
    def _get_next_params(self, current_params, learning_rate):
        decrement = learning_rate*self._calculate_sum_factor(current_params)
        return current_params + decrement

    def _calculate_sum_factor(self, current_params):
        ## this should be a 1 x m where m is the number of features
        total = np.array([0]*len(self.X_train[0]))
        for row in range(len(self.X_train)):
            product_for_sum = self._get_internal_product(row, current_params)
            total = total + product_for_sum
        return total
    
    def _get_internal_product(self, row, current_params):
        classification_difference = self._get_difference(row, current_params)
        return self.X_train[row] * classification_difference

    def _get_difference(self, row, current_params):
        sigmoid_product = np.dot(current_params.T, self.X_train[row])
        sigmoid_result = self._stable_sigmoid(sigmoid_product)

        #this should be mxm
        return self.y_train[row] - sigmoid_result
    
    def predict(self, X_test):
        #returns an array of y_test
        y_test= np.array([0.0]*len(X_test))
        
        for row in range(len(X_test)):
            sigmoid_product = np.dot(self.model_params.T, X_test[row])
            sigmoid_result = self._stable_sigmoid(sigmoid_product)
            y_test[row] = 1 if sigmoid_result >= 0.50000 else 0
            print("sigmoidProduct", sigmoid_product, "sigmoid_result", sigmoid_result, "ytest", y_test[row])
        
        return y_test
    
if __name__ == "__main__":
    model = LogisticRegression()
    np_xtrain = np.array([[1,2,3], [3,4,5]])
    np_ytrain = np.array([0,1])

    np_xtest = np.array([[7,2,4], [5,6,7]])
    model.fit(0.1, 50, np_xtrain, np_ytrain)
    y_test = model.predict(np_xtest)
    print("result", y_test)