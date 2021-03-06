import math
from scipy.special import expit
import numpy as np
import utils


class Logistic:
    def __init__(self, params):
        """
        initialize random weights
        params is the number of parameters
        iterations is how many time we do gradiant descent
        """
        self.weights = np.random.rand(params)

    #    def sigmoid(self,z):
    #        """
    #        sigmoid function
    #        """
    #        return 1./(1.+math.exp(-z))

    def loglikelyhood(self, X, y):
        sum_ = 0
        weights = self.weights

        for i in range(len(X)):
            sig = self.sigmoid(np.dot(weights.T, X[i]))
            sum_ += y[i] * np.log(sig) + (1 - y[i]) * np.log(1 - sig)
        return sum_

    def sigmoid(self, x):
        """
        Numerically stable sigmoid function.
        Taken from https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        """
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            # if x is less than zero then z will be small, denom can't be
            # zero because it's 1+z.
            z = np.exp(x)
            return z / (1 + z)

    def fit(self, X, y, iterations, lr, lr_func, threshold, lamb=0):
        """
        Parameters
        ----------
        X: np.array (m x n)
            The data
        Y: np.array (m x 1))
            The training output data were fitting to

        params: List
            iterations: int
                the number of iterations to run the fit function
            lr: float
                the learning rate ("alpha")
            lr_func: lambda func
                a function that will be used to update the learning rate at every iteration
            lamb: float
                lambda for lasso regularization
        """
        X = np.insert(X, 0, 1, axis=1)

        self.weights = np.random.rand(np.shape(self.weights)[0])
        # assert(n == y.shape[0], "the data and output array shapes are not equal")

        weights = self.weights

        # print(X.shape, y.shape, weights.shape)
        # it = 0

        diff_logodds = np.inf
        prev_logodds = np.inf
        total_log_likelyhood = []
        for i in range(iterations):
            if diff_logodds < threshold:
                break

            sum_ = np.zeros((len(weights),))
            for j, row in enumerate(X):
                sig = self.sigmoid(np.dot(weights, row.T))
                sum_ += np.multiply(row, (y[j] - sig))  # - self.sign(weights,lamb)

            weights += np.multiply(sum_ + lamb * weights, lr_func(lr, i))

            self.weights = weights

            log_likelyhood = self.loglikelyhood(X, y)

            diff_logodds = np.abs(prev_logodds - log_likelyhood)
            prev_logodds = log_likelyhood
            total_log_likelyhood.append(log_likelyhood)
        #print(self.weights)
        #print('weights: ',self.weights)
        return total_log_likelyhood
    def sign(self, weights, lamb):
        weights = np.copy(weights)
        weights[weights > 0] = lamb
        weights[weights < 0] = -lamb
        weights[weights == 0] = 0.
        return weights

    def predict(self, X, threshhold=0.5):
        X0 = np.zeros((X.shape[0]))
        X0[X0 == 0] = 1
        X_prime = np.concatenate((X0[:, np.newaxis], X), axis=1)
        z = np.dot(X_prime, self.weights)
        prob = [self.sigmoid(a) for a in z]

        return [1 if i > 0.5 else 0 for i in prob]

        # if prob > 0.5:
        #    return 1
        # else:
        #    return 0


if __name__ == '__main__':
    import load_files
    import utils
    wines, wine_headers = load_files.load_cancer()
    model = Logistic(wines.shape[1])
    params1 = [1000, 0.004, lambda x, y: x, 0.5]
    print(utils.CrossValidation(wines.copy(), model, 5, params1))

    # cancer, cancer_header = load_files.load_cancer()
    #
    # x = cancer[:,:-1]
    # model2 = Logistic(x.shape[1], 1000)
    # print(utils.CrossValidation(cancer, model2, 5))
    #
