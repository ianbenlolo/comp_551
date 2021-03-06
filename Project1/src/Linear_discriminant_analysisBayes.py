import numpy as np
import pandas as pd

#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class LDA():

    '''constructor to initialize weights'''
    def __init__(self, wY0, w0Y0, wY1, w0Y1):
        self.wY0 = wY0
        self.wY1 = wY1
        self.w0Y0 = w0Y0
        self.w0Y1 = w0Y1

    def fit(self, X, y):
        '''calculate mean vector for each class'''
        X_0 = np.array([row for i, row in enumerate(X) if y[i] == 0])
        X_1 = np.array([row for i, row in enumerate(X) if y[i] == 1])
        mean_0 = np.mean(X_0, axis=0)
        mean_1 = np.mean(X_1, axis=0)
        means = [mean_0, mean_1]
        '''compute the covariance matrix'''
        # covariance = np.cov(np.transpose(X))
        covariance = np.zeros((X.shape[1], X.shape[1]))
        N_0 = np.count_nonzero(y == 0)
        # instances y=1
        N_1 = np.count_nonzero(y == 1)
        total = N_0 + N_1
        for k in [0, 1]:
            for i, row in enumerate(X):
                if y[i] == k:
                    # print(np.shape(np.transpose(row-means[k])))
                    covariance += np.multiply((row - means[k]), ((row - means[k])[np.newaxis]).T)
        covariance = covariance / (total - 2)
        # print(np.shape(covariance))


        probY_0 = N_0 / (total)


        probY_1 = N_1 / (total)
        probY = [probY_0, probY_1]

        try:
            cov_inv = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            print("matrix not invertible!")
        else:
            prediction = []
            for k in [0, 1]:
                if k == 0:
                    self.w0Y0 = (np.log((probY[k])) - (0.5) * np.transpose(means[k]).dot(cov_inv).dot(means[k]))
                    self.wY0 = (cov_inv).dot(means[k])
                else:
                    self.w0Y1 = (np.log((probY[k])) - (0.5) * np.transpose(means[k]).dot(cov_inv).dot(means[k]))
                    self.wY1 = (cov_inv).dot(means[k])



    def predict(self, X):
        prediction = []
        for i in X:
            y = []
            y.append(self.w0Y0 + np.transpose(i).dot(self.wY0))

            y.append(self.w0Y1 + np.transpose(i).dot(self.wY1))
            prediction.append(y)

        #print(y)
        return [0 if x[0] > x[1] else 1 for x in prediction]




if __name__ == '__main__':
    import load_files
    import utils
    import timeit

    start = timeit.default_timer()



    wine, wine_headers = load_files.load_wine()
    X = wine[:, :-1]
    y = wine[:, -1]

    #print(X.shape, y.shape)
    model = LDA(0,0,0,0)
    print(utils.CrossValidation(wine, model, 5))
    # model.fit(X, y)
    stop = timeit.default_timer()

    print('Time: ', stop - start)
    #print(eval.evaluate_acc(y, model.predict(X)))
    #test with sklearn
    # clf = LinearDiscriminantAnalysis()
    # clf.fit(X, y)
    # print(clf.score(X,y))