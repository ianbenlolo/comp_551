import numpy as np

class BernoulliNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def fit(self, X, y):
        sample_count = y.shape[0]
        # group by class (optimize?)
        class_samples = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
        
        self.theta_k = np.array([(y == c).sum() / sample_count for c in range(0, max(y)+1)])
        
        # Sum of words in class (optimize?)
        count_j_k = np.array([np.array(i).sum(axis=0).toarray() for i in class_samples]) + self.alpha
        
        laplace_smoothing = 2 * self.alpha
        #number of subreddit occurences + smoothing (optimize?)
        Ycount = np.array([(y == c).sum() + laplace_smoothing for c in range(0, max(y)+1)])
        
        # probability of each word
        self.theta_j_k = np.array(count_j_k.reshape(20,60093) / Ycount[:,None])
        
        return self
    
    
    def predict(self, X):
        feature_likelihood = X.dot(np.log(self.theta_j_k).T) + (1 - X).dot(np.log(1-self.theta_j_k).T)
        class_prob = np.log(self.theta_k) + feature_likelihood
        
        return np.argmax(class_prob, axis=1)