import numpy as np
import scipy.stats as st

class BayesClassifier:
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.fit()

    def fit(self):
        num_samples = self.y.shape[0]
        num_fetures = self.X.shape[1]
        self.p_y = {}
        self.normal = {}
        unique_array, frequency = np.unique(self.y, return_counts = True)
        for j, y in list(enumerate(unique_array)):
            self.p_y[y] = frequency[j] / num_samples
            temp = []
            del temp[:]
            for i in range(num_samples):
                if self.y[i] == y:
                    temp.append(self.X[i])
            temp = np.array(temp)
            self.normal[y] = (np.mean(temp, axis=0), np.std(temp, axis=0))

    def score(self, X_train, y_train):
        count = 0
        for i, y in list(enumerate(y_train)):
            pred = self.predict(X_train[i])
            if pred == y: count += 1
        return count / y_train.shape[0]

    def predict(self, x):
        index = -1
        max = -1
        for key, value in self.p_y.items():
            molt = 1
            means, stds = self.normal[key]
            for i, mean in list(enumerate(means)):
                molt *= st.norm.pdf(x[i], mean, stds[i])
            molt *= value
            if molt > max:
                max = molt
                index = key
        return index

        
