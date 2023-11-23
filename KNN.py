from sortedcontainers import SortedDict
import numpy as np
import math

class KNN:

    def __init__(self, X_train, y_train, k = 1) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.k = k

    def set_k(self, k):
        self.k = k

    def predict(self, X):
        dists = SortedDict()
        rows = np.shape(self.X_train)[0]
        classes = []
        for i in range(int(rows)):
            dist = math.sqrt(np.sum((X - self.X_train[i])**2))
            dists[dist] = self.y_train[i]
        for j in range(self.k):
            classes.append(dists.peekitem(j)[1])
        classes = np.array(classes)
        unique, frequency = np.unique(classes, return_counts = True)
        return unique[frequency.argmax()]
    
    def score(self, X_test, y_test):
        rows = np.shape(X_test)[0]
        succ = 0
        for i in range(int(rows)):
            pre_y = self.predict(X_test[i])
            if pre_y == y_test[i]:
                succ += 1
        return succ / rows




