from datetime import datetime
from numpy.random import default_rng
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class ANN:

    # list of matrix
    # first column contains the bias
    # each row contains the weight directed to a node to the right
    # size = L
    w = []
    
    # list of size of each out_layer
    # size = L + 1
    topology = []

    # List of output of each out_layer
    # First element is the input out_layer
    # size = L + 1
    out_layer = []

    # contains the norms of (expected output - output of the network)
    list_errors = []

    def __init__(self, topology: list, learning_rate: float) -> None:
        # self.n_in  = n_in
        self.topology  = topology
        self.learning_rate = learning_rate
        for i, n_nodes in list(enumerate(topology)):
            if i == 0:
                continue
            else:
                a = default_rng(int(datetime.now().timestamp())).random((n_nodes,topology[i - 1] + 1)) * 2 - 1
            self.w.append(a)

    def forward(self, x : np.array) -> np.array:
        del self.out_layer[:]
        prev_out = x
        self.out_layer.append(prev_out)
        prev_out = np.insert(prev_out, 0, 1)
        for m in self.w:
            prev_out = np.dot(m, prev_out)
            prev_out = sigmoid(prev_out)
            self.out_layer.append(prev_out)
            prev_out = np.insert(prev_out, 0, 1)
        self.output = self.out_layer[-1]
        return self.output
    
    # m is the matrix of the weights without the biases
    def sum_deltas(self, m : np.array, delta : np.array) -> np.array:
        sums = []
        for i in range(m.shape[1]): #columns
            s = 0
            for j in range(m.shape[0]): #rows
                s += m[j][i] * delta[j]
            sums.append(s)
        return np.array(sums)
            
    def fit(self, X : np.array, exp_y : np.array, epochs = 1):
        delta = []
        for _ in range(epochs):
            del delta[:]
            for i in range(exp_y.shape[0]):
                arr = np.zeros(self.topology[-1])
                arr[int(exp_y[i])] = 1
                exp_out = arr
                self.forward(X[i])
                j = len(self.w) - 1
                if len(delta) != len(self.w):
                    delta.append(self.output * (1 - self.output) * (exp_out - self.output))
                else:
                    delta[j] += (self.output * (1 - self.output) * (exp_out - self.output))
                    j -= 1
                for l, out in reversed(list(enumerate(self.out_layer))):
                    if l == len(self.out_layer) - 1 or l == 0: continue
                    if len(delta) != len(self.w):
                        delta.append(out * (1 - out) * self.sum_deltas(self.w[l][:,1:], delta[-1]))
                    else:
                        delta[j] += out * (1 - out) * self.sum_deltas(self.w[l][:,1:], delta[j + 1])
                        j -= 1
                if j == len(self.w) - 1:
                    delta = list(reversed(delta))
            for L,m in list(enumerate(self.w)):
                for i in range(m.shape[0]): #rows
                    for j in range(m.shape[1]): #columns
                        if j == 0: #bias
                            self.w[L][i][j] += self.learning_rate * delta[L][i]
                        else:
                            self.w[L][i][j] += self.learning_rate * delta[L][i] * self.out_layer[L][j - 1]
            self.list_errors.append(np.average(delta[-1][:]))

                

    def fit_step(self, x : np.array, exp_out : np.array):
        arr = np.zeros(self.topology[-1])
        arr[int(exp_out)] = 1
        exp_out = arr
        self.forward(x)
        delta = []
        delta.append(self.output * (1 - self.output) * (exp_out - self.output))
        for l, out in reversed(list(enumerate(self.out_layer))):
            # outlayer
            if l == len(self.out_layer) - 1 or l == 0: continue
            delta.append(out * (1 - out) * self.sum_deltas(self.w[l][:,1:], delta[-1]))
        delta = list(reversed(delta))
        for L,m in list(enumerate(self.w)):
            for i in range(m.shape[0]): #rows
                for j in range(m.shape[1]): #columns
                    if j == 0: #bias
                        self.w[L][i][j] += self.learning_rate * delta[L][i]
                    else:
                        self.w[L][i][j] += self.learning_rate * delta[L][i] * self.out_layer[L][j - 1]
        
    def score(self, X_test, y_test):
        rows = np.shape(X_test)[0]
        succ = 0
        for i in range(int(rows)):
            out = self.forward(X_test[i])
            index = np.argmax(out)
            if y_test[i] == index:
                succ += 1
        return succ / rows


