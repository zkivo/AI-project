import numpy as np
from numpy.random import default_rng

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class ANN:

    # list of matrix
    # first column contains the bias
    # each row contains the weight directed to a node to the right
    # size = L
    w = []
    
    # it contains the output of each layer, even the input layer
    # size = L + 1 (one is the input layer)
    layer = []

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
                #a = default_rng(int(datetime.now().timestamp())).random((n_nodes,topology[i - 1] + 1))
                a = default_rng(69).random((n_nodes,topology[i - 1] + 1))
            self.w.append(a)
            ##print(self.w[i][:,0:])

    def forward(self, x : np.array) -> np.array:
        del self.layer[:]
        prev_out = x.T
        self.layer.append(prev_out)
        prev_out = np.insert(prev_out,0,1)
        for m in self.w:
            prev_out = np.dot(m, prev_out)
            prev_out = sigmoid(prev_out)
            self.layer.append(prev_out)
            prev_out = np.insert(prev_out,0,1).T
        self.output = self.layer[-1]
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
            

    def fit_step(self, x : np.array, exp_out : np.array):
        self.forward(x)
        #self.list_errors.append(np.linalg.norm(exp_out - self.output)**2)
        delta = []
        delta.append(self.output * (1 - self.output) * (exp_out - self.output))
        for i, out in reversed(list(enumerate(self.layer))):
            # outlayer
            if i == len(self.layer) - 1 or i == 0: continue
            # for j, node_out in out:
            #     delta.append(node_out * (1 - out) * )
            delta.append(out * (1 - out) * self.sum_deltas(self.w[i][:,1:], delta[-1]))
        # print(delta)
        delta = list(reversed(delta))
        for L,m in list(enumerate(self.w)):
            for i in range(m.shape[0]): #rows
                for j in range(m.shape[1]): #columns
                    if j == 0: #bias
                        self.w[L][i][j] += self.learning_rate * delta[L][i]
                    else:
                        self.w[L][i][j] += self.learning_rate * delta[L][i] * self.layer[L][j - 1]


