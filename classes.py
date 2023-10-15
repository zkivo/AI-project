#from random import random
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
    
    # it contains the output of each layer, even the input layer
    # size = L + 1 (one is the input layer)
    out_layer = []

    # contains the norms of (expected output - output of the network)
    list_errors = []

    def __init__(self, n_in: int, topology: list, learning_rate: float) -> None:
        self.n_in  = n_in
        self.topology  = topology
        self.learning_rate = learning_rate
        for i, n_nodes in list(enumerate(topology)):
            if i == 0:
                # a = default_rng(int(datetime.now().timestamp())).random((n_nodes,n_in + 1))
                a = default_rng(69).random((n_nodes,n_in + 1))
            else:
                #a = default_rng(int(datetime.now().timestamp())).random((n_nodes,topology[i - 1] + 1))
                a = default_rng(69).random((n_nodes,topology[i - 1] + 1))
            self.w.append(a)
            ##print(self.w[i][:,0:])

    def forward(self, x : np.array) -> np.array:
        del self.out_layer[:]
        prev_out = x.T
        self.out_layer.append(prev_out)
        prev_out = np.insert(prev_out,0,1)
        for m in self.w:
            prev_out = np.dot(m, prev_out)
            prev_out = sigmoid(prev_out)
            self.out_layer.append(prev_out)
            prev_out = np.insert(prev_out,0,1).T
        self.output = self.out_layer[-1]
        return prev_out
    
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
        delta = []
        delta.append(self.output * (1 - self.output) * (exp_out - self.output))
        for i, out in reversed(list(enumerate(self.out_layer))):
            # outlayer
            if i == len(self.out_layer) - 1: continue
            # for j, node_out in out:
            #     delta.append(node_out * (1 - out) * )
            delta.append(out * (1 - out) * self.sum_deltas(self.w[i][:,1:], delta[-1]))
        #print(delta)
        delta = list(reversed(delta))
        for L,m in list(enumerate(self.w)):
            # weight = m[:,1:]
            # bias = m[:,0]
            for i in range(m.shape[0]): #rows
                for j in range(m.shape[1]): #columns
                    if j == 0: #bias
                        self.w[L][i][j] += self.learning_rate * delta[L][i]
                    else:
                        self.w[L][i][j] += self.learning_rate * delta[L][i] * self.out_layer[L][j - 1]
                        
    # cosa succede ai bias?
    # def fit_step(self, x : np.array, exp_out : np.array):
    #     # self.list_errors.append(np.linalg.norm(exp_out - np.reshape(self.out_layer[-1], (1, self.out_layer[-1].size))))
    #     # self.list_errors.append(exp_out - self.out_layer[-1])
    #     self.list_errors.append(np.linalg.norm(exp_out - self.out_layer[-1]))
    #     #calculating the deltas
    #     delta = []
    #     exp_out = np.insert(exp_out,0,1)
    #     # exp_out = np.reshape(exp_out, (exp_out.size,1))
    #     t = -1
    #     for i, m in reversed(list(enumerate(self.w))):
    #         #output layer
    #         if i == len(self.w) - 1:
    #             delta.append(self.out_layer[i + 1] *
    #                          (1 - self.out_layer[i + 1]) *
    #                          (exp_out - self.out_layer[i + 1]))
    #         else: #hidden layers
    #             temp = []
    #             next_matrix = self.w[i + 1]
    #             for j in range(len(layer)):
    #                 sum = 0
    #                 for k in range(len(next_layer)):
    #                     sum += next_layer[k][j] * delta[t][k]
    #                 temp.append(sum * self.out_layer[i + 1] * (1 - self.out_layer[i + 1]))
    #             delta.append(np.array(temp))
    #         t += 1
    #     # updating weights
    #     for i, layer in list(enumerate(self.w)):
    #         for j, node in list(enumerate(layer)):
    #             for k, weight in list(enumerate(node)):
    #                 self.w[i][j][k] += self.learning_rate * self.out_layer[i][k] * delta[i][j]
    #                 self.w[i][j][k] = self.w[i][j][k].item()


