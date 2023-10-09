#from random import random
from datetime import datetime
from numpy.random import default_rng
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class ANN:

    # contains a list of matrices (array([[..., ...],
    #                                     [..., ...]])
    # each describing a layer
    w = []
    
    # it contains the output of each layer
    # this is a list of np.array
    # the first element is 1, because of biases
    out_layer = []

    # it adds an error after each fit step of the traning
    list_errors = []

    def __init__(self, n_in: int, topology: list, learning_rate: float) -> None:
        self.n_in  = n_in
        self.topology  = topology
        self.learning_rate = learning_rate
        for i, n_nodes in list(enumerate(topology)):
            if i == 0:
                a = default_rng(int(datetime.now().timestamp())).random((n_nodes,n_in + 1))
            else:
                a = default_rng(int(datetime.now().timestamp())).random((n_nodes,topology[i - 1] + 1))
            self.w.append(a)

    def forward(self, x : np.array) -> np.array:
        # if not isinstance(x, np.array):
        #     if isinstance(x, list):
        #         x = np.array(x)
        #     else:
        #         raise Exception("x must be a list of a np.array object")
        prev_out = x
        self.out_layer.append(prev_out)
        prev_out = np.insert(prev_out,0,1)
        # prev_out = np.reshape(prev_out, (prev_out.size,1))k

        for m in self.w:
            # m = np.matrix(layer)
            # print(m,prev_out,"\n")
            prev_out = np.dot(m, prev_out)
            prev_out = sigmoid(prev_out)
            self.out_layer.append(prev_out)
            prev_out = np.insert(prev_out,0,1)
            # prev_out = np.transpose(prev_out)
            # prev_out = np.reshape(prev_out, (prev_out.size,1))
        print(self.out_layer)
        return prev_out
    


    # cosa succede ai bias?
    def fit_step(self, x : np.array, exp_out : np.array):
        # self.list_errors.append(np.linalg.norm(exp_out - np.reshape(self.out_layer[-1], (1, self.out_layer[-1].size))))
        # self.list_errors.append(exp_out - self.out_layer[-1])
        self.list_errors.append(np.linalg.norm(exp_out - self.out_layer[-1]))
        #calculating the deltas
        delta = []
        exp_out = np.insert(exp_out,0,1)
        # exp_out = np.reshape(exp_out, (exp_out.size,1))
        t = -1
        for i, m in reversed(list(enumerate(self.w))):
            #output layer
            if i == len(self.w) - 1:
                delta.append(self.out_layer[i + 1] *
                             (1 - self.out_layer[i + 1]) *
                             (exp_out - self.out_layer[i + 1]))
            else: #hidden layers
                temp = []
                next_matrix = self.w[i + 1]
                for j in range(len(layer)):
                    sum = 0
                    for k in range(len(next_layer)):
                        sum += next_layer[k][j] * delta[t][k]
                    temp.append(sum * self.out_layer[i + 1] * (1 - self.out_layer[i + 1]))
                delta.append(np.array(temp))
            t += 1
        # updating weights
        for i, layer in list(enumerate(self.w)):
            for j, node in list(enumerate(layer)):
                for k, weight in list(enumerate(node)):
                    self.w[i][j][k] += self.learning_rate * self.out_layer[i][k] * delta[i][j]
                    self.w[i][j][k] = self.w[i][j][k].item()


