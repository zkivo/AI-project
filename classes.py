from random import random
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class ANN:

    # this list contains the weight and biases for each layer and nodes
    # every element is a list of list. 
    # [[[]]] <-> layer(node(prev links))
    w = []

    # it contains the output of each layer
    # this is a list of np.array
    # the first element is 1, because of biases
    out_layer = []

    # it adds an error after each fit step of the traning
    list_errors = []

    def __init__(self, n_in: int, n_out: int, 
                 topology: list, learning_rate: int) -> None:
        self.n_in  = n_in
        self.n_out = n_out
        self.topology      = topology
        self.learning_rate = learning_rate
        i = 0
        for n_nodes in topology:
            nodes = []
            for _ in range(n_nodes):
                links = []
                prev_n_nodes = n_in
                if i > 0:
                    prev_n_nodes = len(self.w[-1])
                # plus one because we treat the bias as a weight
                for _ in range(prev_n_nodes + 1):
                    links.append(random())
                nodes.append(links)
            self.w.append(nodes)
            i += 1
        nodes = []
        for _ in range(n_out):
            links = []
            if len(topology) > 0:
                for _ in range(len(self.w[-1]) + 1):
                    links.append(random())
            else:
                for _ in range(n_in + 1):
                    links.append(random())
            nodes.append(links)
        self.w.append(nodes)

    def forward(self, x : np.array) -> np.ndarray:
        prev_out = x
        prev_out = np.insert(prev_out,0,1)
        prev_out = np.reshape(prev_out, (prev_out.size,1))
        self.out_layer.append(prev_out)
        for layer in self.w:
            m = np.matrix(layer)
            print(m,prev_out,"\n")
            prev_out = np.array(np.dot(m, prev_out))
            prev_out = sigmoid(prev_out)
            prev_out = np.insert(prev_out,0,1)
            # prev_out = np.transpose(prev_out)
            prev_out = np.reshape(prev_out, (prev_out.size,1))
            self.out_layer.append(prev_out)
        print(self.out_layer[-1])
        return prev_out
    


    # cosa succede ai bias?
    def fit_step(self, x : np.ndarray, exp_out : np.ndarray):
        self.list_errors.append(np.linalg.norm(exp_out - np.reshape(self.out_layer[-1], (1, self.out_layer[-1].size))))
        #calculating the deltas
        delta = []
        exp_out = np.insert(exp_out,0,1)
        exp_out = np.reshape(exp_out, (exp_out.size,1))
        t = -1
        for i, layer in reversed(list(enumerate(self.w))):
            #output layer
            if i == len(self.w) - 1:
                delta.append(self.out_layer[i + 1] *
                             (1 - self.out_layer[i + 1]) *
                             (exp_out - self.out_layer[i + 1]))
            else: #hidden layers
                temp = []
                next_layer = self.w[i + 1]
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

