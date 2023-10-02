from random import random
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class ANN:

    # this list contains the weight for each layer and nodes
    # every element is a list of list. 
    # [[[]]] <-> layer(node(prev links))
    w = []

    # format [[]] <-> layer(list of delta for each node)
    # delta[0] contains the deltas of the last layer (which is the output layer)
    delta = []

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

    def forward(self, x : np.ndarray) -> np.ndarray:
        prev_out = x
        prev_out = np.insert(prev_out,0,1)
        prev_out = np.reshape(prev_out, (prev_out.size,1))
        for layer in self.w:
            m = np.matrix(layer)
            print(m,prev_out,"\n")
            prev_out = np.dot(m, prev_out)
            prev_out = sigmoid(prev_out)
            prev_out = np.insert(prev_out,0,1)
            prev_out = np.reshape(prev_out, (prev_out.size,1))
        print(prev_out)
        return prev_out
    
    def fit_step(self, x : np.ndarray, y : np.ndarray):
        for i, layer in reversed(list(enumerate(self.w))):
            #output layer
            if i == len(self.w) - 1:
