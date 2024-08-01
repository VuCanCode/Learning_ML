from utility.nodes import (ComputationNetwork,
                    ComputeNode,
                    InputNode,
                    ParameterNode)

from utility.comp import (MatMul,
                     Sigmoid,
                     Loss)
import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression():
    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray):

        self.X = self.convert_to_cpt(X) # add a col of ones
        self.y = y
        self.w = None
        self.model = ComputationNetwork()

    def convert_to_cpt(self, X: np.ndarray):
        ones = np.ones((X.shape[0], 1))
        return np.hstack((X, ones))


    def fit(self,
            epoch: int = 1000,
            step_size: float = 1,
            is_viz: bool = True):
        
        # initialize computation network
        # input and parameters
        X = InputNode(self.X)
        y = InputNode(self.y)
        self.w = ParameterNode(np.ones((X.value.shape[1], 1)))

        # construct computation network
        self.mul = ComputeNode(MatMul(), X, self.w)
        self.model.add_node(self.mul)
        self.sig = ComputeNode(Sigmoid(), self.mul)
        self.model.add_node(self.sig)
        self.loss = ComputeNode(Loss(), self.sig, y)
        self.model.add_node(self.loss)

        prev_loss = 100000
        loss = 100
        i = 0
        losses = []

        # train
        while i < epoch and prev_loss - loss > 1e-4:
            self.model.forward()
            self.model.backward()
            self.w.value -= step_size * self.w.grad
            prev_loss = loss
            loss = self.loss.value[0]
            losses.append(loss[0,0])
            print(f"epoch {i}, loss {loss[0,0]}")
            i+=1

        if is_viz:
            plt.plot(range(i), losses, label="Losses")
            plt.title("Losses vs Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid()
            plt.show()


    def predict(self, X):
        X = self.convert_to_cpt(X)
        X = InputNode(X)
        model = ComputationNetwork()
        # construct computation network
        mul = ComputeNode(MatMul(), X, self.w)
        model.add_node(mul)
        sig = ComputeNode(Sigmoid(), mul)
        model.add_node(sig)
        prediction = model.predict()
        return prediction > 0.5