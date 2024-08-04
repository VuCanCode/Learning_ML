from utility.nodes import (ParameterNode,
                    InputNode,
                    ComputeNode)
from utility.comp import (ReLU,
                         MatMul,
                         CrossEntropyLoss,
                         Softmax)
import numpy as np
from sklearn.preprocessing import LabelBinarizer

def onehot(y: np.ndarray):
    Encoder = LabelBinarizer()
    Encoder.fit(np.arange(max(y) + 1))
    return Encoder.transform(y)


class MLP:
    def __init__(self,
                input_dim: tuple,
                hidden_dim: tuple,
                output_dim: tuple,
                n_classes: int):
        self.nodes = []
        self.W1 = ParameterNode(np.random.normal(loc=0,
                                                 scale=2/input_dim[1],
                                                 size=(input_dim[1] + 1, hidden_dim)))
        print(self.W1.value.shape)
        self.W2 = ParameterNode(np.random.normal(loc=0,
                                                 scale=2/hidden_dim,
                                                 size=(hidden_dim + 1, n_classes)))

    def add_node(self, node):
        self.nodes.append(node)
        

    def forward(self, X, y):

        # convert y to onehot array
        y_onehot = onehot(y)

        self.X = InputNode(X)
        self.y = InputNode(y_onehot)

        self.z1 = ComputeNode(MatMul(), self.X, self.W1)
        self.nodes.append(self.z1)
        self.relu1 = ComputeNode(ReLU(), self.z1)
        self.nodes.append(self.relu1)
        self.z2 = ComputeNode(MatMul(), self.relu1, self.W2)
        self.nodes.append(self.z2)
        self.softmax = ComputeNode(Softmax(), self.z2)
        self.nodes.append(self.softmax)
        self.loss = ComputeNode(CrossEntropyLoss(), self.softmax, self.y)
        self.nodes.append(self.loss)


        # Forward pass
        for node in self.nodes:
            node.forward()


        return self.loss.value

    def backward(self):
        # Initialize gradients
        self.loss.grad = 1

        # Backward pass
        for node in reversed(self.nodes):
            node.backward()

    def update(self, learning_rate: float = 1e-2):
        self.W1.value -= learning_rate * self.W1.grad
        self.W2.value -= learning_rate * self.W2.grad

    
    def predict(self, X: np.ndarray):
        nodes = []
        X = InputNode(X)
        z1 = ComputeNode(MatMul(), X, self.W1)
        nodes.append(z1)
        h1 = ComputeNode(ReLU(), z1)
        nodes.append(h1)
        z2 = ComputeNode(MatMul(), h1, self.W2)
        nodes.append(z2)
        h2 = ComputeNode(Softmax(), z2)
        nodes.append(h2)

        # predict
        for node in nodes:
            node.forward()

        # convert back to original form
        result = [np.argmax(predict) for predict in h2.value]

        return result
        
