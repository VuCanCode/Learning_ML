import numpy as np

### Nodes ###
class Node:
    def __init__(self):
        self.value = None
        self.grad = None
        self.parents = None

    def forward(self):
        pass

    def backward(self):
        pass

class InputNode(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self):
        pass

    def backward(self):
        pass

class ParameterNode(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self):
        pass

    def backward(self):
        pass

class ComputeNode(Node):
    def __init__(self, operation, *parents):
        super().__init__()
        self.operation = operation
        self.parents = parents

    def forward(self):
        self.value = self.operation(*[parent.value for parent in self.parents])

    def backward(self):
        child_grad = self.grad
        parent_grads = self.operation.backward(child_grad, *[p.value for p in self.parents])
        if len(self.parents) > 1:
            for parent, grad in zip(self.parents, parent_grads):
                parent.grad = grad
                try:
                    print(f"CURRENT OPERATION {self.operation}")
                    print(f"PARENT OPERATION {parent.operation}")
                    print(parent.grad.shape)
                    print(f"self grad {self.grad.shape}")
                except:
                    print("input node or para node")
        else:
            self.parents[0].grad = parent_grads

### Computation Network hold all the nodes together ###
class ComputationNetwork():
    def __init__(self):
        self.nodes = []


    def add_node(self, node: Node):
        self.nodes.append(node)

    def forward(self):
        for node in self.nodes:
            node.forward()

        return node.value # last node is the loss node
    
    def backward(self):
        self.nodes[-1].grad = 1
        for node in reversed(self.nodes):
            node.backward()

    def predict(self):
        for node in self.nodes[:-1]:
            node.forward()

        return node.value
        

