import numpy as np


#### Computation Operation ####
class MatMul():
    def __call__(self,
                 A: np.ndarray,
                 B: np.ndarray):
        A = np.hstack((A, np.ones((A.shape[0], 1)))) # cpt notation to add bias
        return np.matmul(A, B)
    
    def backward(self,
                 grad: np.ndarray,
                 A: np.ndarray,
                 B: np.ndarray):
        B = B[:-1, :] # cpt notation to add bias
        A = np.hstack((A, np.ones((A.shape[0], 1)))) # cpt notation to add bias
        return np.dot(grad, B.T), np.dot(A.T, grad)
    
class Sigmoid():
    def __call__(self, x: np.ndarray):
        limit_value = np.clip(x, -10, 10)
        return 1 / (1 + np.exp(-limit_value))
    
    def backward(self,
                 grad,
                 Xw: np.ndarray):
        return np.dot(grad, 1 / (1 + np.exp(-Xw)))
    
    

class ReLU():
    def __call__(self, x: np.ndarray):
        return np.maximum(0, x)
    
    def backward(self,
                 grad: np.ndarray,
                 x: np.ndarray):
        return grad * (x > 0)
    

#%%
class Softmax:
    def __call__(self, X):
        # print("++++++++++ Soft Max In +++++++++++")
        # print(X)
        shifted_X = X - np.max(X, axis=1, keepdims= True)
        exp_X = np.exp(shifted_X)
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)

    def backward(self, grad, X):
        return grad

    
class CrossEntropyLoss:
    def __call__(self, y_pred, y):
        # y_pred = np.clip(y, 1e-6, 1 - 1e-6)
        epsilon = 1e-15
        
        # Clip predictions to avoid log(0) or log(1)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Compute cross-entropy loss for each sample
        loss = -np.sum(y * np.log(y_pred), axis=1)
        
        return np.mean(loss)

    def backward(self, grad, y_pred, y):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        m = y.shape[0]
        return -grad * (y - y_pred) / m, -grad * (y_pred - y) / m

    