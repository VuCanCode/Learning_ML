import numpy as np


#### Computation Operation ####
class MatMul():
    def __call__(self,
                 A: np.ndarray,
                 B: np.ndarray):
        return np.matmul(A, B)
    
    def backward(self,
                 grad: np.ndarray,
                 A: np.ndarray,
                 B: np.ndarray):
        # print(f"a shape {A.shape}")
        # print(f"grad shape {grad.shape}")
        return np.matmul(A.T, grad), np.matmul(A.T, grad)
    
class Sigmoid():
    def __call__(self, x: np.ndarray):
        limit_value = np.clip(x, -10, 10)
        return 1 / (1 + np.exp(-limit_value))
    
    def backward(self,
                 grad,
                 Xw: np.ndarray):
        return grad
    
class Loss():
    def __call__(self, 
                 y_pred: np.ndarray,
                 y: np.ndarray,
                 X: np.ndarray):
        epsilon = 1e-12  # Small value to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(y.T @ np.log(y_pred) + (1 - y).T @ np.log(1 - y_pred))
    
    def backward(self,
                 grad: np.ndarray,
                 y_pred: np.ndarray,
                 y: np.ndarray,
                 X: np.ndarray):
        dif = y_pred - y
        return grad * dif, grad * dif, grad * dif
    