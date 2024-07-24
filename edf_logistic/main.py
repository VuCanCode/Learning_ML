#%%

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

from utility.nodes import (ComputationNetwork,
                    ComputeNode,
                    InputNode,
                    ParameterNode)

from utility.comp import (MatMul,
                     Sigmoid,
                     Loss)

#%%

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
            step_size: float = 1):
        
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
        self.loss = ComputeNode(Loss(), self.sig, y, X)
        self.model.add_node(self.loss)

        prev_loss = 0
        loss = 100000
        i = 0
        # train
        while i < epoch and abs(loss - prev_loss) > 1e-6:
            self.model.forward()
            self.model.backward()
            self.w.value -= step_size * self.w.grad
            print("weight")
            print(self.w.value)
            prev_loss = loss
            loss = self.loss.value[0,0]
            print(f"epoch {i}, loss {loss}")
            i+=1

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


    # def predict(self, X: np.ndarray):

def get_corr_matrix(data: pd.DataFrame,
                    target: pd.DataFrame) -> pd.DataFrame:
    
    # visualize correlation matrix
    df = data.copy()
    df["result"] = target
    corr_matrix = df.corr()

    return corr_matrix

def drop_low_corr(data: np.ndarray,
                  target: np.ndarray,
                  corr_lim: float = 0.1) -> pd.DataFrame:
    # drop low correlation features
    corr_matrix = get_corr_matrix(data=data,
                                target=target)
    low_correlation = corr_matrix["result"][abs(corr_matrix["result"]) < corr_lim]
    for col in low_correlation.keys():
        data.drop(col, axis = 1)

    return data

#%%
# if __name__ == "__main__":
# Load breast cancer data
data, target = load_breast_cancer(return_X_y=True,
                                as_frame=True)

# drop low correlated data
high_corr_data = drop_low_corr(data, target, corr_lim=0.1)

# scaling data
scaler = StandardScaler()
model = scaler.fit(high_corr_data)
scaled_data = model.transform(high_corr_data)


# split the train and test dataset
X_train, X_test,\
    y_train, y_test = train_test_split(scaled_data, target,
                                    test_size=0.30,
                                    random_state=20)

#%%
# train model
y_train = np.resize(y_train, (y_train.shape[0], 1))
LogReg = LogisticRegression(X_train, y_train)
LogReg.fit()

# evaluation for training
y_train_pred = LogReg.predict(X_train)
accuracy_train = accuracy_score(y_train, y_train_pred)
print(f"training accuracy: {accuracy_train}")

# %%
# evaluation for testing
print('TESTING TESTING')
y_test_pred = LogReg.predict(X_test)
accuracy_test = accuracy_score(y_test, y_test_pred)
print(f"testing accuracy: {accuracy_test}")

