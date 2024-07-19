#%%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings


# %%
def get_corr_matrix(data: pd.DataFrame,
                    target: pd.DataFrame) -> pd.DataFrame:
    
    # visualize correlation matrix
    df = data.copy()
    df["result"] = target
    corr_matrix = df.corr()

    return corr_matrix
# %%
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



# %%
def sigmoid(y_pred: np.matrix) -> float:
    y_pred = np.maximum(-10, np.minimum(10, y_pred)) # avoid overflow
    value =  1 / (1 + np.exp(-y_pred)) 
    return value

def predict(x: np.matrix,
            w: np.matrix) -> np.matrix:
    return sigmoid(x @ w)

def get_grad(y_pred: np.matrix,
             y: np.matrix,
             x: np.matrix,
             w: np.matrix,
             w_scale: float = 1e-3) -> np.matrix:
    grad = (x.T @ (y_pred - y) + w_scale * w)
    return grad

def get_loss(y_pred: np.matrix,
             y: np.matrix,
             w: np.matrix,
             w_scale: float = 1e-3) -> float:
    loss = (y_pred - y).T @ (y_pred - y) + w_scale * w.T @ w
    return loss

def activate(y: np.ndarray):
    return (y > 0.5) * 1

def fit(X_train: np.matrix,
        X_test: np.matrix,
        y_train: np.ndarray,
        y_test: np.ndarray,
        is_loss_viz: bool = True) -> np.matrix:
    
    # initializing variables
    epoch = 0
    epoch_lim = 1000
    grad_lim = 10
    grad = 10000
    step_size = 1e-1
    loss = 10000
    w_scale = 1e-3
    losses = []

    # convert data type and init weight
    y = np.matrix(y_train).T
    w = np.matrix(np.ones(X_train.shape[1])).T

    while np.linalg.norm(grad) > grad_lim and epoch < epoch_lim:
        y_pred = predict(X_train, w)
        grad = get_grad(y_pred=y_pred,
                        y=y,
                        x=X_train,
                        w=w, w_scale=w_scale)
        loss = get_loss(y_pred, y, w, w_scale=w_scale)
        w -= step_size * grad
        losses.append(loss[0,0])
        epoch += 1

        print(f"epoch: {epoch}, loss: {loss[0,0]}, grad norm: {np.linalg.norm(grad)}")

    if is_loss_viz:
        plt.plot(np.arange(len(losses)), losses)
        plt.title("Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()
        plt.show()

    return w

def convert_to_cpt(X: np.ndarray) -> np.matrix:
    # convert data to compact notation
    X = np.matrix(X)
    X = np.hstack([X, np.matrix(np.ones(X.shape[0])).T])
    return X


if __name__ == "__main__":
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
                                        random_state=10)

    # %%
    # convert to compact notation
    X_train = convert_to_cpt(X_train)
    X_test = convert_to_cpt(X_test)

    # train model
    w = fit(X_train, X_test, y_train, y_test)

    # evaluation for training
    y_train_pred = activate(np.array(predict(X_train, w)))
    accuracy_train = accuracy_score(y_train, y_train_pred)
    print(f"training accuracy: {accuracy_train}")

    # %%
    # evaluation for testing
    y_test_pred = activate(np.array(predict(X_test, w)))
    accuracy_test = accuracy_score(y_test, y_test_pred)
    print(f"testing accuracy: {accuracy_test}")
