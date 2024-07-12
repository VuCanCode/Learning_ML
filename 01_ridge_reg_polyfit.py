#%%
import numpy as np
import matplotlib.pyplot as plt


def gen_poly_func(gt_coef: np.ndarray,
                  x: np.ndarray) -> float:
    order = len(gt_coef) - 1
    result = 0
    for coef in gt_coef:
        result += coef * x ** order
        order -= 1

    return result
    

def gen_data(func,
             input: np.ndarray) -> np.ndarray:
    # generate input
    x = input.copy()
    noise = np.random.normal(loc=0, scale=0.5, size=len(x))
    y = func(x) + noise * x.max()
    # y = func(x)
    return y

def predict(weight: np.ndarray,
            x: np.ndarray) -> float:
    weight = np.matrix(weight)
    x = np.matrix(x)
    y_pred = x @ weight
    return y_pred

def get_l2_loss(y_pred: np.ndarray,
                y: np.ndarray,
                weight: np.ndarray,
                pen: float) -> float:
    l2_loss = (y_pred - y).T @ (y_pred - y) + pen * weight.T @ weight
    return l2_loss

def get_grad(weight:np.ndarray,
             y: np.ndarray,
             x_train: np.ndarray,
             pen: float) -> np.ndarray:
    # l2 grad formula
    grad = 2 * x_train.T @ (predict(weight, x_train) - y) + 2 * pen * weight
    return grad

def train(order: int, 
          x: np.ndarray,
          y:np.ndarray,
          pen: float = 1e-8):
    # put x in right format and cpt notation
    x_train = np.ones(len(x)).T
    for i in range(1, order + 1):
        x_train = np.vstack([x_train, x**i])
    x_train = x_train.T
    # initial weight
    w = np.matrix(np.random.randint(-5,5,len(x_train[0])), dtype=float).T
    y = np.matrix(y).T
    print(f"y shape {y.shape}")
    print(f"x shape {x_train.shape}")
    print(f"weight shape {w.shape}")
    # initial loss
    pre_loss = 1000000
    cur_loss = pre_loss - 100
    losses = []
    # epoch to limit forever running time
    lim = 2000
    epoch = 0
    step_size = 1e-5
    
    while abs(pre_loss - cur_loss) > 1e-6 and epoch < lim:
        y_pred = predict(weight=w, x=x_train)
        # print(f"y_pred shape {y_pred.shape}")
        pre_loss = cur_loss
        cur_loss = get_l2_loss(y_pred=y_pred,
                               y=y,
                               weight=w,
                               pen=pen)[0, 0]
        grad = get_grad(weight=w,
                        y=y,
                        x_train=x_train,
                        pen=pen)
        # print(f"grad shape {(step_size * grad).shape}")
        w -= step_size * grad
        losses.append(cur_loss)
        epoch += 1
        print(f"epoch {epoch}, curr loss {cur_loss}")
    
    # visualize loss function
    plt.plot(range(epoch), losses, c='r', marker='o')
    plt.title("loss vs epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid()
    plt.show()
    return w, losses[-1]


if __name__ == "__main__":
    # generate data
    order = 5
    x = np.arange(0,2,0.01, dtype=float)
    # gt_coef = np.random.randint(-5, 5, order + 1)
    # y_func = lambda x : gen_poly_func(gt_coef, x)
    y_func = lambda x : 4 * np.sin(3 * x) + 8
    y = gen_data(y_func, x)

    # train model to fit
    weight, loss = train(order = 6, x=x, y=y, pen=1e-3)
    print(f"predicted weight {weight[::-1]}")
    # print(f"gt weight {gt_coef}")


    # Visualize data
    plt.scatter(x, y, label = "raw data")
    plt.plot(x, y_func(x), label = "ground truth")
    plt.plot(x, weight[0,0] + weight [1, 0] * x +
                weight[2,0] * x**2 + weight[3,0] * x**3 +
                weight[4,0] * x**4 + weight[5,0] * x**5 +
                weight[6,0] * x**6,
                label = "predicted curve")
    plt.title("Data")
    plt.legend()
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

# %%
