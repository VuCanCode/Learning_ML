#%%
import numpy as np
from model import MLP
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model import onehot
# Example usage
# X = np.random.randn(100, 10)
# y = np.random.randn(100, 1)
df = load_wine()
X = df["data"]
y = df["target"]
#%%

scaler = StandardScaler()
model = scaler.fit(X)
scaled_data = model.transform(X)

X_train, X_test, y_train, y_test = train_test_split(scaled_data,
                                                    y,
                                                    test_size=30,
                                                    train_size=70,
                                                    random_state=8)


mlp = MLP(input_dim=X_train.shape,
          hidden_dim=50,
          output_dim=3,
          n_classes=3)

losses = []
pre_loss = 1000000
current_loss = 1000
i = 0
epoch = 500
test_accs = []

while i <= epoch and pre_loss - current_loss > 1e-5:
    pre_loss = current_loss
    current_loss = mlp.forward(X_train, y_train)
    mlp.backward()
    mlp.update(learning_rate=1e-1)
    losses.append(current_loss)
    if i % 100 == 0:
        print(f"Epoch {i}, current_loss: {current_loss}")

    y_pred = mlp.predict(X_test)
    # print(y_pred)
    test_acc = accuracy_score(y_pred, y_test)
    # print(f"test acc {test_acc}")
    test_accs.append(test_acc)
    i += 1

plt.plot(range(len(losses)), test_accs, label="Test Acc")
plt.plot(range(len(losses)), losses, label="Loss")
plt.title("Loss and Test Accuracy vs Epoch")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid()
plt.legend()
plt.show()

y_pred = mlp.predict(X_train)
print(f"prediction {y_pred}")
print(f"gt {y_train}")
print(f"train acc {accuracy_score(y_pred, y_train)}")

y_pred = mlp.predict(X_test)
print(y_pred)
print(f"test acc {accuracy_score(y_pred, y_test)}")
#%%