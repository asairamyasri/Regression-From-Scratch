import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# Load Dataset
# -----------------------------------
data = pd.read_csv("logistic.csv")

print("First 5 rows:")
print(data.head())

X = data["x"].values
Y = data["y"].values

# -----------------------------------
# Initialize Parameters
# -----------------------------------
weight = 0.0
bias = 0.0

learning_rate = 0.1
epochs = 5000
n = len(X)

# -----------------------------------
# Sigmoid Function
# -----------------------------------


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# -----------------------------------
# Binary Cross Entropy Loss
# -----------------------------------


def compute_loss(weight, bias):
    total_loss = 0

    for i in range(n):
        prediction = sigmoid(weight * X[i] + bias)

        # Prevent log(0)
        prediction = np.clip(prediction, 1e-10, 1 - 1e-10)

        total_loss += -(Y[i] * np.log(prediction) +
                        (1 - Y[i]) * np.log(1 - prediction))

    return total_loss / n


# -----------------------------------
# Gradient Descent
# -----------------------------------
for epoch in range(epochs):

    dw = 0
    db = 0

    for i in range(n):

        prediction = sigmoid(weight * X[i] + bias)

        dw += (prediction - Y[i]) * X[i]
        db += (prediction - Y[i])

    dw /= n
    db /= n

    weight -= learning_rate * dw
    bias -= learning_rate * db

    if epoch % 500 == 0:
        loss = compute_loss(weight, bias)
        print(
            f"Epoch {epoch:4d} | Loss = {loss:.5f} | Weight = {weight:.4f} | Bias = {bias:.4f}"
        )

    if abs(dw) < 1e-6 and abs(db) < 1e-6:
        print(f"\nConverged after {epoch} epochs.")
        break

# -----------------------------------
# Predictions
# -----------------------------------
predictions = []

for x in X:
    probability = sigmoid(weight * x + bias)

    if probability >= 0.5:
        predictions.append(1)
    else:
        predictions.append(0)

# -----------------------------------
# Accuracy
# -----------------------------------
correct = 0

for i in range(n):
    if predictions[i] == Y[i]:
        correct += 1

accuracy = (correct / n) * 100

print("\nTraining Complete")
print(f"Weight   : {weight:.6f}")
print(f"Bias     : {bias:.6f}")
print(f"Loss     : {compute_loss(weight, bias):.6f}")
print(f"Accuracy : {accuracy:.2f}%")

# -----------------------------------
# Plot
# -----------------------------------
plt.figure(figsize=(8, 6))

plt.scatter(
    X[Y == 0],
    Y[Y == 0],
    color="blue",
    label="Class 0"
)

plt.scatter(
    X[Y == 1],
    Y[Y == 1],
    color="red",
    label="Class 1"
)

x_values = np.linspace(min(X), max(X), 200)
probabilities = sigmoid(weight * x_values + bias)

plt.plot(
    x_values,
    probabilities,
    color="green",
    linewidth=2,
    label="Sigmoid Curve"
)

plt.title("Logistic Regression From Scratch")
plt.xlabel("X")
plt.ylabel("Probability")
plt.legend()
plt.grid(True)

plt.show()
