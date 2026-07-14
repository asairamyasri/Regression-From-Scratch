import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Load Dataset
# -------------------------------
data = pd.read_csv("csv_file.csv")

print("First 5 rows of the dataset:")
print(data.head())

# -------------------------------
# Initialize Parameters
# -------------------------------
slope = 0.0
intercept = 0.0
learning_rate = 0.01
epochs = 3500
n = len(data)

# -------------------------------
# Gradient Functions
# -------------------------------


def gradient_slope(slope, intercept, data):
    gradient = 0

    for i in range(len(data)):
        x = data.iloc[i].x
        y = data.iloc[i].y
        prediction = slope * x + intercept
        gradient += (prediction - y) * x

    return gradient / len(data)


def gradient_intercept(slope, intercept, data):
    gradient = 0

    for i in range(len(data)):
        x = data.iloc[i].x
        y = data.iloc[i].y
        prediction = slope * x + intercept
        gradient += (prediction - y)

    return gradient / len(data)


# -------------------------------
# Mean Squared Error
# -------------------------------
def mse(slope, intercept, data):
    error = 0

    for i in range(len(data)):
        x = data.iloc[i].x
        y = data.iloc[i].y
        prediction = slope * x + intercept
        error += (prediction - y) ** 2

    return error / len(data)


# -------------------------------
# Gradient Descent
# -------------------------------
for epoch in range(epochs):

    grad_slope = gradient_slope(slope, intercept, data)
    grad_intercept = gradient_intercept(slope, intercept, data)

    slope -= learning_rate * grad_slope
    intercept -= learning_rate * grad_intercept

    # Print progress every 200 iterations
    if epoch % 200 == 0:
        loss = mse(slope, intercept, data)
        print(
            f"Epoch {epoch:4d} | Loss = {loss:.4f} | "
            f"Slope = {slope:.4f} | Intercept = {intercept:.4f}"
        )

    # Early stopping
    if abs(grad_slope) < 1e-6 and abs(grad_intercept) < 1e-6:
        print(f"\nConverged after {epoch} epochs.")
        break


# -------------------------------
# Final Results
# -------------------------------
print("\nTraining Complete")
print(f"Slope      : {slope:.6f}")
print(f"Intercept  : {intercept:.6f}")
print(f"Final Loss : {mse(slope, intercept, data):.6f}")

# -------------------------------
# Plot Results
# -------------------------------
plt.figure(figsize=(8, 6))

plt.scatter(data.x, data.y, label="Data Points")

predicted_y = slope * data.x + intercept

plt.plot(
    data.x,
    predicted_y,
    color="red",
    linewidth=2,
    label="Regression Line"
)

plt.title("Linear Regression using Gradient Descent (From Scratch)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)

plt.show()
