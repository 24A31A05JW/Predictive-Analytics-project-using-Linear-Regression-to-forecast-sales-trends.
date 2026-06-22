import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("data/Advertising Budget and Sales.csv")

# Features
X = df[['TV Ad Budget ($)',
        'Radio Ad Budget ($)',
        'Newspaper Ad Budget ($)']]

# Target
y = df['Sales ($)']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully!")

# Predictions
predictions = model.predict(X_test)

print(predictions[:5])

# Accuracy
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# Save Predictions
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})

results.to_csv("predictions.csv", index=False)

# Visualization
plt.figure(figsize=(8,6))
plt.scatter(y_test, predictions)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

plt.savefig("images/prediction_graph.png")

print("Predictions saved successfully!")

plt.show()