import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = pd.read_csv("data.csv")

# Features and label
X = data[["soil_moisture", "temperature"]]
y = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Check accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Take user input
print("\n--- Smart Agriculture Prediction System ---")

soil = float(input("Enter soil moisture: "))
temp = float(input("Enter temperature: "))

input_data = pd.DataFrame([[soil, temp]], columns=["soil_moisture", "temperature"])
prediction = model.predict(input_data)

if prediction[0] == 1:
    print("Result: Water required → Pump ON")
else:
    print("Result: No water required → Pump OFF")

# Predict
import pandas as pd

input_data = pd.DataFrame([[soil, temp]], columns=["soil_moisture", "temperature"])
prediction = model.predict(input_data)

if prediction[0] == 1:
    print("Pump ON (Water needed)")
else:
    print("Pump OFF (No water needed)")

#graph
import matplotlib.pyplot as plt

plt.scatter(data["soil_moisture"], data["temperature"], c=data["label"])
plt.xlabel("Soil Moisture")
plt.ylabel("Temperature")
plt.title("Smart Agriculture Decision Boundary")
plt.colorbar(label="Water Needed (1) / Not Needed (0)")
plt.show()


