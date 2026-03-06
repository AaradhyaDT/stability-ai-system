import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Generate synthetic dataset
np.random.seed(42)
data_size = 1000

tilt_x = np.random.uniform(-10, 10, data_size)
tilt_y = np.random.uniform(-10, 10, data_size)
angular_velocity = np.random.uniform(0, 5, data_size)

# Simple rule for labeling
labels = [
    1 if abs(x) < 5 and abs(y) < 5 and av < 3 else 0
    for x, y, av in zip(tilt_x, tilt_y, angular_velocity)
]

df = pd.DataFrame({
    "tilt_x": tilt_x,
    "tilt_y": tilt_y,
    "angular_velocity": angular_velocity,
    "stable": labels
})

# Save dataset
df.to_csv("data/synthetic_sensor_data.csv", index=False)

X = df[["tilt_x", "tilt_y", "angular_velocity"]]
y = df["stable"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print("Model Accuracy:", accuracy)

# Save model
joblib.dump(model, "model/stability_model.joblib")
print("Model saved to model/stability_model.joblib")